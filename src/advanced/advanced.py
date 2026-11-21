"""Evaluate and deploy spatially calibrated PatchCore and DINOv2 hybrids."""

import argparse
from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from zipfile import ZIP_STORED, ZipFile

import numpy as np
import torch
from torch.nn import functional as F

from balanced_maps import adjust_scores
from benchmark import Localization, measure
from deployment import create_encoder, encode_tensors
from dual_head import DENSE, IMAGE, DualHead, embeddings
from experiment import WORK, Scorer, array_save, save_csv, start_device
from inference_experiment import test_paths
from patchcore_budget.io import sha256
from patchcore_budget.metrics import binary_auroc
from raw_scores import (HEADS, RESULTS, CosineBank, dino_path, encode_dino,
                        make_dino, manifest_parts, views, wrn_path)
from robust_maps import image_score
from summarize import CATEGORIES


@dataclass(frozen=True)
class Recipe:
    maps: dict[str, float]
    image: dict[str, float]
    spatial: float = .40
    dino_contrast: float = .30
    fine_window: int = 37
    fine_contrast: float = .75
    normalize_spread: bool = True
    gaussian: bool = True

    @property
    def heads(self):
        return tuple(name for name in HEADS if name in self.maps or name in self.image)


REFERENCE = 'dual-fusion25'
GPU_REPLAY = 'wrn-gpu-replay'
RECIPES = {
    'wrn-spatial': Recipe({'fine': .75, 'coarse': .25}, {'coarse': 1.}),
    'wrn-spatial-wide': Recipe({'fine': .65, 'coarse': .35}, {'coarse': .90, 'fine': .10},
                               spatial=.70, fine_window=47),
    'dino-mid': Recipe({'mid': 1.}, {'mid': 1.}, spatial=.30),
    'dino-deep': Recipe({'deep': 1.}, {'deep': 1.}, spatial=.30),
    'dino-two': Recipe({'mid': .60, 'deep': .40}, {'mid': .35, 'deep': .65}),
    'hybrid-compact-mid': Recipe({'fine': .60, 'mid': .40}, {'mid': 1.}),
    'hybrid-compact-deep': Recipe({'fine': .60, 'deep': .40}, {'deep': 1.}),
    'hybrid25': Recipe({'fine': .55, 'coarse': .20, 'deep': .25}, {'coarse': .35, 'deep': .65}, spatial=.25),
    'hybrid45': Recipe({'fine': .40, 'coarse': .15, 'deep': .45}, {'coarse': .35, 'deep': .65}, spatial=.40),
    'hybrid-mid45': Recipe({'fine': .40, 'coarse': .15, 'mid': .45}, {'coarse': .35, 'mid': .65}, spatial=.40),
    'hybrid-wide': Recipe({'fine': .40, 'coarse': .20, 'deep': .40}, {'coarse': .35, 'deep': .65},
                          spatial=.65, dino_contrast=.55, fine_window=47),
    'hybrid-agreement': Recipe({'fine': .45, 'mid': .25, 'deep': .30},
                               {'coarse': .25, 'mid': .25, 'deep': .50}, spatial=.60),
}
METHODS = (REFERENCE, GPU_REPLAY, *RECIPES)
GRIDS = {'fine': 56, 'coarse': 28, 'mid': 28, 'deep': 28}
SIGMAS = {'fine': 3., 'coarse': 3.5, 'mid': 4., 'deep': 4.}


def needs(method):
    heads = RECIPES[method].heads if method in RECIPES else ('fine', 'coarse')
    return any(h in heads for h in ('fine', 'coarse')), any(h in heads for h in ('mid', 'deep'))


def model_path(root, category, seed, method):
    if method in (REFERENCE, GPU_REPLAY):
        return wrn_path(category, seed)
    return root / 'models' / category / str(seed) / f'{method}.zip'


def smooth(flat, side, size, sigma):
    if not sigma:
        return F.interpolate(flat.reshape(-1, 1, side, side), size, mode='bilinear', align_corners=True)[:, 0]
    radius = int(sigma * 3 + .5)
    x = torch.arange(-radius, radius + 1, device=flat.device, dtype=torch.float64)
    kernel = torch.softmax(-.5 * (x / sigma).square(), dim=0).float()
    parts = []
    for rows in flat.split(12):
        maps = F.interpolate(rows.reshape(-1, 1, side, side), size, mode='bilinear', align_corners=True)
        maps = F.conv2d(F.pad(maps, (radius, radius, 0, 0), mode='replicate'), kernel.reshape(1, 1, 1, -1))
        maps = F.conv2d(F.pad(maps, (0, 0, radius, radius), mode='replicate'), kernel.reshape(1, 1, -1, 1))
        parts.append(maps[:, 0])
    return torch.cat(parts)


def base_patch(head, raw, recipe, scorers):
    if head == 'fine':
        return adjust_scores(raw, (56, 56), recipe.fine_window, recipe.fine_contrast) if recipe.fine_contrast else raw
    if head == 'coarse':
        return scorers[head].map_values(raw)
    return adjust_scores(raw, (28, 28), 19, recipe.dino_contrast) if recipe.dino_contrast else raw


def raw_image(head, rows):
    return image_score(rows, 16 if head in ('fine', 'coarse') else 10)


def moments(calibration, recipe, scorers):
    stats = {}
    for head in recipe.maps:
        side = GRIDS[head]
        values = base_patch(head, calibration[head], recipe, scorers)
        stats[f'{head}_dispersion'] = np.float32(values.std().clamp_min(1e-5).item())
        if not recipe.spatial:
            continue
        fit = values.reshape(-1, 1, side, side)
        quantiles = torch.quantile(fit, torch.tensor([.50, .90], device=fit.device), dim=0)
        center = F.adaptive_avg_pool2d(quantiles[0], (14, 14))[0]
        stats[f'{head}_center'] = center.cpu().numpy().astype(np.float16)
        if recipe.normalize_spread:
            scale = F.adaptive_avg_pool2d(quantiles[1] - quantiles[0], (14, 14))[0]
            typical = scale.median().clamp_min(1e-5)
            stats[f'{head}_scale'] = torch.lerp(scale, typical, .55).clamp_min(1e-5).cpu().numpy().astype(np.float16)
            stats[f'{head}_typical'] = np.float32(typical.item())
    if len(recipe.image) > 1:
        for head in recipe.image:
            scores = raw_image(head, calibration[head])
            q50, q90 = torch.quantile(scores, torch.tensor([.50, .90], device=scores.device))
            stats[f'{head}_image'] = np.array([q50.item(), max((q90 - q50).item(), 1e-5)], np.float32)
    return stats


def assemble(root, category, seed, method, calibration, device, dino_banks=None):
    path = model_path(root, category, seed, method)
    if path.exists():
        return path.read_bytes()
    recipe = RECIPES[method]
    payloads = {}
    if any(head in recipe.heads for head in ('fine', 'coarse')):
        with ZipFile(wrn_path(category, seed)) as archive:
            for head, member in (('fine', 'dense.npz'), ('coarse', 'image.npz')):
                if head in recipe.heads:
                    payloads[head] = (archive.read(member), archive.getinfo(member).compress_type)
    for head in ('mid', 'deep'):
        if head in recipe.heads:
            source = dino_banks[head] if dino_banks is not None else dino_path(root, category, seed, head)
            payloads[head] = (source.read_bytes(), ZIP_STORED)
    scorers = {}
    if 'coarse' in payloads:
        scorers['coarse'] = Scorer(payloads['coarse'][0], IMAGE, device, (28, 28))
    stats = BytesIO()
    np.savez_compressed(stats, **moments(calibration, recipe, scorers))
    buffer = BytesIO()
    with ZipFile(buffer, 'w') as archive:
        for head, (data, compression) in payloads.items():
            archive.writestr(f'{head}.bank', data, compress_type=compression)
        archive.writestr('normal.npz', stats.getvalue())
        archive.writestr('recipe.json', json.dumps({'method': method}))
    payload = buffer.getvalue()
    array_save(path, payload)
    return payload


class Predictor:
    def __init__(self, payload, method, device):
        self.method = method
        self.recipe = RECIPES.get(method)
        self.heads = self.recipe.heads if self.recipe else ('fine', 'coarse')
        if self.recipe is None:
            self.reference = DualHead(payload, device)
            self.scorers = {'fine': self.reference.dense, 'coarse': self.reference.image}
            return
        with ZipFile(BytesIO(payload)) as archive:
            if json.loads(archive.read('recipe.json'))['method'] != method:
                raise ValueError('Requested configuration differs from the bank recipe')
            self.scorers = {}
            for head in self.heads:
                data = archive.read(f'{head}.bank')
                if head in ('fine', 'coarse'):
                    name = DENSE if head == 'fine' else IMAGE
                    side = GRIDS[head]
                    self.scorers[head] = Scorer(data, name, device, (side, side))
                else:
                    self.scorers[head] = CosineBank(data, device)
            with np.load(BytesIO(archive.read('normal.npz')), allow_pickle=False) as stats:
                self.stats = {key: torch.as_tensor(stats[key], device=device).float() for key in stats.files}
        self.spatial = {}
        for head in self.recipe.maps:
            if not self.recipe.spatial:
                continue
            size = GRIDS[head]
            moments = [self.stats[f'{head}_center']]
            if self.recipe.normalize_spread:
                moments.append(self.stats[f'{head}_scale'])
            grid = torch.stack(moments).unsqueeze(0)
            self.spatial[head] = F.interpolate(grid, (size, size), mode='bilinear',
                                               align_corners=False)[0].flatten(1)

    def raw(self, low, high):
        features = dict(zip(HEADS, (*low, *high), strict=True))
        return {
            head: self.scorers[head].raw(features[head]) if head in ('fine', 'coarse')
            else self.scorers[head].scores(features[head])
            for head in self.heads
        }

    def predict(self, raw, size=(224, 224)):
        if self.method == REFERENCE:
            return self.reference.predict((raw['fine'], raw['coarse']), size, .25)
        if self.method == GPU_REPLAY:
            images = raw_image('coarse', raw['coarse'])
            fine = smooth(self.scorers['fine'].map_values(raw['fine']), 56, size, 3.)
            coarse = smooth(self.scorers['coarse'].map_values(raw['coarse']), 28, size, 3.5)
            return images, .75 * fine + .25 * self.reference.relative_scale * coarse
        image = 0.
        for head, weight in self.recipe.image.items():
            score = raw_image(head, raw[head])
            if len(self.recipe.image) > 1:
                center, scale = self.stats[f'{head}_image']
                score = (score - center) / scale
            image = image + weight * score
        maps = 0.
        for head, weight in self.recipe.maps.items():
            values = base_patch(head, raw[head], self.recipe, self.scorers)
            if self.recipe.spatial:
                center = self.spatial[head][0]
                corrected = values - center
                if self.recipe.normalize_spread:
                    scale = self.spatial[head][1]
                    corrected = corrected * (self.stats[f'{head}_typical'] / scale)
                values = torch.lerp(values, corrected, self.recipe.spatial)
            values = values / self.stats[f'{head}_dispersion']
            maps = maps + weight * smooth(values, GRIDS[head], size, SIGMAS[head] if self.recipe.gaussian else 0.)
        return image, maps


def quality(args):
    device = start_device()
    for category in args.categories:
        fingerprint, paths, ids = manifest_parts(args.data, category)
        with np.load(WORK / 'cache' / category / 'test.npz', allow_pickle=False) as original:
            labels, masks = original['labels'], original['masks']
            if list(original['image_ids']) != ids:
                raise ValueError('Mask cache image ordering differs')
        evaluator = Localization(masks).on_device(device)
        with np.load(args.output / 'raw' / f'{category}.npz', allow_pickle=False) as stored:
            metadata = json.loads(stored['metadata'].item())
            if metadata['manifest'] != fingerprint or not set(args.seeds) <= set(metadata['seeds']):
                raise ValueError('Raw scores are incompatible with this split or seed list')
            for seed in args.seeds:
                calibration = {head: torch.from_numpy(stored[f'calibration_{seed}_{head}']).to(device) for head in HEADS}
                raw = {head: torch.from_numpy(stored[f'test_{seed}_{head}']).to(device) for head in HEADS}
                for method in args.methods:
                    target = args.output / 'quality' / category / f'{seed}-{method}.json'
                    if target.exists():
                        continue
                    payload = assemble(args.output, category, seed, method, calibration, device)
                    model = Predictor(payload, method, device)
                    images, maps = model.predict(raw, masks.shape[1:])
                    result = dict(category=category, seed=seed, method=method, images=len(paths['test']),
                                  manifest_sha256=fingerprint, artifact_bytes=len(payload),
                                  image_auroc=binary_auroc(labels, images.cpu().numpy()),
                                  **evaluator.evaluate_device(maps))
                    array_save(target, json.dumps(result).encode())
                    del maps, images, model
        print(f'Quality for {category}: {", ".join(args.methods)}', flush=True)
        del evaluator
        torch.cuda.empty_cache()
    rows = [json.loads(p.read_text()) for p in sorted((args.output / 'quality').glob('*/*.json'))]
    save_csv(args.output / 'quality.csv', rows)


def encoders(method, device, package=None):
    wrn, dino = needs(method)
    if package is not None:
        from encoder_package import load_packaged

        return load_packaged(package, wrn, dino, device)
    return create_encoder(device, DENSE, True) if wrn else None, make_dino(device) if dino else None


def predict_path(path, model, wrn, dino, device):
    low, high = views(path, wrn is not None, dino is not None)
    low = embeddings(encode_tensors(low.unsqueeze(0), wrn, device, True)) if wrn is not None else (None, None)
    high = encode_dino(high.unsqueeze(0), dino, device) if dino is not None else (None, None)
    images, maps = model.predict(model.raw(low, high))
    return images.cpu().numpy(), torch.as_tensor(maps).cpu().numpy()


def profile(args, predict=predict_path):
    device = start_device()
    method = args.methods[0]
    wrn, dino = encoders(method, device, args.encoder_package)
    folder = 'resources-packaged' if args.encoder_package else 'resources'
    for category in args.categories:
        dest = args.output / folder / f'{method}-{category}.json'
        if dest.exists():
            continue
        paths = test_paths(WORK, args.data, category)
        payload = model_path(args.output, category, 0, method).read_bytes()
        torch.cuda.empty_cache()
        before = torch.cuda.memory_allocated(device)
        model = Predictor(payload, method, device)
        resident = torch.cuda.memory_allocated(device) - before
        for path in paths[:5]:
            predict(path, model, wrn, dino, device)
        timing = []
        with measure(device) as memory:
            for path in paths:
                start = perf_counter()
                predict(path, model, wrn, dino, device)
                torch.cuda.synchronize(device)
                timing.append(1000 * (perf_counter() - start))
        row = dict(category=category, method=method, seed=0, images=len(paths), artifact_bytes=len(payload),
                   encoder_bytes=before, bank_vram_bytes=resident, p50_ms=np.median(timing),
                   p95_ms=np.percentile(timing, 95), **memory,
                   manifest_sha256=sha256(WORK / 'cache' / f'{category}.json'))
        array_save(dest, json.dumps(row).encode())
        del model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('quality', 'profile', 'predict'))
    parser.add_argument('--data', type=Path)
    parser.add_argument('--output', type=Path, default=RESULTS)
    parser.add_argument('--categories', nargs='+', choices=CATEGORIES, default=CATEGORIES)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0])
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=METHODS)
    parser.add_argument('--bank', type=Path)
    parser.add_argument('--image', type=Path)
    parser.add_argument('--map', type=Path)
    parser.add_argument('--encoder-package', type=Path)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.phase == 'predict':
        if len(args.methods) != 1 or any(p is None for p in (args.bank, args.image, args.map)):
            parser.error('predict requires one method, --bank, --image and --map')
        with torch.inference_mode():
            device = start_device()
            method = args.methods[0]
            model = Predictor(args.bank.read_bytes(), method, device)
            score, maps = predict_path(args.image, model, *encoders(method, device, args.encoder_package), device)
        args.map.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.map, maps[0])
        print(json.dumps(dict(score=float(score[0]), shape=list(maps[0].shape), bank_bytes=args.bank.stat().st_size)))
        return
    if args.data is None:
        parser.error(f'{args.phase} requires --data')
    if args.phase == 'quality' or args.worker:
        with torch.inference_mode():
            (quality if args.phase == 'quality' else profile)(args)
        return
    for method in args.methods:
        command = [sys.executable, __file__, 'profile', '--worker', '--data', str(args.data),
                   '--output', str(args.output), '--methods', method, '--categories', *args.categories]
        if args.encoder_package:
            command.extend(('--encoder-package', str(args.encoder_package)))
        subprocess.run(command, check=True)
        print(f'Profiled {method}', flush=True)
    folder = 'resources-packaged' if args.encoder_package else 'resources'
    rows = [json.loads(p.read_text()) for p in sorted((args.output / folder).glob('*.json'))]
    save_csv(args.output / f'{folder}.csv', rows)


if __name__ == '__main__':
    main()
