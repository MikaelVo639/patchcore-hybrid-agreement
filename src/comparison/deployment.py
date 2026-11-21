"""Profile deployment backends and check the FP16 encoder on every test image."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import torch
from PIL import Image
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
from torchvision.models.feature_extraction import create_feature_extractor

from benchmark import Localization, measure
from experiment import EARLY, HERE, WORK, VARIANTS, Scorer, array_save, bank_dimensions, bank_path, save_csv, start_device
from inference_experiment import test_paths
from patchcore_budget.datasets import load_manifest, resolve_dataset_path
from patchcore_budget.io import sha256
from patchcore_budget.metrics import binary_auroc
from profile_resources import prepare_image
from summarize import CATEGORIES


BACKENDS = ('fp32-packed', 'fp32-decoded', 'fp16-packed', 'fp16-decoded')


def model_grid(variant):
    return (56, 56) if variant in EARLY else (28, 28)


def create_encoder(device, variant, half):
    if variant in EARLY:
        nodes = {'layer1': 'layer1'}
        if variant != 'layer1-dense':
            nodes['layer2'] = 'layer2'
    else:
        nodes = {'layer2': 'layer2'}
        if bank_dimensions(variant) != 512:
            nodes['layer3'] = 'layer3'
    model = create_feature_extractor(wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1),
                                     return_nodes=nodes).eval().requires_grad_(False)
    if half:
        return model.to(device, dtype=torch.float16, memory_format=torch.channels_last)
    return model.to(device)


def encode_tensors(images, encoder, device, half):
    images = images.to(
        device=device, dtype=torch.float16 if half else torch.float32,
        memory_format=torch.channels_last if half else torch.contiguous_format)
    return encoder(images)


def extract_layers(paths, encoder, device, half):
    return encode_tensors(torch.stack([prepare_image(path) for path in paths]), encoder, device, half)


def features_from_layers(layers):
    if 'layer1' in layers:
        low = torch.nn.functional.avg_pool2d(layers['layer1'].float(), 3, stride=1, padding=1)
        if 'layer2' in layers:
            stage = torch.nn.functional.avg_pool2d(layers['layer2'].float(), 3, stride=1, padding=1)
            batch, channels, height, width = stage.shape
            stage = stage.reshape(batch, channels // 2, 2, height, width).mean(2)
            stage = torch.nn.functional.interpolate(stage, size=low.shape[-2:], mode='bilinear',
                                                     align_corners=False)
            low = torch.cat((low, stage * .75), dim=1)
    else:
        low = torch.nn.functional.avg_pool2d(layers['layer2'].float(), 3, stride=1, padding=1)
    if 'layer3' in layers:
        high = torch.nn.functional.avg_pool2d(layers['layer3'].float(), 3, stride=1, padding=1)
        high = torch.nn.functional.interpolate(high, size=low.shape[-2:], mode='bilinear', align_corners=False)
        low = torch.cat((low, high), dim=1)
    return low.permute(0, 2, 3, 1).contiguous().flatten(0, 2)


def features(paths, encoder, device, half):
    return features_from_layers(extract_layers(paths, encoder, device, half))


def profile(args):
    device = start_device()
    half, decoded = args.backend.split('-')
    half, decoded = half == 'fp16', decoded == 'decoded'
    encoder = create_encoder(device, args.variant, half)
    grid = model_grid(args.variant)
    for category in args.categories:
        dest = args.output / 'runtime' / f'{args.variant}-{args.backend}-{category}.json'
        if dest.exists():
            continue
        paths = test_paths(WORK, args.data, category)
        payload = bank_path(args.output, category, 0, args.variant).read_bytes()
        torch.cuda.empty_cache()
        before = torch.cuda.memory_allocated(device)
        scorer = Scorer(payload, args.variant, device, grid, decoded=decoded)
        resident = torch.cuda.memory_allocated(device) - before

        def infer(path):
            raw = scorer.raw(features([path], encoder, device, half))
            return scorer.output(raw, (224, 224))

        if decoded:
            packed = Scorer(payload, args.variant, device, grid)
            for path in paths[:5]:
                query = features([path], encoder, device, half)
                if not torch.equal(packed.raw(query), scorer.raw(query)):
                    raise ValueError(f'Decoded bank changed nearest-neighbor scores for {category}')
            del packed, query
            torch.cuda.empty_cache()
        for path in paths[:5]:
            infer(path)
        timing, compute = [], []
        with measure(device) as memory:
            for path in paths:
                start = perf_counter()
                image = prepare_image(path).unsqueeze(0)
                model_start = perf_counter()
                rows = features_from_layers(encode_tensors(image, encoder, device, half))
                scorer.output(scorer.raw(rows), (224, 224))
                torch.cuda.synchronize(device)
                end = perf_counter()
                timing.append(1000 * (end - start))
                compute.append(1000 * (end - model_start))
                del rows
        row = dict(method=args.variant, backend=args.backend, category=category, images=len(paths),
                   seed=0, artifact_bytes=len(payload), bank_vram_bytes=resident,
                   manifest_sha256=sha256(WORK / 'cache' / f'{category}.json'),
                   p50_ms=np.median(timing), p95_ms=np.percentile(timing, 95),
                   model_p50_ms=np.median(compute), model_p95_ms=np.percentile(compute, 95),
                   images_per_second=1000 * len(paths) / sum(timing), **memory)
        array_save(dest, json.dumps(row).encode())
        del scorer


def evaluate(args):
    device = start_device()
    encoder = create_encoder(device, args.variant, True)
    grid = model_grid(args.variant)
    methods = (args.variant, *args.also)
    if args.also and (grid != (28, 28) or any(
            bank_dimensions(name) != 512 or name in EARLY for name in methods)):
        raise ValueError('Shared FP16 evaluation requires the same layer2 encoder for every variant')
    for category in args.categories:
        pending = {(name, seed): args.output / 'runtime_quality' / category / f'{seed}-{name}.json'
                   for seed in args.seeds for name in methods}
        pending = {key: path for key, path in pending.items() if not path.exists()}
        if not pending:
            continue
        manifest_path = WORK / 'cache' / f'{category}.json'
        manifest = load_manifest(manifest_path)
        records = [r for r in manifest['records'] if r['split'] == 'test']
        paths = [resolve_dataset_path(args.data, r['path']) for r in records]
        with np.load(WORK / 'cache' / category / 'test.npz', allow_pickle=False) as cached:
            ids, labels, masks = [cached[name] for name in ('image_ids', 'labels', 'masks')]
            metadata = json.loads(str(cached['metadata_json'].item()))
        fingerprint = sha256(manifest_path)
        if metadata['manifest_sha256'] != fingerprint or list(ids) != [r['image_id'] for r in records]:
            raise ValueError(f'FP16 test identity differs for {category}')
        loc = Localization(masks).on_device(device)
        size = masks.shape[1:]
        # Batch-one reproduces deployed embeddings. Streaming keeps only test
        # embeddings for the current class and shares them across bank seeds.
        test = torch.cat([features([path], encoder, device, True).cpu() for path in paths])
        last_bank, raw = None, None
        for (name, seed), dest in pending.items():
            path = bank_path(args.output, category, seed, name)
            payload = path.read_bytes()
            scorer = Scorer(payload, name, device, grid)
            if last_bank != path:
                raw = torch.cat([scorer.raw(batch.to(device)) for batch in test.split(8 * grid[0] * grid[1])])
                last_bank = path
            scores, maps = scorer.output(raw, size)
            row = dict(method=name, backend='fp16-packed', category=category, seed=seed,
                       manifest_sha256=fingerprint, artifact_bytes=len(payload), test_images=len(paths),
                       image_auroc=binary_auroc(labels, scores.cpu().numpy()), **loc.evaluate_device(maps))
            array_save(dest, json.dumps(row).encode())
            del maps, scores, scorer
        del test, loc, masks
        print(f'Checked FP16 for {category} ({", ".join(methods)})', flush=True)
    rows = [json.loads(p.read_text()) for p in sorted((args.output / 'runtime_quality').glob('*/*.json'))]
    save_csv(args.output / 'runtime-quality.csv', rows)


def predict(args):
    device = start_device()
    half, decoded = args.backend.split('-')
    half = half == 'fp16'
    encoder = create_encoder(device, args.variant, half)
    payload = args.bank.read_bytes()
    scorer = Scorer(payload, args.variant, device, model_grid(args.variant), decoded=decoded == 'decoded')
    raw = scorer.raw(features([args.image], encoder, device, half))
    score, maps = scorer.output(raw, (224, 224))
    args.map.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.map, maps[0])
    print(json.dumps(dict(score=score.item(), shape=list(maps[0].shape),
                          bank_bytes=len(payload), backend=args.backend, map=str(args.map))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('profile', 'quality', 'predict'))
    parser.add_argument('--variant', choices=tuple(VARIANTS), required=True)
    parser.add_argument('--also', nargs='+', choices=tuple(VARIANTS), default=[])
    parser.add_argument('--data', type=Path)
    parser.add_argument('--output', type=Path, default=HERE / 'results')
    parser.add_argument('--backend', choices=BACKENDS)
    parser.add_argument('--categories', nargs='+', choices=CATEGORIES, default=CATEGORIES)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--bank', type=Path)
    parser.add_argument('--image', type=Path)
    parser.add_argument('--map', type=Path)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.also and args.phase != 'quality':
        parser.error('--also applies only to quality evaluation')
    if args.phase == 'predict':
        args.backend = args.backend or 'fp16-packed'
        if any(path is None for path in (args.bank, args.image, args.map)):
            parser.error('predict requires --bank, --image and --map')
        with torch.inference_mode():
            predict(args)
        return
    if args.data is None:
        parser.error(f'{args.phase} requires --data')
    if args.phase == 'quality' or args.worker:
        args.backend = args.backend or 'fp16-packed'
        with torch.inference_mode():
            (evaluate if args.phase == 'quality' else profile)(args)
        return
    for backend in (args.backend,) if args.backend else BACKENDS:
        subprocess.run([sys.executable, __file__, 'profile', '--worker', '--variant', args.variant,
                        '--backend', backend, '--data', str(args.data), '--output', str(args.output),
                        '--categories', *args.categories], check=True)
        print(f'Profiled {args.variant} with {backend}', flush=True)
    rows = [json.loads(p.read_text()) for p in sorted((args.output / 'runtime').glob('*.json'))]
    save_csv(args.output / 'runtime-resources.csv', rows)


if __name__ == '__main__':
    main()
