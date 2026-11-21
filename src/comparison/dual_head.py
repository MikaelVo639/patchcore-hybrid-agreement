"""Combine a dense localization bank and a larger-pool layer2 image scorer."""

import argparse
from io import BytesIO
import json
from pathlib import Path
from time import perf_counter
from zipfile import ZIP_LZMA, ZipFile

import numpy as np
import torch

from benchmark import Localization, measure
from deployment import create_encoder, encode_tensors, extract_layers, features_from_layers
from experiment import HERE, WORK, Scorer, array_save, bank_path, start_device
from fast_inference import ordered_prefetch
from inference_experiment import test_paths
from patchcore_budget.datasets import load_manifest, resolve_dataset_path
from patchcore_budget.io import sha256
from patchcore_budget.metrics import binary_auroc
from profile_resources import prepare_image
from robust_maps import image_score
from score_tuning import make_maps
from summarize import CATEGORIES


DENSE, IMAGE = 'layer1-layer2-wide', 'uint4-32k-cal'
DUAL = {'dual-dense': 0., 'dual-fusion25': .25, 'dual-fusion50': .50}


def extract(paths, encoder, device):
    return embeddings(extract_layers(paths, encoder, device, True))


def embeddings(layers):
    return features_from_layers(layers), features_from_layers({'layer2': layers['layer2']})


def mini_batches(paths, size=1):
    for start in range(0, len(paths), size):
        yield paths[start:start + size]


def head_maps(scorer, raw, size):
    return make_maps(scorer.map_values(raw), scorer.grid, size, scorer.sigma)


class DualHead:
    def __init__(self, payload, device):
        with ZipFile(BytesIO(payload)) as archive:
            self.dense = Scorer(archive.read('dense.npz'), DENSE, device, (56, 56))
            self.image = Scorer(archive.read('image.npz'), IMAGE, device, (28, 28))
            self.relative_scale = json.loads(archive.read('normal-scales.json'))

    def raw(self, embedding):
        return self.dense.raw(embedding[0]), self.image.raw(embedding[1])

    def predict(self, raw, size, mix):
        score = image_score(raw[1], self.image.top_k)
        maps = head_maps(self.dense, raw[0], size)
        if mix:
            second = head_maps(self.image, raw[1], size)
            maps = (1 - mix) * maps + mix * self.relative_scale * second
        return score, maps


def artifact(output, category, seed, calibration, device):
    destination = output / 'models' / category / str(seed) / 'dual-head.zip'
    if destination.exists():
        return destination.read_bytes()
    dense = bank_path(output, category, seed, DENSE).read_bytes()
    image = bank_path(output, category, seed, IMAGE).read_bytes()
    scorer_dense = Scorer(dense, DENSE, device, (56, 56))
    scorer_image = Scorer(image, IMAGE, device, (28, 28))
    scales = []
    for scorer, embeddings in zip((scorer_dense, scorer_image), calibration, strict=True):
        raw = torch.cat([scorer.raw(rows) for rows in embeddings])
        scales.append(scorer.map_values(raw).std().clamp_min(1e-6).item())
    buffer = BytesIO()
    with ZipFile(buffer, 'w') as archive:
        archive.writestr('dense.npz', dense, compress_type=ZIP_LZMA)
        archive.writestr('image.npz', image)
        archive.writestr('normal-scales.json', json.dumps(scales[0] / scales[1]))
    payload = buffer.getvalue()
    array_save(destination, payload)
    return payload


def quality(args, device, encoder):
    for category in args.categories:
        dest = args.output / 'quality' / category
        pending = [seed for seed in args.seeds if any(not (dest / f'{seed}-{name}.json').exists() for name in DUAL)]
        if not pending:
            continue
        path = WORK / 'cache' / f'{category}.json'
        manifest = load_manifest(path)
        fingerprint = sha256(path)
        records = manifest['records']
        test = [row for row in records if row['split'] == 'test']
        normal_paths = [resolve_dataset_path(args.data, row['path']) for row in records if row['split'] == 'calibration']
        test_files = [resolve_dataset_path(args.data, row['path']) for row in test]
        with np.load(WORK / 'cache' / category / 'test.npz', allow_pickle=False) as saved:
            ids, labels, masks = (saved[key] for key in ('image_ids', 'labels', 'masks'))
            cached = json.loads(str(saved['metadata_json'].item()))
        if cached['manifest_sha256'] != fingerprint or list(ids) != [row['image_id'] for row in test]:
            raise ValueError(f'Dual-head test identity differs for {category}')
        normal = [extract(paths, encoder, device) for paths in mini_batches(normal_paths, 4)]
        calibration = tuple(zip(*normal, strict=True))
        payloads = {seed: artifact(args.output, category, seed, calibration, device) for seed in pending}
        del calibration, normal
        scorers = {seed: DualHead(payload, device) for seed, payload in payloads.items()}
        rows = {seed: [] for seed in pending}
        for image in ordered_prefetch(test_files, prepare_image, workers=3):
            features = embeddings(encode_tensors(image.unsqueeze(0), encoder, device, True))
            for seed, scorer in scorers.items():
                rows[seed].append(scorer.raw(features))
        evaluation = Localization(masks).on_device(device)
        for seed in pending:
            raw = tuple(torch.cat(values) for values in zip(*rows[seed], strict=True))
            for name, mix in DUAL.items():
                scores, maps = scorers[seed].predict(raw, masks.shape[1:], mix)
                result = dict(method=name, category=category, seed=seed, test_images=len(test_files),
                              manifest_sha256=fingerprint, artifact_bytes=len(payloads[seed]),
                              patches=len(scorers[seed].dense.prepared.bank) + len(scorers[seed].image.prepared.bank),
                              image_auroc=binary_auroc(labels, scores.cpu().numpy()),
                              **evaluation.evaluate_device(maps))
                array_save(dest / f'{seed}-{name}.json', json.dumps(result).encode())
                del maps, scores
        del evaluation
        print(f'Dual-head quality for {category}', flush=True)


def profile(args, device, encoder):
    name = args.variant
    for category in args.categories:
        dest = args.output / 'resources' / f'{name}-{category}.json'
        if dest.exists():
            continue
        paths = test_paths(WORK, args.data, category)
        payload = (args.output / 'models' / category / '0' / 'dual-head.zip').read_bytes()
        torch.cuda.empty_cache()
        before = torch.cuda.memory_allocated(device)
        model = DualHead(payload, device)
        resident = torch.cuda.memory_allocated(device) - before

        def predict(path):
            rows = extract([path], encoder, device)
            return model.predict(model.raw(rows), (224, 224), DUAL[name])

        for path in paths[:5]:
            predict(path)
        timings = []
        with measure(device) as memory:
            for path in paths:
                start = perf_counter()
                predict(path)
                torch.cuda.synchronize(device)
                timings.append(1000 * (perf_counter() - start))
        row = dict(method=name, category=category, seed=0, images=len(paths), artifact_bytes=len(payload),
                   manifest_sha256=sha256(WORK / 'cache' / f'{category}.json'), bank_vram_bytes=resident,
                   p50_ms=np.median(timings), p95_ms=np.percentile(timings, 95),
                   images_per_second=1000 * len(paths) / sum(timings), **memory)
        array_save(dest, json.dumps(row).encode())
        del model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('quality', 'profile', 'predict'))
    parser.add_argument('--data', type=Path)
    parser.add_argument('--output', type=Path, default=HERE / 'results')
    parser.add_argument('--categories', nargs='+', choices=CATEGORIES, default=CATEGORIES)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument('--variant', choices=tuple(DUAL), default='dual-fusion25')
    parser.add_argument('--bank', type=Path)
    parser.add_argument('--image', type=Path)
    parser.add_argument('--map', type=Path)
    args = parser.parse_args()
    if args.phase == 'predict' and any(p is None for p in (args.bank, args.image, args.map)):
        parser.error('predict requires --bank, --image and --map')
    if args.phase != 'predict' and args.data is None:
        parser.error(f'{args.phase} requires --data')
    with torch.inference_mode():
        device = start_device()
        encoder = create_encoder(device, DENSE, True)
        if args.phase == 'quality':
            quality(args, device, encoder)
        elif args.phase == 'profile':
            profile(args, device, encoder)
        else:
            model = DualHead(args.bank.read_bytes(), device)
            score, maps = model.predict(model.raw(extract([args.image], encoder, device)), (224, 224), DUAL[args.variant])
            args.map.parent.mkdir(exist_ok=True, parents=True)
            np.save(args.map, maps[0])
            print(json.dumps(dict(score=score.item(), shape=list(maps[0].shape),
                                  bank_bytes=args.bank.stat().st_size, map=str(args.map))))


if __name__ == '__main__':
    main()
