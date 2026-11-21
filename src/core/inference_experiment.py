"""Profile exact inference variants and verify the selected backend on all MVTec AD seeds."""

import argparse
import csv
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import torch
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
from torchvision.models.feature_extraction import create_feature_extractor

from benchmark import measure
from fast_inference import CacheMode, CachedDetector, ordered_prefetch
from patchcore_budget.cache import FeatureCache
from patchcore_budget.datasets import load_manifest, resolve_dataset_path
from patchcore_budget.io import sha256
from profile_resources import encode_image, prepare_image
from score_tuning import Detector
from summarize import CATEGORIES


PROFILE_CLASSES = ('bottle', 'carpet', 'screw')


def load_bank(work, category, seed):
    folder = work / 'lossless-confirm'
    codec = json.loads((folder / 'compression.json').read_text())['codec']
    return Detector.load((folder / category / f'{seed}-{codec}.npz').read_bytes())


def init():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    return torch.device('cuda:0')


def encoder_on(device):
    weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1
    model = create_feature_extractor(wide_resnet50_2(weights=weights), return_nodes={'layer2': 'layer2'})
    return model.eval().requires_grad_(False).to(device)


def test_paths(work, data, category):
    manifest = load_manifest(work / 'cache' / f'{category}.json')
    paths = [resolve_dataset_path(data, row['path']) for row in manifest['records'] if row['split'] == 'test']
    return [paths[i] for i in np.linspace(0, len(paths) - 1, 20, dtype=int)]


def create(detector, device, name):
    if name == 'previous':
        prepared = detector.prepare(device)
        return lambda features: detector.infer(features, prepared)
    return CachedDetector.create(detector, device, name).infer


def same(left, right):
    return torch.equal(left[0], right[0]) and np.array_equal(left[1], right[1])


def scoring_trials(work, data, output):
    device = init()
    encoder = encoder_on(device)
    results = []
    for category in PROFILE_CLASSES:
        detector = load_bank(work, category, 0)
        inputs = [encode_image(prepare_image(path), encoder, device, True)
                  for path in test_paths(work, data, category)]
        infer = create(detector, device, 'previous')
        expected = [infer(features) for features in inputs]
        del infer
        for name in ('previous', *(mode.value for mode in CacheMode)):
            torch.cuda.empty_cache()
            with measure(device) as resources:
                infer = create(detector, device, name)
                torch.cuda.synchronize(device)
                resident = torch.cuda.memory_allocated(device) - resources['vram_start_bytes']
                actual = [infer(features) for features in inputs]
                timing = []
                for features in inputs * 3:
                    start = perf_counter()
                    infer(features)
                    timing.append(1000 * (perf_counter() - start))
            exact = all(same(left, right) for left, right in zip(expected, actual, strict=True))
            results.append(dict(category=category, mode=name, exact=exact,
                                p50_ms=float(np.percentile(timing, 50)), p95_ms=float(np.percentile(timing, 95)),
                                resident_bytes=resident,
                                score_peak_bytes=resources['vram_peak_bytes'] - resources['vram_start_bytes']))
            del infer, actual
        print(f'Profiled scoring on {category}', flush=True)
    ranks = {}
    for mode in CacheMode:
        group = [row for row in results if row['mode'] == mode]
        if all(row['exact'] for row in group):
            ranks[mode.value] = (median(row['p95_ms'] for row in group),
                                 median(row['score_peak_bytes'] for row in group))
    if not ranks:
        raise ValueError('No variant preserved predictions')
    candidate = min(ranks, key=ranks.get)
    report = dict(candidate=candidate, results=results)
    (output / 'scoring.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


def check_accuracy(work, output):
    device = init()
    candidate = json.loads((output / 'scoring.json').read_text())['candidate']
    modes = tuple(dict.fromkeys((candidate, CacheMode.NORMS.value)))
    rows = []
    for category in CATEGORIES:
        test = FeatureCache.load(work / 'cache' / category / 'test.npz', 'test')
        if sha256(work / 'cache' / f'{category}.json') != test.metadata['manifest_sha256']:
            raise ValueError('Test cache manifest mismatch')
        features = torch.from_numpy(np.ascontiguousarray(test.features[..., :512])).to(device)
        grid, size = features.shape[1:3], test.masks.shape[1:]
        patches = features.shape[1] * features.shape[2]
        for seed in range(5):
            detector = load_bank(work, category, seed)
            prepared = detector.prepare(device)
            optimized = [CachedDetector.create(detector, device, mode) for mode in modes]
            for batch in features.split(12):
                flat = batch.reshape(-1, 512)
                reference = detector.infer(flat, prepared, grid, size)
                for mode, backend in zip(modes, optimized, strict=True):
                    if not same(reference, backend.infer(flat, grid, size)):
                        raise ValueError(f'Predictions changed: {category} seed {seed} ({mode})')
            rows.append(dict(category=category, seed=seed, images=len(features), patches_per_image=patches,
                             exact_image_scores=True, exact_anomaly_maps=True, verified_backends=' '.join(modes)))
        print(f'Exact score and map confirmed for {category}, all five seeds', flush=True)
    with (output / 'accuracy.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare(path, copy_rgb=True):
    with torch.inference_mode():
        return prepare_image(path, copy_rgb=copy_rgb)


def pipeline_trials(work, data, output):
    device = init()
    encoder = encoder_on(device)
    candidate = json.loads((output / 'scoring.json').read_text())['candidate']
    all_rows = []
    for category in PROFILE_CLASSES:
        detector = load_bank(work, category, 0)
        paths = test_paths(work, data, category)
        encode = lambda image: encode_image(image, encoder, device, True)
        old = create(detector, device, 'previous')
        expected = [old(encode(prepare(path))) for path in paths]
        del old
        stages = [('sequential', 'previous', 0), ('cached', candidate, 0), ('prefetch2', candidate, 2),
                  ('prefetch2-norms', CacheMode.NORMS.value, 2)]
        for repeat in range(3):
            for stage, mode, workers in stages[repeat:] + stages[:repeat]:
                torch.cuda.empty_cache()
                with measure(device) as resources:
                    infer = create(detector, device, mode)
                    reader = lambda path: prepare(path, copy_rgb=mode == 'previous')
                    for path in paths[:5]:
                        infer(encode(reader(path)))
                    torch.cuda.synchronize(device)
                    if workers:
                        inputs = ordered_prefetch(paths, reader, workers)
                    else:
                        inputs = map(reader, paths)
                    per_image = []
                    elapsed_start = start = perf_counter()
                    for image, reference in zip(inputs, expected, strict=True):
                        results = infer(encode(image))
                        stop = perf_counter()
                        per_image.append(1000 * (stop - start))
                        if not same(reference, results):
                            raise ValueError(f'Image pipeline changed output for {category}')
                        start = perf_counter()
                    elapsed = perf_counter() - elapsed_start
                all_rows.append(dict(category=category, stage=stage, repeat=repeat, images=len(paths),
                                     images_per_second=len(paths) / elapsed, p95_interval_ms=np.percentile(per_image, 95),
                                     **resources))
                del infer
        print(f'Profiled and checked real-image pipeline for {category}', flush=True)
    with (output / 'pipeline.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('scoring', 'accuracy', 'pipeline'))
    parser.add_argument('--work', type=Path, default=Path(__file__).parent)
    parser.add_argument('--data', type=Path)
    args = parser.parse_args()
    output = args.work / 'fast-confirm'
    output.mkdir(exist_ok=True)
    with torch.inference_mode():
        if args.phase == 'accuracy':
            check_accuracy(args.work, output)
        elif args.data is None:
            parser.error('--data is required for real-image profiling')
        elif args.phase == 'scoring':
            scoring_trials(args.work, args.data, output)
        else:
            pipeline_trials(args.work, args.data, output)
