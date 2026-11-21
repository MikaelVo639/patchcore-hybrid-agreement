"""Profile one matched category/method in an independent process on real test images."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as T

from benchmark import anomaly_maps, coreset, measure, projected_pool
from compression import Bank, fit_bank
from patchcore_budget.datasets import load_manifest, resolve_dataset_path


def prepare_image(path, copy_rgb=True):
    with Image.open(path) as raw:
        rgb = raw.convert('RGB') if copy_rgb or raw.mode != 'RGB' else raw
        image = T.center_crop(T.resize(rgb, 256, InterpolationMode.BILINEAR,
                                       antialias=True), 224)
        image = T.normalize(T.to_tensor(image), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return image


def encode_image(image, encoder, device, layer2_only=False):
    tensors = encoder(image.unsqueeze(0).to(device))
    low = F.avg_pool2d(tensors['layer2'], 3, stride=1, padding=1)
    if layer2_only:
        return low.permute(0, 2, 3, 1).contiguous().flatten(0, 2)
    high = F.avg_pool2d(tensors['layer3'], 3, stride=1, padding=1)
    high = F.interpolate(high, size=low.shape[-2:], mode='bilinear', align_corners=False)
    return torch.cat((low, high), dim=1).permute(0, 2, 3, 1).contiguous().flatten(0, 2)


def encode(path, encoder, device, layer2_only=False):
    return encode_image(prepare_image(path), encoder, device, layer2_only)


def profile(args):
    device = torch.device(args.device)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    synchronize = (lambda: torch.cuda.synchronize(device)) if device.type == 'cuda' else (lambda: None)
    if device.type == 'cuda':
        torch.empty(1, device=device).square()
        torch.cuda.empty_cache()
    rows = json.loads((args.run / args.category / str(args.seed) / 'metrics.json').read_text())
    row = next(row for row in rows if row['method'] == args.method and row['budget'] == args.budget)
    artifact = args.run / row['artifact']
    budget = args.budget or None
    with np.load(args.cache / args.category / 'build.npz', allow_pickle=False) as saved:
        data = saved['features']
        data = data.reshape(-1, data.shape[-1])
    rng = np.random.default_rng(args.seed)
    ids = np.sort(rng.choice(len(data), min(8192, len(data)), replace=False))
    sampled = data[ids]
    del data
    with measure(device) as build:
        pool = torch.as_tensor(sampled, device=device)
        mean = basis = residual = None
        if args.method.startswith('pca'):
            pool, mean, basis, energy = projected_pool(pool, row['dimension'])
            residual = energy if args.method.endswith('-residual') else None
        bits = 4 if 'int4' in args.method else 8 if 'int8' in args.method else 16 if 'fp16' in args.method else 32
        order = np.arange(len(pool)) if row['patches'] == len(pool) else coreset(pool, row['patches'], args.seed)
        built = fit_bank(pool, order, bits, budget, mean=mean, projection=basis,
                         residual=residual, unsigned=args.method == 'uint4-full')
        rebuilt = built.serialize()
        if len(rebuilt) != row['artifact_bytes']:
            raise ValueError('Build profile does not match the evaluated artifact size')
        if not args.method.startswith('pca') and rebuilt != artifact.read_bytes():
            raise ValueError('Profile rebuilt a bank different from the evaluated artifact')
    del pool, sampled, built
    if args.method.startswith('pca'):
        del energy, residual
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
    encoder = create_feature_extractor(backbone, return_nodes={'layer2': 'layer2', 'layer3': 'layer3'})
    encoder.eval().requires_grad_(False).to(device)
    del backbone
    manifest = load_manifest(args.cache / f'{args.category}.json')
    paths = [resolve_dataset_path(args.data, item['path']) for item in manifest['records']
             if item['split'] == 'test']
    chosen = [paths[i] for i in np.linspace(0, len(paths) - 1, 20, dtype=int)]
    query = encode(chosen[0], encoder, device)
    del query
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    image_ms, cached_ms = [], []
    with measure(device) as inference:
        bank = Bank.deserialize(artifact.read_bytes()).prepare(device)
        if device.type == 'cuda':
            inference['vram_bank_delta_bytes'] = torch.cuda.memory_allocated(device) - inference['vram_start_bytes']
        for i, path in enumerate([chosen[0]] * 5 + chosen):
            start = perf_counter()
            features = encode(path, encoder, device)
            synchronize()
            score_start = perf_counter()
            scores = bank.score(features)
            anomaly_maps(scores, (28, 28), (224, 224))
            stop = perf_counter()
            if i >= 5:
                image_ms.append((stop - start) * 1000)
                cached_ms.append((stop - score_start) * 1000)
    payload = dict(category=args.category, method=args.method, budget=args.budget, seed=args.seed,
                   artifact_bytes=row['artifact_bytes'], patches=row['patches'], build=build,
                   inference=inference, image_ms=np.percentile(image_ms, [50, 95]).tolist(),
                   feature_ms=np.percentile(cached_ms, [50, 95]).tolist(), device=str(device))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(args.category, args.method, args.budget, payload['image_ms'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('cache', 'run', 'data', 'output'):
        parser.add_argument(f'--{key}', type=Path, required=True)
    parser.add_argument('--category', required=True)
    parser.add_argument('--method', required=True)
    parser.add_argument('--budget', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    with torch.inference_mode():
        profile(parser.parse_args())
