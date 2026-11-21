"""Fit and evaluate PatchCore variants under one fixed MVTec AD manifest."""

import argparse
import csv
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZIP_LZMA, ZipFile

import numpy as np
import torch
from torch.nn import functional as F

from balanced_maps import adjust_scores
from benchmark import Localization, coreset, measure
from compression import Bank, fit_bank, unpack_rows
from patchcore_budget.cache import FeatureCache
from patchcore_budget.datasets import load_manifest
from patchcore_budget.io import sha256
from patchcore_budget.metrics import binary_auroc
from robust_maps import image_score
from score_tuning import calibrate, make_maps
from summarize import CATEGORIES


HERE = Path(__file__).resolve().parent
WORK = HERE.parent
BASELINES = ('fp32-10pct', 'robust-11', 'robust-19')
# Bank: feature representation, width, covariance power, sampling, bits.
BANKS = {
    'aso': ('full', 0, 0., 'elbow', 32),
    'lite32': ('fused', 32, 0., 'greedy', 16),
    'mh64': ('fused', 64, .50, 'greedy', 8),
    'soft4': ('low', 0, 0., 'denoise', 4),
    'uint4': ('low', 0, 0., 'greedy', 4),
    'uint4-384': ('low', 0, 0., 'small', 4),
    'uint4-32k': ('low', 0, 0., 'greedy', 4),
    'uint4-pos': ('low', 0, 0., 'greedy', 4),
    'pca64': ('fused', 64, 0., 'greedy', 4),
    'pca64-spatial': ('fused', 64, .25, 'greedy', 4),
    'pca96-low': ('low', 96, 0., 'greedy', 4),
    'pca64-centroids': ('fused', 64, 0., 'centroids', 4),
}
# Variant: bank, contrast window, Gaussian sigma, positional calibration, top-k.
VARIANTS = {
    'fp32-10pct': ('fp32-10pct', 0, 4., 0., 1),
    'robust-11': ('robust', 11, 4., 0., 16),
    'robust-19': ('robust', 19, 4., 0., 16),
    'robust-sharp': ('robust', 15, 2., 0., 16),
    'robust-cal': ('robust-cal', 15, 2., .50, 16),
    'aso-fp32': ('aso', 0, 4., 0., 1),
    'lite-pca32': ('lite32', 0, 4., 0., 16),
    'mh-pca64': ('mh64', 0, 4., 0., 16),
    'soft-int4': ('soft4', 15, 2., 0., 16),
    'uint4': ('uint4', 15, 2., 0., 16),
    'uint4-384': ('uint4-384', 15, 2., 0., 16),
    'fusion-pca64': ('pca64', 15, 2., 0., 16),
    'fusion-pca64-wide': ('pca64', 21, 3., 0., 16),
    'fusion-spatial': ('pca64-spatial', 15, 2., 0., 16),
    'layer2-pca96': ('pca96-low', 15, 2., 0., 16),
    'fusion-centroids': ('pca64-centroids', 15, 2., 0., 16),
}
SCREEN = tuple(VARIANTS)
VARIANTS.update({
    'robust-cal11': ('robust-cal', 11, 3., .50, 16),
    'robust-cal19': ('robust-cal', 19, 3.5, .50, 16),
    'robust-cal21': ('robust-cal', 21, 4., .75, 16),
    'robust-cal27': ('robust-cal', 27, 4., .25, 16),
    'robust-strong': ('robust', 17, 3., 0., 16),
    'uint4-strong': ('uint4', 19, 3., 0., 16),
    'uint4-32k': ('uint4-32k', 19, 3., 0., 16),
    'uint4-pos': ('uint4-pos', 19, 3., 0., 16),
    'uint4-cal11': ('uint4-cal', 11, 3., .50, 16),
    'uint4-cal19': ('uint4-cal', 19, 3.5, .50, 16),
    'uint4-cal27': ('uint4-cal', 27, 4., .25, 16),
    'uint4-32k-cal': ('uint4-32k-cal', 19, 3.5, .50, 16),
})
REFINE = tuple(name for name in VARIANTS if name not in SCREEN)
VARIANTS.update({
    'robust-coarse11': ('robust-coarse', 11, 3., .50, 16),
    'robust-coarse15': ('robust-coarse', 15, 3., .50, 16),
    'robust-coarse19': ('robust-coarse', 19, 3.5, .50, 16),
    'robust-coarse15-k24': ('robust-coarse', 15, 3., .50, 24),
    'robust-coarse19-k24': ('robust-coarse', 19, 3.5, .50, 24),
    'robust-coarse15-k32': ('robust-coarse', 15, 3., .50, 32),
    'robust-coarse19-k32': ('robust-coarse', 19, 3.5, .50, 32),
})
COMPACT = tuple(name for name in VARIANTS if name not in SCREEN + REFINE)
EARLY = ('layer1-dense', 'layer1-layer2', 'layer1-layer2-wide')
VARIANTS.update({
    'layer1-dense': ('early-layer1', 27, 2., 0., 32),
    'layer1-layer2': ('early-fusion', 27, 2., 0., 32),
    'layer1-layer2-wide': ('early-fusion', 37, 3., 0., 32),
})
METRICS = ('small_aupro_05', 'aupro_05', 'aupro_30', 'image_auroc')


def start_device():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    return torch.device('cuda:0')


def array_save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_bytes(data)
    tmp.replace(path)


def bank_path(output, category, seed, variant):
    return output / 'models' / category / str(seed) / f'{VARIANTS[variant][0]}.npz'


def bank_dimensions(variant):
    bank = VARIANTS[variant][0]
    return 512 if bank.startswith(('robust', 'uint4')) or BANKS.get(bank, ('full',))[0] == 'low' else 1536


def feature_space(rows, dimensions):
    if dimensions == 'fused':
        return torch.cat((rows[:, :512], rows[:, 512:] * .35), dim=1)
    return rows


def grid_coordinates(grid, device, nonnegative=False):
    axes = [torch.linspace(0 if nonnegative else -1, 1, side, device=device) for side in grid]
    return torch.stack(torch.meshgrid(*axes, indexing='ij'), dim=-1).reshape(-1, 2)


def spatial_weakness(pool, pos, npositions):
    # Distances to the two closest build patches at the same grid position.
    # The robust tail filter is inspired by SoftPatch's position-wise scoring.
    groups = pool.new_zeros((len(pool),))
    for index in range(npositions):
        ids = torch.from_numpy(np.flatnonzero(pos == index)).to(pool.device)
        if len(ids) < 3:
            continue
        block = pool.index_select(0, ids)
        distances = torch.cdist(block, block)
        distances.fill_diagonal_(torch.inf)
        local = distances.topk(2, dim=1, largest=False).values.mean(1)
        groups.index_copy_(0, ids, local / local.median().clamp_min(1e-6))
    return groups


def elbow_order(pool, seed):
    # ASO uses the nearest point to the origin after normalizing both axes.
    # Here the coverage distances come from exact greedy selection on our pool.
    count = max(2, len(pool) // 10)
    indices = []
    coverage = []
    index = torch.tensor([np.random.default_rng(seed).integers(len(pool))], device=pool.device)
    norms = pool.square().sum(1)
    best = torch.full_like(norms, torch.inf)
    for _ in range(count):
        indices.append(index)
        center = pool.index_select(0, index).flatten()
        best = torch.minimum(best, norms + center.square().sum() - 2 * torch.mv(pool, center))
        radius, next_index = best.max(0)
        coverage.append(radius)
        index = next_index.reshape(1)
    radii = torch.stack(coverage).clamp_min(0).sqrt()
    radii = (radii - radii[-1]) / (radii[0] - radii[-1]).clamp_min(1e-6)
    counts = torch.linspace(0, 1, count, device=pool.device)
    size = int((counts.square() + radii.square()).argmin()) + 1
    return torch.cat(indices[:size]).cpu().numpy()


def centroids(pool, order):
    centers = pool[order].clone()
    point_norms = pool.square().sum(1, keepdim=True)
    for _ in range(8):
        distances = point_norms + centers.square().sum(1)
        distances.addmm_(pool, centers.T, alpha=-2)
        labels = distances.argmin(1)
        sums = torch.zeros_like(centers).index_add_(0, labels, pool)
        counts = torch.bincount(labels, minlength=len(centers)).unsqueeze(1)
        centers = torch.where(counts > 0, sums / counts.clamp_min(1), centers)
    return centers


def fields(bank, **extra):
    with np.load(BytesIO(bank.serialize()), allow_pickle=False) as archive:
        values = dict(archive)
    values.update(extra)
    buffer = BytesIO()
    np.savez(buffer, **values)
    return buffer.getvalue()


class Scorer:
    def __init__(self, payload, variant, device, grid, *, decoded=False):
        self.kind, self.window, self.sigma, self.cal_weight, self.top_k = VARIANTS[variant]
        self.grid = grid
        with np.load(BytesIO(payload), allow_pickle=False) as archive:
            bank = Bank.from_fields(archive)
            self.center = torch.as_tensor(archive['cal_center'], device=device).float() if 'cal_center' in archive else None
            self.scale = torch.as_tensor(archive['cal_scale'], device=device).float() if 'cal_scale' in archive else None
            self.position_weight = float(archive['position_weight']) if 'position_weight' in archive else 0.
            self.point_weights = torch.as_tensor(archive['point_weights'], device=device).float() if 'point_weights' in archive else None
        if self.center is not None and self.center.ndim == 2:
            moments = torch.stack((self.center, self.scale)).unsqueeze(0)
            center, scale = F.interpolate(moments, grid, mode='bilinear', align_corners=False)[0]
            self.center, self.scale = center.flatten(), scale.flatten()
        self.transform = BANKS.get(self.kind, ('full',))[0]
        self.mean = torch.as_tensor(bank.mean, device=device)
        self.projection = torch.as_tensor(bank.projection, device=device).float()
        self.coords = grid_coordinates(grid, device, bank.unsigned) * self.position_weight if self.position_weight else None
        self.prepared = bank.prepare(device)
        blocks = tuple(unpack_rows(block, self.prepared.scale, bank.unsigned)
                       for block in self.prepared.bank.split(512))
        self.norms = tuple(block.square().sum(1) for block in blocks)
        self.decoded = blocks if decoded else None

    def raw(self, features, tile=784):
        prepared = self.prepared
        points = self.grid[0] * self.grid[1]
        if len(features) % points:
            raise ValueError('Query features must contain whole images')
        rows = []
        offset = 0
        for query in features.split(tile):
            query = feature_space(query, self.transform).contiguous()
            if self.projection.numel():
                query = (query - self.mean) @ self.projection
            if self.coords is not None:
                positions = torch.arange(offset, offset + len(query), device=query.device) % points
                query = torch.cat((query, self.coords.index_select(0, positions)), dim=1)
            norms = query.square().sum(1, keepdim=True)
            best = torch.full_like(norms, torch.inf)
            weights = torch.ones_like(norms) if self.point_weights is not None else None
            blocks = self.decoded if self.decoded is not None else prepared.bank.split(512)
            for j, (block, bank_norms) in enumerate(zip(blocks, self.norms, strict=True)):
                if self.decoded is None:
                    block = unpack_rows(block, prepared.scale, prepared.unsigned)
                distances = norms + bank_norms
                distances.addmm_(query, block.T, alpha=-2)
                if self.point_weights is None:
                    best = torch.minimum(best, distances.amin(1, keepdim=True))
                else:
                    value, ids = distances.min(1, keepdim=True)
                    weights = torch.where(value < best, self.point_weights[j * 512 + ids], weights)
                    best = torch.minimum(best, value)
            scores = best.clamp_min_(0).sqrt_()
            rows.append(scores if weights is None else scores * weights)
            offset += len(query)
        return torch.cat(rows).reshape(-1, points)

    def map_values(self, raw):
        values = raw
        if self.cal_weight:
            residual = raw - self.center
            # Preserve the raw score units while correcting easier/harder positions.
            corrected = residual * (self.scale.median() / self.scale)
            values = torch.lerp(raw, corrected, self.cal_weight)
        if self.window:
            strength = .90 if self.kind == 'robust' and self.window == 17 else .75
            values = adjust_scores(values, self.grid, self.window, strength)
        return values

    def output(self, raw, size):
        return image_score(raw, self.top_k), make_maps(self.map_values(raw), self.grid, size, self.sigma)


def train_bank(kind, pool, ids, grid, seed):
    representation, width, covariance, sampling, bits = BANKS[kind]
    pool = feature_space(pool, representation).contiguous()
    pos = ids % (grid[0] * grid[1])
    mean = basis = None
    position_weight = 0.
    if width:
        center = pool.mean(0)
        centered = pool - center
        _, singular, vectors = torch.pca_lowrank(centered, q=width + 12, center=False, niter=4)
        variance = singular[:width].square() / (len(pool) - 1)
        shrink = .93 * variance + .07 * variance.mean()
        scaling = (shrink / shrink.mean()).pow(-covariance)
        projection = (vectors[:, :width] * scaling).half().contiguous()
        pool = centered @ projection.float()
        mean, basis = center.cpu().numpy(), projection.cpu().numpy()
    if kind in ('pca64-spatial', 'uint4-pos'):
        multiplier = 1.10 if kind == 'uint4-pos' else .55
        position_weight = float(pool.std(0).square().sum().sqrt()) * multiplier
        coords = grid_coordinates(grid, pool.device, kind == 'uint4-pos') * position_weight
        pool = torch.cat((pool, coords[pos]), dim=1)
    count = 384 if sampling == 'small' else 740
    weights = None
    if sampling == 'elbow':
        order = elbow_order(pool, seed)
    elif sampling == 'denoise':
        weakness = spatial_weakness(pool, pos, grid[0] * grid[1])
        clean = torch.nonzero(weakness <= weakness.quantile(.95)).flatten()
        order = coreset(pool[clean], count, seed)
        pool = pool[clean]
        weights = torch.lerp(torch.ones_like(weakness[clean]), weakness[clean].clamp(.5, 2), .20)
    else:
        order = coreset(pool, count, seed)
    if sampling == 'centroids':
        pool = centroids(pool, order)
        order = np.arange(len(pool))
    model = fit_bank(pool, order, bits, mean=mean, projection=basis, unsigned=kind.startswith('uint4'))
    extra = {}
    if position_weight:
        extra['position_weight'] = np.float32(position_weight)
    if weights is not None:
        extra['point_weights'] = weights[order].cpu().numpy().astype(np.float16)
    return fields(model, **extra) if extra else model.serialize()


def add_calibration(payload, scorer, cal, grid, coarse):
    center, scale = calibrate(scorer.raw(cal))
    if coarse:
        moments = torch.stack((center, scale)).reshape(1, 2, *grid)
        moments = F.avg_pool2d(moments, 2, 2)
        center, scale = moments[0]
    with np.load(BytesIO(payload), allow_pickle=False) as archive:
        values = dict(archive)
    values.update(cal_center=center.cpu().numpy().astype(np.float16),
                  cal_scale=scale.cpu().numpy().astype(np.float16))
    stream = BytesIO()
    with ZipFile(stream, 'w', compression=ZIP_LZMA) as archive:
        for key, value in values.items():
            with archive.open(key + '.npy', 'w') as member:
                np.lib.format.write_array(member, value, allow_pickle=False)
    return stream.getvalue()


def read_cache(cache_dir, category, split, manifest, dimensions=1536):
    # Validate IDs and manifests against the same image-level split for every fit.
    path = cache_dir / category / f'{split}.npz'
    with np.load(path, allow_pickle=False) as archive:
        cached = FeatureCache(archive['features'], archive['image_ids'], str(archive['split'].item()),
                              json.loads(str(archive['metadata_json'].item())),
                              archive['labels'] if 'labels' in archive else None,
                              archive['masks'] if 'masks' in archive else None)
    records = [x for x in manifest['records'] if x['split'] == split]
    fingerprint = sha256(cache_dir / f'{category}.json')
    extractor = json.loads((cache_dir / category / 'extraction.json').read_text())
    if (cached.split != split or cached.metadata['manifest_sha256'] != fingerprint
            or cached.metadata['category'] != category
            or cached.metadata['feature_signature'] != extractor['feature_signature']
            or list(cached.image_ids) != [x['image_id'] for x in records]):
        raise ValueError(f'Cache identity mismatch for {category} {split}')
    if dimensions < cached.features.shape[-1]:
        cached.features = np.ascontiguousarray(cached.features[..., :dimensions])
    return cached


def fit_or_load(output, category, seed, variant, features, cal, grid, device):
    destination = bank_path(output, category, seed, variant)
    if destination.exists():
        return destination.read_bytes()
    kind = VARIANTS[variant][0]
    if kind.endswith('-cal') and kind != 'robust-cal':
        base = kind.removesuffix('-cal')
        payload = fit_or_load(output, category, seed, base, features, cal, grid, device)
        payload = add_calibration(payload, Scorer(payload, base, device, grid), cal, grid, coarse=True)
        array_save(destination, payload)
        return payload
    if kind == 'fp32-10pct':
        payload = (WORK / 'confirm' / category / str(seed) / 'fp32-10pct-0.npz').read_bytes()
    elif kind.startswith('robust'):
        folder = WORK / 'lossless-confirm'
        codec = json.loads((folder / 'compression.json').read_text())['codec']
        payload = (folder / category / f'{seed}-{codec}.npz').read_bytes()
        if kind == 'robust-coarse':
            scorer = Scorer(payload, 'robust-11', device, grid)
            payload = add_calibration(payload, scorer, cal, grid, coarse=True)
        elif kind == 'robust-cal':
            scorer = Scorer(payload, 'robust-11', device, grid)
            center, scale = calibrate(scorer.raw(cal))
            # Store only the two new arrays alongside the original packed archive.
            buffer = BytesIO()
            with np.load(BytesIO(payload), allow_pickle=False) as archive:
                original = dict(archive)
            np.savez_compressed(buffer, **original, cal_center=center.cpu().numpy().astype(np.float16),
                                cal_scale=scale.cpu().numpy().astype(np.float16))
            payload = buffer.getvalue()
    else:
        dims = bank_dimensions(variant)
        rng = np.random.default_rng(seed)
        limit = 32768 if kind == 'uint4-32k' else 8192
        ids = np.sort(rng.choice(len(features), min(limit, len(features)), replace=False))
        pool = torch.as_tensor(np.ascontiguousarray(features[ids, :dims]), device=device)
        torch.manual_seed(seed)
        payload = train_bank(kind, pool, ids, grid, seed)
    array_save(destination, payload)
    return payload


def save_csv(path, rows):
    if rows:
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
            writer.writeheader()
            writer.writerows(rows)


def evaluate(args):
    device = start_device()
    cache = WORK / 'cache'
    for category in args.categories:
        manifest_path = cache / f'{category}.json'
        manifest = load_manifest(manifest_path)
        fingerprint = sha256(manifest_path)
        completed = args.output / 'quality' / category
        expected = [(seed, variant) for seed in args.seeds for variant in args.variants]
        pending = [(seed, name) for seed, name in expected if not (completed / f'{seed}-{name}.json').exists()]
        if not pending:
            continue
        dims = max(bank_dimensions(variant) for _, variant in pending)
        test = read_cache(cache, category, 'test', manifest, dims)
        test_rows = test.rows()
        evaluator = Localization(test.masks).on_device(device)
        grid, size = test.features.shape[1:3], test.masks.shape[1:]
        build = None
        if any(VARIANTS[name][0].removesuffix('-cal') in BANKS and not bank_path(args.output, category, seed, name).exists()
               for seed, name in pending):
            build = read_cache(cache, category, 'build', manifest, dims).rows()
        cal = None
        if any(VARIANTS[name][3] for _, name in pending):
            cal_data = read_cache(cache, category, 'calibration', manifest, 512)
            cal = torch.as_tensor(cal_data.rows(), device=device)
            del cal_data
        all_features = torch.as_tensor(test_rows, device=device)
        last_bank, last_raw = None, None
        for seed, name in pending:
            path = bank_path(args.output, category, seed, name)
            with measure(device) as build_measured:
                payload = fit_or_load(args.output, category, seed, name, build, cal, grid, device)
            dims = bank_dimensions(name)
            with measure(device) as inference:
                scorer = Scorer(payload, name, device, grid)
                if last_bank != path:
                    last_raw = scorer.raw(all_features[:, :dims], tile=4096)
                    last_bank = path
                image, maps = scorer.output(last_raw, size)
            result = dict(category=category, seed=seed, method=name, test_images=len(test.labels),
                          manifest_sha256=fingerprint, artifact_bytes=len(payload),
                          patches=len(scorer.prepared.bank),
                          image_auroc=binary_auroc(test.labels, image.cpu().numpy()),
                          **evaluator.evaluate_device(maps),
                          **{f'cached_{key}': val for key, val in inference.items()},
                          **{f'fit_or_load_{key}': val for key, val in build_measured.items()})
            array_save(completed / f'{seed}-{name}.json', json.dumps(result).encode())
            print(category, seed, name, len(payload), *(round(result[key] * 100, 2) for key in METRICS), flush=True)
            del scorer, image, maps
        del build, cal, all_features, test_rows, test, evaluator, last_raw
        torch.cuda.empty_cache()
    rows = [json.loads(p.read_text()) for p in sorted((args.output / 'quality').glob('*/*.json'))]
    save_csv(args.output / 'quality.csv', rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'results')
    parser.add_argument('--categories', nargs='+', choices=CATEGORIES, default=CATEGORIES)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0])
    parser.add_argument('--variants', nargs='+', choices=SCREEN + REFINE + COMPACT)
    stages = parser.add_mutually_exclusive_group()
    stages.add_argument('--refine', action='store_true', help='run the second-stage settings')
    stages.add_argument('--compact', action='store_true', help='run coarse calibration settings')
    args = parser.parse_args()
    if args.variants and (args.refine or args.compact):
        parser.error('--variants cannot be combined with a stage flag')
    args.variants = args.variants or (COMPACT if args.compact else REFINE if args.refine else SCREEN)
    with torch.inference_mode():
        evaluate(args)


if __name__ == '__main__':
    main()
