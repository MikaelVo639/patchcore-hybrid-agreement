"""Paired PatchCore compression experiments on the local MVTec AD feature cache."""

import argparse
import csv
from contextlib import contextmanager
import json
from pathlib import Path
import threading
from time import perf_counter

import numpy as np
import psutil
from scipy import ndimage
import torch

from compression import Bank, fit_bank, residual_norm
from patchcore_budget.cache import FeatureCache, check_compatible
from patchcore_budget.io import sha256
from patchcore_budget.metrics import binary_auroc


BUDGETS = (512 * 1024, 1024 * 1024)
METHODS = (
    'fp32-budget', 'fp16-full', 'int8-full', 'int4-full', 'pca64-int8', 'pca128-int8',
    'pca64-int8-residual', 'pca128-int8-residual', 'uint4-full',
)


@contextmanager
def measure(device):
    process = psutil.Process()
    results = {'rss_start_bytes': process.memory_info().rss}
    readings = [results['rss_start_bytes']]
    stop = threading.Event()

    def monitor():
        while not stop.wait(0.01):
            readings.append(process.memory_info().rss)

    cuda = device.type == 'cuda'
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        results['vram_start_bytes'] = torch.cuda.memory_allocated(device)
    timer = threading.Thread(target=monitor, daemon=True)
    timer.start()
    started = perf_counter()
    try:
        yield results
    finally:
        if cuda:
            torch.cuda.synchronize(device)
            results.update(vram_peak_bytes=torch.cuda.max_memory_allocated(device),
                           vram_reserved_peak_bytes=torch.cuda.max_memory_reserved(device))
        results['seconds'] = perf_counter() - started
        stop.set()
        timer.join()
        results['rss_peak_bytes'] = max(*readings, process.memory_info().rss)


def coreset(pool, count, seed):
    """Exact greedy farthest-first traversal with O(pool) auxiliary memory."""
    selected = np.empty(count, dtype=np.int64)
    first = np.random.default_rng(seed).integers(len(pool))
    norms = torch.cat([block.square().sum(1) for block in pool.split(512)])
    best = torch.full_like(norms, torch.inf)
    index = torch.as_tensor([first], device=pool.device)
    chosen = []
    for _ in range(count):
        chosen.append(index)
        center = pool.index_select(0, index).reshape(-1)
        distance = norms + center.square().sum() - 2 * torch.mv(pool, center)
        best = torch.minimum(best, distance)
        best.index_fill_(0, index, -torch.inf)
        index = best.argmax().reshape(1)
    selected[:] = torch.cat(chosen).cpu().numpy()
    return selected


def projected_pool(pool, dimensions):
    mean = pool.mean(0)
    centered = pool - mean
    _, _, vectors = torch.pca_lowrank(centered, q=dimensions + 8, center=False, niter=4)
    basis = vectors[:, :dimensions].half().contiguous()
    projected = centered @ basis.float()
    return projected, mean.cpu().numpy(), basis.cpu().numpy(), residual_norm(centered, projected)


def build_model(pool, name, budget, seed, projection_cache, order_cache):
    if name.startswith('pca'):
        dimensions = int(name.split('-')[0][3:])
        if dimensions not in projection_cache:
            projection_cache[dimensions] = projected_pool(pool, dimensions)
        model_pool, mean, basis, residual = projection_cache[dimensions]
        residual = residual if name.endswith('-residual') else None
        bits = 8
    else:
        dimensions = pool.shape[1]
        model_pool, mean, basis, residual = pool, None, None, None
        bits = {'fp32-10pct': 32, 'fp32-budget': 32, 'fp16-full': 16, 'int8-full': 8,
                'int4-full': 4, 'uint4-full': 4}[name]
    if dimensions not in order_cache:
        max_count = min(len(pool), BUDGETS[-1] // dimensions)
        if dimensions == pool.shape[1]:
            max_count = max(max_count * 2 + 2, len(pool) // 10)
        # When the full pool fits the budget, every patch is kept.
        exact_count = min(max_count, len(pool))
        if exact_count == len(pool) and basis is not None:
            overhead = basis.nbytes + mean.nbytes + 1024
            exact_count = max(1, min(exact_count, (BUDGETS[-1] - overhead) // dimensions))
            if exact_count == len(pool) and overhead + (dimensions + 2) * len(pool) + 4 * dimensions + 512 < BUDGETS[-1]:
                exact_count = max(1, min(exact_count, (BUDGETS[0] - overhead) // dimensions))
        prefix = coreset(model_pool, exact_count, seed)
        order_cache[dimensions] = np.r_[prefix, np.setdiff1d(np.arange(len(pool)), prefix)]
    order = order_cache[dimensions]
    if name == 'fp32-10pct':
        order = order[:max(1, len(pool) // 10)]
    model = fit_bank(model_pool, order, bits, budget, mean=mean, projection=basis,
                     residual=residual, unsigned=name == 'uint4-full')
    return model


class Localization:
    def __init__(self, masks):
        self.foreground = masks.astype(bool).ravel()
        region_ids = np.empty(masks.shape, dtype=np.int32)
        offset = 0
        for index, mask in enumerate(masks):
            regions, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
            region_ids[index] = np.where(regions > 0, regions + offset, 0)
            offset += count
        region_ids = region_ids.ravel()
        area = np.bincount(region_ids, minlength=offset + 1)
        if offset < 1 or not area[0]:
            raise ValueError('AUPRO requires anomaly regions and background pixels')
        small = area[1:] <= np.quantile(area[1:], 0.25)
        self.small_count = int(small.sum())
        self.region_count = offset
        weights = np.zeros((2, offset + 1), dtype=np.float64)
        weights[0, 1:] = 1 / (area[1:] * offset)
        weights[1, 1:] = small / (area[1:] * self.small_count)
        self.weights = weights
        self.region_ids = region_ids
        self.background_count = int(area[0])

    def evaluate(self, maps):
        scores = maps.ravel()
        order = np.argsort(-scores, kind='stable')
        scores = scores[order]
        regions = self.region_ids[order]
        negatives = np.cumsum(regions == 0)
        end = min(len(order) - 1, np.searchsorted(negatives, self.background_count * 0.3, side='right'))
        end = min(len(order), np.searchsorted(-scores, -scores[end], side='right'))
        scores, regions, negatives = scores[:end], regions[:end], negatives[:end]
        ties = np.r_[np.flatnonzero(scores[1:] != scores[:-1]), len(scores) - 1]
        fpr = np.r_[0., negatives[ties] / self.background_count]
        overlap = np.c_[np.zeros(2), np.cumsum(self.weights[:, regions], axis=1)[:, ties]]

        def integrate(row, cap):
            before = fpr <= cap
            x = np.r_[fpr[before], cap]
            y = np.r_[row[before], np.interp(cap, fpr, row)]
            return float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2) / cap)

        return dict(aupro_05=integrate(overlap[0], .05), aupro_30=integrate(overlap[0], .30),
                    small_aupro_05=integrate(overlap[1], .05), regions=self.region_count,
                    small_regions=self.small_count)

    def on_device(self, device):
        self.device_regions = torch.as_tensor(self.region_ids, device=device)
        self.device_weights = torch.as_tensor(self.weights, device=device)
        return self

    def evaluate_device(self, maps):
        scores = torch.as_tensor(maps, device=self.device_regions.device).flatten()
        scores, order = scores.sort(descending=True)
        regions = self.device_regions[order]
        negatives = (regions == 0).cumsum(0, dtype=torch.int32)
        device = scores.device
        boundary = torch.tensor(int(self.background_count * 0.3), device=device, dtype=negatives.dtype)
        end = min(len(order) - 1, torch.searchsorted(negatives, boundary, right=True).item())
        end = torch.searchsorted(-scores, -scores[end], right=True).item()
        scores, regions, negatives = scores[:end], regions[:end], negatives[:end]
        ties = torch.cat(((scores[1:] != scores[:-1]).nonzero().flatten(),
                          torch.tensor([end - 1], device=device)))
        fpr = torch.cat((torch.zeros(1, device=device, dtype=torch.float64),
                         negatives[ties].double() / self.background_count))
        overlap = torch.cat((torch.zeros((2, 1), device=device, dtype=torch.float64),
                             self.device_weights[:, regions].cumsum(1)[:, ties]), dim=1)

        def integrate(row, cap):
            target = torch.tensor(cap, device=device, dtype=fpr.dtype)
            index = torch.searchsorted(fpr, target, right=True).item() - 1
            x, y = fpr[:index + 1], row[:index + 1]
            area = ((x[1:] - x[:-1]) * (y[1:] + y[:-1]) * .5).sum()
            width = cap - fpr[index]
            if index + 1 < len(fpr):
                endpoint = row[index] + (row[index + 1] - row[index]) * width / (fpr[index + 1] - fpr[index])
                area = area + width * (row[index] + endpoint) * .5
            return area.item() / cap

        return dict(aupro_05=integrate(overlap[0], .05), aupro_30=integrate(overlap[0], .30),
                    small_aupro_05=integrate(overlap[1], .05), regions=self.region_count,
                    small_regions=self.small_count)


def anomaly_maps(scores, shape, size):
    maps = torch.nn.functional.interpolate(scores.reshape(-1, 1, *shape), size,
                                            mode='bilinear', align_corners=True)
    return ndimage.gaussian_filter(maps[:, 0].cpu().numpy(), sigma=(0, 4, 4), mode='nearest', truncate=3)


def run_category(cache, output, seeds, device, methods, large_baseline):
    build = FeatureCache.load(cache / 'build.npz', 'build')
    cal = FeatureCache.load(cache / 'calibration.npz', 'calibration')
    test = FeatureCache.load(cache / 'test.npz', 'test')
    check_compatible(build, cal, test)
    category = build.metadata['category']
    manifest = cache.parent / f'{category}.json'
    if sha256(manifest) != build.metadata['manifest_sha256']:
        raise ValueError('Feature cache manifest hash differs from manifest on disk')
    grid = test.features.shape[1:3]
    full_test = torch.from_numpy(test.rows()).to(device)
    calibration_rows = cal.rows()
    local = Localization(test.masks).on_device(device)
    for seed in seeds:
        destination = output / category / str(seed)
        results = destination / 'metrics.json'
        if results.exists():
            continue
        destination.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        ids = np.sort(rng.choice(len(build.rows()), min(8192, len(build.rows())), replace=False))
        cal_ids = np.sort(rng.choice(len(calibration_rows), min(4096, len(calibration_rows)), replace=False))
        pool = torch.as_tensor(build.rows()[ids], device=device)
        calibration = torch.as_tensor(calibration_rows[cal_ids], device=device)
        ref = Bank(pool.cpu().numpy(), np.empty(0), np.empty((0, 0)), np.empty(0), np.empty(0))
        reference = ref.score(calibration)
        projections, orders, configured, rows = {}, {}, [], []
        settings = [('fp32-10pct', None)] if large_baseline else []
        settings.extend((name, cap) for cap in BUDGETS for name in methods)
        for name, budget in settings:
            with measure(device) as measurement:
                model = build_model(pool, name, budget, seed, projections, orders)
                payload = model.serialize()
            artifact = destination / f'{name}-{budget or 0}.npz'
            artifact.write_bytes(payload)
            if budget is not None and artifact.stat().st_size > budget:
                raise ValueError('Artifact exceeds its budget')
            model = Bank.deserialize(artifact.read_bytes())
            error = torch.quantile((model.score(calibration) - reference).abs(), 0.95).item()
            row = dict(category=category, seed=seed, method=name, budget=budget or 0,
                       artifact_bytes=artifact.stat().st_size, patches=len(model.bank),
                       dimension=model.dimension, calibration_error_q95=error,
                       artifact=str(artifact.relative_to(output)), manifest_sha256=sha256(manifest),
                       feature_signature=build.metadata['feature_signature'], test_images=len(test.labels),
                       build_images=len(build.image_ids), calibration_images=len(cal.image_ids),
                       **{f'shared_build_{k}': v for k, v in measurement.items()})
            configured.append((model, row))
        # Persist candidate construction and calibration before scoring official test images.
        (destination / 'calibration.json').write_text(json.dumps([row for _, row in configured], indent=2))
        for model, row in configured:
            with measure(device) as measurement:
                scores = model.score(full_test)
                image_scores = scores.reshape(len(test.labels), -1).amax(1).cpu().numpy()
                maps = anomaly_maps(scores, grid, test.masks.shape[1:])
            row.update(local.evaluate_device(maps))
            row.update({f'cached_batch_{k}': v for k, v in measurement.items()})
            row['image_auroc'] = binary_auroc(test.labels, image_scores)
            rows.append(row)
            print(category, seed, row['method'], row['budget'], f"small PRO={row['small_aupro_05']:.4f}", flush=True)
        staging = results.with_suffix('.tmp')
        staging.write_text(json.dumps(rows, indent=2))
        staging.replace(results)


def export(output):
    rows = [row for path in sorted(output.glob('*/*/metrics.json')) for row in json.loads(path.read_text())]
    if rows:
        columns = list(dict.fromkeys(key for row in rows for key in row))
        with (output / 'results.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0])
    parser.add_argument('--methods', choices=METHODS, nargs='+', default=METHODS)
    parser.add_argument('--categories', nargs='+')
    parser.add_argument('--omit-large-baseline', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    caches = sorted(path for path in args.cache.iterdir() if path.is_dir())
    if args.categories:
        caches = [args.cache / category for category in args.categories]
    for path in caches:
        if not (path / 'extraction.json').is_file():
            continue
        with torch.inference_mode():
            run_category(path, args.output, args.seeds, torch.device(args.device),
                         args.methods, not args.omit_large_baseline)
        export(args.output)


if __name__ == '__main__':
    main()
