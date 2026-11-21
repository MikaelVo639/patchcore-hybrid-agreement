"""Score-level PatchCore improvements with a packed bank and normal-only calibration."""

import argparse
from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from torch.nn import functional as F

from benchmark import Localization, export, measure
from compression import Bank, unpack_rows
from patchcore_budget.cache import FeatureCache, check_compatible
from patchcore_budget.io import sha256
from patchcore_budget.metrics import binary_auroc


CAP = 1024 * 1024
# Each setting specifies feature metric, k, map normalization, and smoothing sigma.
SETTINGS = {
    'baseline': ('l2', 1, 'none', 4),
    'position': ('l2', 1, 'position', 4),
    'position-center': ('l2', 1, 'position-center', 4),
    'position-mean': ('l2', 1, 'position-mean', 4),
    'image-center': ('l2', 1, 'image-center', 4),
    'position-k3': ('l2', 3, 'position', 4),
    'position-sigma2': ('l2', 1, 'position', 2),
    'position-layer2': ('layer2', 1, 'position', 4),
    'layer2': ('layer2', 1, 'none', 4),
    'position-layer-mix': ('layer-mix', 1, 'position', 4),
    'position-cosine': ('cosine', 1, 'position', 4),
    'cosine': ('cosine', 1, 'none', 4),
    'local-contrast': ('l2', 1, 'local-contrast', 4),
    'position-contrast': ('l2', 1, 'position-contrast', 4),
    'layer2-contrast': ('layer2', 1, 'local-contrast', 4),
}


def transform(rows, mode):
    if mode == 'layer2':
        return rows[:, :512].contiguous()
    if mode == 'layer-mix':
        return torch.cat((rows[:, :512], 0.5 * rows[:, 512:]), dim=1)
    if mode == 'cosine':
        return F.normalize(rows, dim=1)
    if mode != 'l2':
        raise ValueError(f'Unknown feature metric {mode}')
    return rows


def patch_scores(query, decoded_bank, mode, k=1, bank_block=512):
    bank = lambda start: transform(decoded_bank[start:start + bank_block], mode)
    return search(query, bank, len(decoded_bank), mode, k, bank_block)


def search(query, bank, count, mode, k, bank_block):
    parts = []
    for rows in query.split(4096):
        rows = transform(rows, mode)
        norms = rows.square().sum(1, keepdim=True)
        best = rows.new_full((len(rows), k), torch.inf)
        for start in range(0, count, bank_block):
            block = bank(start)
            distances = norms + block.square().sum(1)
            distances.addmm_(rows, block.T, alpha=-2)
            if k == 1:
                best = torch.minimum(best, distances.amin(1, keepdim=True))
            else:
                near = distances.topk(min(k, len(block)), dim=1, largest=False).values
                best = torch.cat((best, near), dim=1).topk(k, dim=1, largest=False).values
        parts.append(best.clamp_min_(0).sqrt_().mean(1))
    return torch.cat(parts)


def calibrate(normal):
    center = normal.mean(0)
    spread = normal.std(0, unbiased=False)
    # Shrink noisy patch-level variance estimates toward the typical variance.
    typical = spread.square().median().clamp_min(1e-6)
    spread = (0.5 * spread.square() + 0.5 * typical).sqrt()
    return center, spread


def patch_median(values):
    return values.sort(dim=1).values[:, (values.shape[1] - 1) // 2: (values.shape[1] + 1) // 2]


def local_contrast(values):
    side = int(values.shape[1] ** 0.5)
    maps = values.reshape(-1, 1, side, side)
    background = F.avg_pool2d(F.pad(maps, (3, 3, 3, 3), mode='replicate'), 7, stride=1)
    return values - 0.75 * background.flatten(1)


def rescale(raw, center, spread, mode):
    if mode in ('none', 'image-center'):
        return raw if mode == 'none' else raw - patch_median(raw)
    if mode == 'local-contrast':
        return local_contrast(raw)
    residual = raw - center
    if mode == 'position-mean':
        return residual
    values = residual / spread
    if mode == 'position-center':
        return values - patch_median(values)
    if mode == 'position-contrast':
        return local_contrast(values)
    if mode != 'position':
        raise ValueError(f'Unknown score normalization {mode}')
    return values


@dataclass
class Detector:
    bank: Bank
    method: str
    center: np.ndarray
    scale: np.ndarray

    def serialize(self):
        with np.load(BytesIO(self.bank.serialize()), allow_pickle=False) as bank:
            fields = dict(bank)
        fields.update(method=self.method, position_center=self.center, position_scale=self.scale)
        buffer = BytesIO()
        np.savez(buffer, **fields)
        return buffer.getvalue()

    @classmethod
    def load(cls, payload):
        with np.load(BytesIO(payload), allow_pickle=False) as metadata:
            bank = Bank.from_fields(metadata)
            return cls(bank, str(metadata['method'].item()), metadata['position_center'], metadata['position_scale'])

    def prepare(self, device):
        return (self.bank.prepare(device), torch.as_tensor(self.center, device=device).float(),
                torch.as_tensor(self.scale, device=device).float())

    def infer(self, features, prepared, grid=(28, 28), size=(224, 224)):
        packed, center, spread = prepared
        metric, k, normalization, sigma = SETTINGS[self.method]
        def bank_block(start):
            return transform(unpack_rows(packed.bank[start:start + 512], packed.scale, packed.unsigned), metric)
        scores = search(features, bank_block, len(packed.bank), metric, k, 512)
        values = rescale(scores.reshape(-1, grid[0] * grid[1]), center, spread, normalization)
        image_values = scores.reshape_as(values) if 'contrast' in normalization else values
        return image_values.amax(1), make_maps(values, grid, size, sigma)


def make_maps(scores, shape, size, sigma):
    maps = F.interpolate(scores.reshape(-1, 1, *shape), size, mode='bilinear', align_corners=True)
    return ndimage.gaussian_filter(maps[:, 0].cpu().numpy(), sigma=(0, sigma, sigma), mode='nearest', truncate=3)


def fit_detector(bank, method, calibration, grid):
    metric, k, normalization, _ = SETTINGS[method]
    n_positions = grid[0] * grid[1]
    need = normalization.startswith('position')
    stub = np.zeros(n_positions if need else 0, np.float16)
    detector = Detector(bank, method, stub, stub)
    overflow = len(detector.serialize()) - CAP
    removed = max(0, (overflow + bank.bank.shape[1] - 1) // bank.bank.shape[1])
    if removed:
        detector.bank = Bank(bank.bank[:-removed], bank.mean, bank.projection, bank.scale, bank.residual, bank.unsigned)
    if metric == 'layer2':
        # The discarded channels never contribute to the layer2 distance.
        packed = detector.bank
        detector.bank = Bank(np.ascontiguousarray(packed.bank[:, :256]), packed.mean, packed.projection,
                             np.ascontiguousarray(packed.scale[:512]), packed.residual, packed.unsigned)
    if need:
        scores = patch_scores(calibration, detector.bank.decoded(calibration.device), metric, k)
        center, spread = calibrate(scores.reshape(-1, n_positions))
        detector.center = center.cpu().numpy().astype(np.float16)
        detector.scale = spread.cpu().numpy().astype(np.float16)
    if len(detector.serialize()) > CAP:
        raise ValueError('Detector exceeds bank cap')
    return detector


def run_category(path, prior, output, seeds, methods, device):
    cal = FeatureCache.load(path / 'calibration.npz', 'calibration')
    test = FeatureCache.load(path / 'test.npz', 'test')
    check_compatible(cal, test)
    category = path.name
    manifest_hash = sha256(path.parent / f'{category}.json')
    if manifest_hash != test.metadata['manifest_sha256']:
        raise ValueError('Feature cache manifest hash changed')
    grid = test.features.shape[1:3]
    size = test.masks.shape[1:]
    scorer = Localization(test.masks).on_device(device)
    test_rows = torch.as_tensor(test.rows(), device=device)
    cal_rows = torch.as_tensor(cal.rows(), device=device)
    for seed in seeds:
        destination = output / category / str(seed)
        result = destination / 'metrics.json'
        if result.exists():
            continue
        destination.mkdir(parents=True, exist_ok=True)
        prior_run = prior / category / str(seed)
        bank_payload = (prior_run / f'int4-full-{CAP}.npz').read_bytes()
        prior_metrics = next(r for r in json.loads((prior_run / 'metrics.json').read_text())
                             if r['method'] == 'int4-full' and r['budget'] == CAP)
        original = Bank.deserialize(bank_payload)
        rows = []
        for method in ('baseline', *methods):
            if method == 'baseline':
                current = Detector(original, method, np.empty(0, np.float16), np.empty(0, np.float16))
                data = bank_payload
            else:
                current = fit_detector(original, method, cal_rows, grid)
                data = current.serialize()
                (destination / f'{method}.npz').write_bytes(data)
                current = Detector.load(data)
            metric, k, norm, sigma = SETTINGS[method]
            with measure(device) as phase:
                raw = patch_scores(test_rows, current.bank.decoded(device), metric, k, bank_block=4096)
                center = torch.as_tensor(current.center, device=device).float()
                spread = torch.as_tensor(current.scale, device=device).float()
                values = rescale(raw.reshape(len(test.labels), -1), center, spread, norm)
                maps = make_maps(values, grid, size, sigma)
            row = scorer.evaluate_device(maps)
            row.update(category=category, seed=seed, method=method, bytes=len(data), patches=len(current.bank.bank),
                       image_auroc=binary_auroc(test.labels, (raw.reshape_as(values) if 'contrast' in norm else values).amax(1).cpu().numpy()),
                       previous_small_aupro=prior_metrics['small_aupro_05'], manifest_sha256=manifest_hash,
                       **{f'batch_{key}': value for key, value in phase.items()})
            if method == 'baseline' and abs(row['small_aupro_05'] - row['previous_small_aupro']) > 1e-5:
                raise ValueError(f'Baseline mismatch for {category} {seed}: {row}')
            print(category, seed, method, round(row['small_aupro_05'] * 100, 3), flush=True)
            rows.append(row)
        staging = result.with_suffix('.tmp')
        staging.write_text(json.dumps(rows, indent=2))
        staging.replace(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('cache', 'prior', 'output'):
        parser.add_argument(f'--{key}', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0])
    parser.add_argument('--methods', choices=list(SETTINGS)[1:], nargs='+', default=list(SETTINGS)[1:])
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    args = parser.parse_args()
    for path in sorted(args.cache.iterdir()):
        if (path / 'extraction.json').exists():
            run_category(path, args.prior, args.output, args.seeds, args.methods, torch.device('cuda:0'))
            export(args.output)


if __name__ == '__main__':
    with torch.inference_mode():
        main()
