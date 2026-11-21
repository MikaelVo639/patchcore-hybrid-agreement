"""Fit compact DINOv2 PatchCore banks and cache matched raw scores."""

import argparse
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZIP_LZMA, ZipFile

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as T

from benchmark import coreset
from compression import Bank, fit_bank
from deployment import create_encoder, encode_tensors
from dual_head import DENSE, DualHead, embeddings
from experiment import WORK, array_save, start_device
from fast_inference import ordered_prefetch
from patchcore_budget.datasets import load_manifest, resolve_dataset_path
from patchcore_budget.io import sha256
from summarize import CATEGORIES

ROOT = Path(__file__).resolve().parent
OLD = WORK / 'PatchCore-Comparison' / 'results'
RESULTS = ROOT / 'results'
CHECKPOINT = WORK.parent / '.torch-cache' / 'hub' / 'checkpoints' / 'vit_small_patch14_dinov2.lvd142m.safetensors'
CHECKPOINT_SHA = '04d27f3400d059fc0cfd7d17dd1909a75bf3ea8fb3eeb48b97cb99e57ee20081'
SIDE = 28
TOKENS = SIDE * SIDE
DINO = ('mid', 'deep')
HEADS = ('fine', 'coarse', *DINO)


def manifest_parts(data, category):
    path = WORK / 'cache' / f'{category}.json'
    records = load_manifest(path)['records']
    paths = {
        split: [resolve_dataset_path(data, row['path']) for row in records if row['split'] == split]
        for split in ('build', 'calibration', 'test')
    }
    return sha256(path), paths, [r['image_id'] for r in records if r['split'] == 'test']


def crop(rgb, resize, crop_side):
    image = T.center_crop(T.resize(rgb, resize, InterpolationMode.BILINEAR, antialias=True), crop_side)
    return T.normalize(T.to_tensor(image), [.485, .456, .406], [.229, .224, .225])


def views(path, need_wrn=True, need_dino=True):
    with Image.open(path) as raw:
        rgb = raw if raw.mode == 'RGB' else raw.convert('RGB')
        return (crop(rgb, 256, 224) if need_wrn else None,
                crop(rgb, 448, 392) if need_dino else None)


def batches(paths, batch_size):
    for start in range(0, len(paths), batch_size):
        yield paths[start:start + batch_size]


def make_dino(device):
    import timm
    from safetensors.torch import load_file

    if sha256(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError('DINOv2 checkpoint checksum mismatch')
    encoder = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=False, dynamic_img_size=True)
    encoder.load_state_dict(load_file(str(CHECKPOINT)), strict=True)
    return encoder.eval().requires_grad_(False).to(device, dtype=torch.float16)


def encode_dino(inputs, encoder, device):
    features = encoder.get_intermediate_layers(inputs.to(device, dtype=torch.float16), n=(5, 11), norm=True)
    middle, final = (F.normalize(block.float(), dim=-1).flatten(0, 1) for block in features)
    return (F.normalize(middle * .70 + final * .30, dim=1),
            F.normalize(middle * .25 + final * .75, dim=1))


class CosineBank:
    def __init__(self, payload, device):
        with ZipFile(BytesIO(payload)) as archive:
            self.bank = F.normalize(Bank.deserialize(archive.read('bank.npz')).decoded(device), dim=1)

    def scores(self, query):
        """Exact cosine search over the decoded UINT4/INT4 representation."""
        return torch.cat([(1 - (q @ self.bank.T).amax(1)).clamp_min_(0.)
                          for q in query.split(2048)]).reshape(-1, TOKENS)


def dino_path(output, category, seed, head):
    return output / 'banks' / category / str(seed) / f'dino-{head}.zip'


def wrn_path(category, seed):
    return OLD / 'models' / category / str(seed) / 'dual-head.zip'


def train(output, category, paths, seeds, model, device):
    pending = {(seed, head) for seed in seeds for head in DINO
               if not dino_path(output, category, seed, head).exists()}
    if not pending:
        return
    if not paths:
        raise ValueError('Empty build split')
    indices = {
        seed: np.sort(np.random.default_rng(seed).choice(len(paths) * TOKENS,
                      min(len(paths) * TOKENS, 16384), replace=False))
        for seed, _ in pending
    }
    pools = {(seed, head): torch.empty((len(indices[seed]), 384), device=device) for seed, head in pending}
    offset = 0

    def prepare(batch):
        return torch.stack([views(path, need_wrn=False)[1] for path in batch])

    for inputs in ordered_prefetch(batches(paths, 6), prepare, workers=3):
        all_rows = dict(zip(DINO, encode_dino(inputs, model, device), strict=True))
        end = offset + len(inputs) * TOKENS
        for (seed, head), pool in pools.items():
            ids = indices[seed]
            lo, hi = np.searchsorted(ids, (offset, end))
            pool[lo:hi] = all_rows[head][torch.as_tensor(ids[lo:hi] - offset, device=device)]
        offset = end
    for (seed, head), pool in pools.items():
        order = coreset(pool, 1024, seed)
        raw = fit_bank(pool, order, 4).serialize()
        buffer = BytesIO()
        with ZipFile(buffer, 'w') as archive:
            archive.writestr('bank.npz', raw, compress_type=ZIP_LZMA)
        array_save(dino_path(output, category, seed, head), buffer.getvalue())


def score_cache(output, category, fingerprint, paths, file_hashes, seeds, wrn, dino, device):
    old = {seed: DualHead(wrn_path(category, seed).read_bytes(), device) for seed in seeds}
    modern = {(seed, head): CosineBank(dino_path(output, category, seed, head).read_bytes(), device)
              for seed in seeds for head in DINO}
    fields = {'metadata': json.dumps(dict(manifest=fingerprint, seeds=list(seeds), checkpoint=CHECKPOINT_SHA,
                                         bank_hashes=file_hashes))}
    for split in ('calibration', 'test'):
        scores = {(seed, head): [] for seed in seeds for head in HEADS}
        # Single-image forward passes reproduce the deployed encoding.
        for low, high in ordered_prefetch(paths[split], views, workers=3):
            low = embeddings(encode_tensors(low.unsqueeze(0), wrn, device, True))
            high = encode_dino(high.unsqueeze(0), dino, device)
            for seed in seeds:
                a, b = old[seed].raw(low)
                scores[seed, 'fine'].append(a)
                scores[seed, 'coarse'].append(b)
                for head, tokens in zip(DINO, high, strict=True):
                    scores[seed, head].append(modern[seed, head].scores(tokens))
        for (seed, head), rows in scores.items():
            fields[f'{split}_{seed}_{head}'] = torch.cat(rows).cpu().numpy()
    buffer = BytesIO()
    np.savez(buffer, **fields)
    array_save(output / 'raw' / f'{category}.npz', buffer.getvalue())


def materialize(args):
    device = start_device()
    wrn = create_encoder(device, DENSE, True)
    dino = make_dino(device)
    for category in args.categories:
        fingerprint, paths, _ = manifest_parts(args.data, category)
        train(args.output, category, paths['build'], args.seeds, dino, device)
        files = [wrn_path(category, seed) for seed in args.seeds]
        files.extend(dino_path(args.output, category, seed, head) for seed in args.seeds for head in DINO)
        file_hashes = [sha256(path) for path in files]
        expected = dict(manifest=fingerprint, seeds=list(args.seeds), checkpoint=CHECKPOINT_SHA, bank_hashes=file_hashes)
        destination = args.output / 'raw' / f'{category}.npz'
        if destination.is_file():
            with np.load(destination, allow_pickle=False) as archive:
                if json.loads(archive['metadata'].item()) == expected:
                    continue
        score_cache(args.output, category, fingerprint, paths, file_hashes, args.seeds, wrn, dino, device)
        print(f'Cached matching DINOv2 and WRN scores for {category}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=RESULTS)
    parser.add_argument('--categories', nargs='+', choices=CATEGORIES, default=CATEGORIES)
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(5)))
    with torch.inference_mode():
        materialize(parser.parse_args())


if __name__ == '__main__':
    main()
