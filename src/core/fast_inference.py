"""Exact cached 1-NN and bounded preprocessing for the layer2-contrast detector."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from itertools import islice

import torch

from compression import unpack_rows
from score_tuning import local_contrast, make_maps


class CacheMode(str, Enum):
    NORMS = 'norms'
    DECODED = 'decoded'
    TILED = 'tiled'


@dataclass
class CachedDetector:
    bank: torch.Tensor
    scale: torch.Tensor
    norms: tuple[torch.Tensor, ...]
    unsigned: bool
    mode: CacheMode

    @classmethod
    def create(cls, detector, device, mode):
        if detector.method != 'layer2-contrast' or detector.bank.dimension != 512:
            raise ValueError('Expected the 512-dimensional layer2-contrast bank')
        mode = CacheMode(mode)
        packed, _, _ = detector.prepare(device)
        if mode == CacheMode.DECODED:
            bank = unpack_rows(packed.bank, packed.scale, packed.unsigned)
            norms = tuple(block.square().sum(1) for block in bank.split(512))
        else:
            bank = packed.bank
            norms = tuple(unpack_rows(block, packed.scale, packed.unsigned).square().sum(1)
                          for block in bank.split(512))
        return cls(bank, packed.scale, norms, packed.unsigned, mode)

    def scores(self, features):
        parts = []
        # Keep the original bank block boundaries to preserve arithmetic.
        query_rows = 392 if self.mode == CacheMode.TILED else 4096
        for query in features[:, :512].split(query_rows):
            query = query.contiguous()
            norms = query.square().sum(1, keepdim=True)
            best = torch.full_like(norms, torch.inf)
            for block, bank_norms in zip(self.bank.split(512), self.norms, strict=True):
                if self.mode != CacheMode.DECODED:
                    block = unpack_rows(block, self.scale, self.unsigned)
                distances = norms + bank_norms
                distances.addmm_(query, block.T, alpha=-2)
                best = torch.minimum(best, distances.amin(1, keepdim=True))
            parts.append(best.clamp_min_(0).sqrt_().flatten())
        return torch.cat(parts)

    def infer(self, features, grid=(28, 28), size=(224, 224)):
        raw = self.scores(features).reshape(-1, grid[0] * grid[1])
        return raw.amax(1), make_maps(local_contrast(raw), grid, size, 4)


def ordered_prefetch(items, prepare, workers=2):
    """Yield in order, with at most one pending input per worker."""
    if workers < 1:
        raise ValueError('workers must be positive')
    items = iter(items)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = deque(executor.submit(prepare, item) for item in islice(items, workers))
        while pending:
            result = pending.popleft().result()
            for item in islice(items, 1):
                pending.append(executor.submit(prepare, item))
            yield result
