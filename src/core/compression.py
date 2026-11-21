"""Size-capped PatchCore artifacts and shared exact 1-NN scoring."""

from dataclasses import dataclass
from io import BytesIO

import numpy as np
import torch

from lossless_bank import recover_planes


def residual_norm(centered, projected):
    energy = centered.square().sum(1, keepdim=True) - projected.square().sum(1, keepdim=True)
    return energy.clamp_min(0).sqrt()


def unpack_rows(rows, scale, unsigned=False):
    if rows.dtype == torch.uint8:
        low = rows.bitwise_and(15)
        high = rows.bitwise_right_shift(4)
        rows = torch.stack((low, high), dim=-1).flatten(1)[:, :len(scale)].float()
        if not unsigned:
            rows.sub_(7)
    else:
        rows = rows.float()
    if scale.numel():
        rows.mul_(scale)
    return rows


@dataclass
class Bank:
    bank: np.ndarray
    mean: np.ndarray
    projection: np.ndarray
    scale: np.ndarray
    residual: np.ndarray
    unsigned: bool = False

    def serialize(self):
        with BytesIO() as stream:
            fields = dict(bank=self.bank, mean=self.mean, projection=self.projection,
                          scale=self.scale, residual=self.residual)
            if self.unsigned:
                fields['unsigned'] = np.array(True)
            np.savez(stream, **fields)
            return stream.getvalue()

    @classmethod
    def deserialize(cls, payload):
        with np.load(BytesIO(payload), allow_pickle=False) as arrays:
            return cls.from_fields(arrays)

    @classmethod
    def from_fields(cls, arrays):
        bank = recover_planes(arrays['bank_planes']) if 'bank_planes' in arrays.files else arrays['bank']
        return cls(bank, *(arrays[name] for name in ('mean', 'projection', 'scale', 'residual')),
                   unsigned='unsigned' in arrays.files and bool(arrays['unsigned'].item()))

    def decoded(self, device):
        decoded = unpack_rows(torch.as_tensor(self.bank, device=device), torch.as_tensor(self.scale, device=device),
                              self.unsigned)
        if self.residual.size:
            residual = torch.as_tensor(self.residual, device=device, dtype=torch.float32)
            decoded = torch.cat((decoded, residual), dim=1)
        return decoded

    @property
    def dimension(self):
        return self.scale.size if self.scale.size else self.bank.shape[1]

    def score(self, features):
        device = features.device
        bank = self.decoded(device)
        if self.projection.size:
            mean = torch.as_tensor(self.mean, device=device)
            basis = torch.as_tensor(self.projection, device=device, dtype=torch.float32)
        parts = []
        for query in features.split(4096):
            if self.projection.size:
                centered = query - mean
                query = centered @ basis
                if self.residual.size:
                    query = torch.cat((query, residual_norm(centered, query)), dim=1)
            parts.append(nearest(query, bank))
        return torch.cat(parts)

    def prepare(self, device):
        arrays = [torch.as_tensor(value, device=device) for value in
                  (self.bank, self.mean, self.projection, self.scale, self.residual)]
        return PreparedBank(*arrays, unsigned=self.unsigned)


@dataclass
class PreparedBank:
    bank: torch.Tensor
    mean: torch.Tensor
    projection: torch.Tensor
    scale: torch.Tensor
    residual: torch.Tensor
    unsigned: bool

    def score(self, features):
        query = features
        if self.projection.numel():
            centered = query - self.mean
            query = centered @ self.projection.float()
            if self.residual.numel():
                query = torch.cat((query, residual_norm(centered, query)), dim=1)
        best = torch.full_like(query[:, 0], torch.inf)
        norms = query.square().sum(1)
        for start in range(0, len(self.bank), 512):
            block = unpack_rows(self.bank[start:start + 512], self.scale, self.unsigned)
            if self.residual.numel():
                block = torch.cat((block, self.residual[start:start + 512].float()), dim=1)
            distances = norms[:, None] + block.square().sum(1)
            distances.addmm_(query, block.T, alpha=-2)
            best = torch.minimum(best, distances.amin(1))
        return best.clamp_min_(0).sqrt_()


def nearest(query, bank):
    query_norm = query.square().sum(1)
    best = torch.full_like(query_norm, torch.inf)
    for block in bank.split(4096):
        distances = query_norm[:, None] + block.square().sum(1)
        distances.addmm_(query, block.T, alpha=-2)
        best = torch.minimum(best, distances.amin(1))
    return best.clamp_min_(0).sqrt_()


def fit_bank(pool, order, bits, budget=None, *, mean=None, projection=None, residual=None, unsigned=False):
    """Fit representation on the build pool and count actual uncompressed NPZ bytes."""
    if bits not in (4, 8, 16, 32):
        raise ValueError('bits must be 4, 8, 16, or 32')
    if unsigned and (bits != 4 or pool.amin().item() < 0):
        raise ValueError('Unsigned INT4 requires nonnegative features')
    mean = np.empty(0, dtype=np.float32) if mean is None else mean
    projection = np.empty((0, 0), dtype=np.float32) if projection is None else projection
    encoding = {4: np.uint8, 8: np.int8, 16: np.float16, 32: np.float32}[bits]
    levels = (1 << (bits if unsigned else bits - 1)) - 1
    minimum = 0 if unsigned else -levels
    scale = None
    if bits <= 8:
        chunks = [torch.maximum(block.amax(0), -block.amin(0)) for block in pool.split(512)]
        maximum = torch.stack(chunks).amax(0)
        scale = torch.clamp_min(maximum / levels, torch.finfo(pool.dtype).tiny)
    saved_scale = np.empty(0, dtype=np.float32) if scale is None else scale.cpu().numpy()
    residual_stub = np.empty((1, 1) if residual is not None else (0, 1), dtype=np.float16)
    stored_width = (pool.shape[1] + 1) // 2 if bits == 4 else pool.shape[1]
    probe = Bank(np.empty((1, stored_width), dtype=encoding), mean, projection, saved_scale, residual_stub, unsigned)
    row_bytes = probe.bank.nbytes + residual_stub.nbytes
    count = len(order)
    if budget is not None:
        overhead = len(probe.serialize()) - row_bytes
        count = min(count, (budget - overhead) // row_bytes)
    while count > 0:
        ids = torch.as_tensor(order[:count], device=pool.device)
        rows = pool[ids].cpu().numpy()
        if scale is not None:
            rows = np.clip(np.rint(rows / saved_scale), minimum, levels).astype(np.int8)
        if bits == 4:
            rows = (rows - minimum).astype(np.uint8)
            if rows.shape[1] % 2:
                rows = np.pad(rows, ((0, 0), (0, 1)))
            rows = rows[:, ::2] | (rows[:, 1::2] << 4)
        stored_residual = residual_stub if residual is None else residual[ids].cpu().numpy().astype(np.float16)
        model = Bank(rows.astype(encoding, copy=False), mean, projection, saved_scale,
                     stored_residual, unsigned)
        if budget is None or len(model.serialize()) <= budget:
            return model
        count -= 1
    raise ValueError('Budget cannot fit one patch and model metadata')
