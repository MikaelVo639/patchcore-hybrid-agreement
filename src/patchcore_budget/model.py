"""Portable detector with a strict budget on the entire serialized model.npz."""

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .numerics import bank_order, matrix, nearest


@dataclass
class BudgetMemory:
    bank: np.ndarray
    mean: np.ndarray
    projection: np.ndarray

    def __post_init__(self):
        self.bank = matrix(self.bank, "bank")
        self.mean = np.asarray(self.mean, dtype=np.float32)
        self.projection = np.asarray(self.projection, dtype=np.float32)
        if self.mean.ndim != 1 or self.projection.ndim != 2:
            raise ValueError("Invalid transform shapes")
        if self.projection.size:
            if self.projection.shape != (len(self.mean), self.bank.shape[1]):
                raise ValueError("Projection and bank shapes disagree")
        elif self.mean.size or self.projection.shape != (0, 0):
            raise ValueError("Identity model must have empty mean and projection")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.projection).all():
            raise ValueError("Transform contains nonfinite values")

    @property
    def dimension(self):
        return self.bank.shape[1]

    @property
    def input_dimension(self):
        return len(self.mean) if self.projection.size else self.dimension

    def transform(self, query):
        query = matrix(query, "query")
        if query.shape[1] != self.input_dimension:
            raise ValueError("Query feature dimension does not match the model")
        return (query - self.mean) @ self.projection if self.projection.size else query

    def score(self, query):
        """Patch scores are Euclidean 1-NN distances; higher means more anomalous."""
        return np.sqrt(nearest(self.transform(query), self.bank)[0])

    def to_bytes(self):
        stream = io.BytesIO()
        np.savez(
            stream,
            bank=self.bank,
            mean=self.mean,
            projection=self.projection,
            format_version=np.array([1], dtype=np.int64),
        )
        return stream.getvalue()

    def storage(self):
        parts = {
            "bank_bytes": self.bank.nbytes,
            "mean_bytes": self.mean.nbytes,
            "projection_bytes": self.projection.nbytes,
        }
        parts["artifact_bytes"] = len(self.to_bytes())
        # The version array and NPZ headers are included in container_bytes.
        parts["container_bytes"] = parts["artifact_bytes"] - (
            parts["bank_bytes"] + parts["mean_bytes"] + parts["projection_bytes"]
        )
        return parts

    def save(self, path, budget_bytes=None):
        payload = self.to_bytes()
        if budget_bytes is not None and len(payload) > budget_bytes:
            raise ValueError("Serialized model exceeds its storage budget")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return len(payload)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != {"bank", "mean", "projection", "format_version"}:
                raise ValueError("Unexpected model archive fields")
            if saved["format_version"].tolist() != [1]:
                raise ValueError("Unsupported model format version")
            return cls(saved["bank"], saved["mean"], saved["projection"])


def max_bank_count(dimension, mean, forward, budget_bytes, available=None):
    """Largest bank size N whose serialized artifact fits B.

    The NPZ size depends only on array shapes and dtypes, so a zero-valued probe gives
    the same byte count as real features. With ``available=None`` the result is the
    count allowed by the budget alone, used to report whether B or the pool is binding.
    """
    if isinstance(budget_bytes, bool) or not isinstance(budget_bytes, int) or budget_bytes < 1:
        raise ValueError("Budget must be a positive integer number of bytes")
    if type(dimension) is not int or dimension < 1:
        raise ValueError("Bank dimension must be a positive integer")
    if available is not None and available < 1:
        raise ValueError("At least one pool patch is required")
    row_bytes = dimension * 4

    def probe(count):
        return len(BudgetMemory(np.zeros((count, dimension), np.float32), mean, forward).to_bytes())

    overhead = probe(1) - row_bytes
    count = (budget_bytes - overhead) // row_bytes
    if available is not None:
        count = min(available, count)
    if count < 1:
        raise ValueError("Budget cannot fit transform, archive headers and one patch")
    # Header alignment can change with the number of digits in the bank shape.
    size = probe(count)
    while count > 0 and size > budget_bytes:
        count -= max(1, (size - budget_bytes + row_bytes - 1) // row_bytes)
        if count:
            size = probe(count)
    if count < 1:
        raise ValueError("No serialized bank fits the budget")
    return int(count)


def fit_budget_bank(
    projected_pool, mean, forward, budget_bytes, seed=0, sampler="coreset", order=None
):
    """Fit the largest feasible bank, charging transform and archive overhead.

    ``order`` may hold a precomputed sampling order of at least the required length.
    Farthest-first and seeded random orders are prefix-consistent, so one order computed
    for the largest budget gives the same bank as recomputing it for each smaller budget.
    """
    pool = matrix(projected_pool)
    count = max_bank_count(pool.shape[1], mean, forward, budget_bytes, len(pool))
    if order is None:
        ids = bank_order(pool, count, seed, sampler)
    else:
        if len(order) < count:
            raise ValueError("Precomputed bank order is shorter than the feasible bank")
        ids = np.asarray(order[:count], dtype=np.int64)
    model = BudgetMemory(pool[ids], mean, forward)
    if len(model.to_bytes()) > budget_bytes:
        raise RuntimeError("Internal storage accounting mismatch")
    return model
