"""Validated, blockwise numerical building blocks; no full query-by-bank matrix."""

import numpy as np


def matrix(value, name="features"):
    x = np.asarray(value, dtype=np.float32)
    if x.ndim != 2 or min(x.shape) == 0 or not np.isfinite(x).all():
        raise ValueError(f"{name} must be a nonempty finite 2D array")
    return np.ascontiguousarray(x)


def nearest(query, bank, query_block=256, bank_block=1024):
    """Return squared Euclidean distance and index, with deterministic tie breaking."""
    q, b = matrix(query, "query"), matrix(bank, "bank")
    if q.shape[1] != b.shape[1] or query_block < 1 or bank_block < 1:
        raise ValueError("Incompatible dimensions or nonpositive block size")
    distances = np.empty(len(q), dtype=np.float64)
    indices = np.empty(len(q), dtype=np.int64)
    for start in range(0, len(q), query_block):
        # Float64 avoids cancellation for nearby, high-magnitude descriptors.
        a = q[start : start + query_block].astype(np.float64)
        best = np.full(len(a), np.inf)
        ix = np.zeros(len(a), dtype=np.int64)
        for offset in range(0, len(b), bank_block):
            z = b[offset : offset + bank_block].astype(np.float64)
            d = (a * a).sum(1)[:, None] + (z * z).sum(1)[None, :] - 2 * a @ z.T
            np.maximum(d, 0, out=d)
            local = d.argmin(axis=1)
            val = d[np.arange(len(a)), local]
            update = val < best
            best[update], ix[update] = val[update], local[update] + offset
        distances[start : start + len(a)] = best
        indices[start : start + len(a)] = ix
    return distances, indices


def sample_rows(x, limit, seed):
    if limit < 1:
        raise ValueError("Sampling limit must be positive")
    if len(x) <= limit:
        return x
    ids = np.sort(np.random.default_rng(seed).choice(len(x), limit, replace=False))
    return x[ids]


def bank_order(x, count, seed=0, method="coreset", max_float64_bytes=1 << 29):
    """Seeded farthest-first traversal, or seeded random sampling, without duplicates."""
    x = matrix(x)
    if not 1 <= count <= len(x):
        raise ValueError("Bank count must be in [1, number of available patches]")
    rng = np.random.default_rng(seed)
    if method == "random":
        return rng.permutation(len(x))[:count]
    if method != "coreset":
        raise ValueError("Sampler must be 'coreset' or 'random'")
    result = np.empty(count, dtype=np.int64)
    available = np.ones(len(x), dtype=bool)
    best = np.full(len(x), np.inf)
    chosen = int(rng.integers(len(x)))
    # Small pools are converted to float64 once instead of once per selected point.
    # The arithmetic per row is unchanged, so the order is identical either way.
    wide = x.astype(np.float64) if x.size * 8 <= max_float64_bytes else None
    # About 16 MB of float64 per block: low dimensions use far fewer Python iterations.
    block = max(4096, (1 << 21) // x.shape[1])
    buffer = np.empty((min(block, len(x)), x.shape[1]), dtype=np.float64)
    dist = np.empty(min(block, len(x)), dtype=np.float64)
    for i in range(count):
        result[i] = chosen
        available[chosen] = False
        center = x[chosen].astype(np.float64)
        # One point at a time, bounded temporary memory even for a large patch pool.
        for start in range(0, len(x), block):
            stop = min(start + block, len(x))
            delta, part = buffer[: stop - start], dist[: stop - start]
            source = wide[start:stop] if wide is not None else x[start:stop]
            np.subtract(source, center, out=delta)
            np.einsum("ij,ij->i", delta, delta, out=part)
            np.minimum(best[start:stop], part, out=best[start:stop])
        best[~available] = -np.inf
        chosen = int(best.argmax())
    return result
