"""Seeded low-rank normal features with local synthetic defects; never benchmark evidence."""

from pathlib import Path

import numpy as np

from .cache import FeatureCache


def make_demo(directory, seed=0):
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Demo cache directory must be empty; choose a new path")
    rng = np.random.default_rng(seed)
    p, h, w = 24, 4, 4
    basis, _ = np.linalg.qr(rng.normal(size=(p, 5)))
    metadata = {
        "dataset": "synthetic",
        "category": "low_rank_demo",
        "synthetic": True,
        "feature_signature": f"synthetic-low-rank-v1-seed-{seed}",
        "note": "Software smoke test; no evidence of real anomaly-detection performance",
    }
    for split, count in (("build", 24), ("calibration", 8), ("test", 16)):
        features = (
            rng.normal(size=(count, h, w, 5)) @ basis.T + 0.05 * rng.normal(size=(count, h, w, p))
        ).astype(np.float32)
        labels, masks = None, None
        if split == "test":
            labels = np.zeros(count, dtype=np.uint8)
            labels[count // 2 :] = 1
            masks = np.zeros((count, h, w), dtype=np.uint8)
            for i in range(count // 2, count):
                y, x = rng.integers(h), rng.integers(w)
                masks[i, y, x] = 1
                features[i, y, x] += rng.normal(0, 3, p).astype(np.float32)
        FeatureCache(
            features,
            np.array([f"synthetic/{split}/{i:03d}" for i in range(count)]),
            split,
            metadata,
            labels,
            masks,
        ).save(directory / f"{split}.npz")
    return directory
