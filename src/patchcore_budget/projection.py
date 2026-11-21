"""Build-only PCA. Whitening is fused into the deployed projection."""

from dataclasses import dataclass

import numpy as np

from .numerics import matrix, sample_rows


@dataclass
class PCAFit:
    mean: np.ndarray
    components: np.ndarray
    eigenvalues: np.ndarray
    total_variance: float
    fitted_rows: int

    @classmethod
    def fit(cls, build, max_rows=20000, seed=0):
        x = sample_rows(matrix(build), max_rows, seed).astype(np.float64)
        if len(x) < 2:
            raise ValueError("PCA needs at least two build patches")
        mean = x.mean(0)
        centered = x - mean
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
        usable = min(len(x) - 1, x.shape[1])
        eig = singular[:usable] ** 2 / (len(x) - 1)
        return cls(
            mean.astype(np.float32),
            vt[:usable].astype(np.float32),
            eig,
            float(np.sum(centered * centered) / (len(x) - 1)),
            len(x),
        )

    def transform_for(self, dim, whiten=False, regularization=0.01):
        if not 1 <= dim <= len(self.components):
            raise ValueError("PCA dimension exceeds the fitted dimensional limit")
        if regularization <= 0:
            raise ValueError("Whitening regularization must be positive")
        gain = np.ones(dim, dtype=np.float64)
        if whiten:
            scale = max(float(self.eigenvalues[:dim].mean()), 1e-12)
            gain = 1 / np.sqrt(self.eigenvalues[:dim] + regularization * scale)
        forward = (self.components[:dim].T * gain).astype(np.float32)
        inverse = (self.components[:dim] / gain[:, None]).astype(np.float32)
        return forward, inverse

    def variance_dimension(self, threshold):
        if not 0 < threshold <= 1:
            raise ValueError("Variance threshold must be in (0, 1]")
        if self.total_variance <= 1e-12:
            return 1
        cumulative = np.cumsum(self.eigenvalues) / self.total_variance
        return min(int(np.searchsorted(cumulative, threshold)) + 1, len(cumulative))
