"""Shared bilinear resize and optional separable Gaussian smoothing."""

import numpy as np


def resize_map(values, shape):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) == 0 or min(shape) < 1:
        raise ValueError("Map and output shape must be nonempty 2D")
    # Align corners: documented here so every method uses exactly the same mapping.
    yy = np.linspace(0, values.shape[0] - 1, shape[0])
    xx = np.linspace(0, values.shape[1] - 1, shape[1])
    y0, x0 = np.floor(yy).astype(int), np.floor(xx).astype(int)
    y1, x1 = np.minimum(y0 + 1, values.shape[0] - 1), np.minimum(x0 + 1, values.shape[1] - 1)
    wy, wx = (yy - y0)[:, None], (xx - x0)[None, :]
    top = values[y0[:, None], x0] * (1 - wx) + values[y0[:, None], x1] * wx
    bottom = values[y1[:, None], x0] * (1 - wx) + values[y1[:, None], x1] * wx
    return top * (1 - wy) + bottom * wy


def smooth_map(values, sigma):
    if sigma == 0:
        return values
    radius = max(1, int(np.ceil(3 * sigma)))
    axis = np.arange(-radius, radius + 1)
    kernel = np.exp(-(axis**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    result = np.asarray(values, dtype=np.float64)
    for dim in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[dim] = (radius, radius)
        padded = np.pad(result, pad, mode="edge")
        result = np.apply_along_axis(
            lambda row: np.convolve(row, kernel, mode="valid"), dim, padded
        )
    return result
