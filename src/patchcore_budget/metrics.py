"""Exact tied-score AUROC and connected-region AUPRO on evaluated maps."""

from collections import deque

import numpy as np


def _binary(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape or labels.size == 0:
        raise ValueError("Labels and scores must be nonempty with matching shapes")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Labels must be binary and scores finite")
    return labels.astype(bool).ravel(), scores.ravel()


def _curve(labels, scores, weights):
    order = np.argsort(-scores, kind="stable")
    y, s, w = labels[order], scores[order], weights[order]
    ends = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    fpr = np.r_[0.0, np.cumsum(~y)[ends] / (~y).sum()]
    recall = np.r_[0.0, np.cumsum(w)[ends]]
    return fpr, recall


def binary_auroc(labels, scores):
    y, s = _binary(labels, scores)
    if y.all() or not y.any():
        return None
    fpr, tpr = _curve(y, s, y.astype(float) / y.sum())
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))


def _regions(mask):
    """8-connected foreground regions; each image is labeled separately."""
    seen = np.zeros_like(mask, dtype=bool)
    for y, x in zip(*np.nonzero(mask), strict=True):
        if seen[y, x]:
            continue
        queue, region = deque([(int(y), int(x))]), []
        seen[y, x] = True
        while queue:
            cy, cx = queue.popleft()
            region.append((cy, cx))
            for ny in range(max(0, cy - 1), min(mask.shape[0], cy + 2)):
                for nx in range(max(0, cx - 1), min(mask.shape[1], cx + 2)):
                    if mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        queue.append((ny, nx))
        yield region


def aupro(masks, maps, max_fpr=0.3):
    y, s = _binary(masks, maps)
    masks = np.asarray(masks, dtype=bool)
    if masks.ndim != 3 or not 0 < max_fpr <= 1:
        raise ValueError("AUPRO needs [images, height, width] masks and max_fpr in (0,1]")
    weights = np.zeros(masks.shape, dtype=np.float64)
    count = 0
    for i, mask in enumerate(masks):
        for region in _regions(mask):
            yy, xx = np.array(region).T
            weights[i, yy, xx] = 1 / len(region)
            count += 1
    if count == 0 or y.all():
        return None
    fpr, pro = _curve(y, s, weights.ravel() / count)
    # Preserve vertical segments at both FPR=0 and the integration limit.
    # Dropping points exactly at the limit lets interpolation replace a vertical
    # rise there with a diagonal from the previous FPR, adding spurious area.
    before = fpr <= max_fpr
    boundary = float(np.interp(max_fpr, fpr, pro))
    x = np.r_[fpr[before], max_fpr]
    v = np.r_[pro[before], boundary]
    return float(np.sum(np.diff(x) * (v[1:] + v[:-1]) / 2) / max_fpr)


def evaluate(labels, image_scores, masks, maps):
    return {
        "image_auroc": binary_auroc(labels, image_scores),
        "pixel_auroc": binary_auroc(masks, maps),
        "aupro_0_3": aupro(masks, maps, 0.3),
        "aupro_connectivity": 8,
    }
