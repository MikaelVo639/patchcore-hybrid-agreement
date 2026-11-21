"""Post-processing for PatchCore anomaly maps."""

from torch.nn import functional as F


def adjust_scores(scores, grid, window, strength):
    if window < 1 or window % 2 == 0:
        raise ValueError('Window must be positive and odd')
    if strength == 0:
        return scores
    radius = window // 2
    maps = scores.reshape(-1, 1, *grid)
    background = F.avg_pool2d(F.pad(maps, (radius,) * 4, mode='replicate'), window, stride=1)
    return scores - strength * background.flatten(1)
