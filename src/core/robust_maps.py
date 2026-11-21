"""Local anomaly contrast with a ceiling estimated from normal images."""

from torch.nn import functional as F

from balanced_maps import adjust_scores


def contrast_map(scores, grid, mode, window, ceiling):
    if mode == 'standard':
        return adjust_scores(scores, grid, window, 0.75)
    if mode not in ('before', 'after') or ceiling is None or window < 1 or window % 2 == 0:
        raise ValueError('Invalid robust contrast parameters')
    radius = window // 2
    background_input = scores.clamp_max(ceiling) if mode == 'before' else scores
    patches = background_input.reshape(-1, 1, *grid)
    background = F.avg_pool2d(F.pad(patches, (radius,) * 4, mode='replicate'), window, stride=1).flatten(1)
    if mode == 'after':
        background = background.clamp_max(ceiling)
    return scores - 0.75 * background


def image_score(scores, top_k):
    if top_k == 1:
        return scores.amax(1)
    return scores.topk(top_k, dim=1).values.mean(1)
