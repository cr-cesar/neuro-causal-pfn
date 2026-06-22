"""E11 equity audit: stratified treatment-effect error.

The same root-PEHE that Tier 4 reports as a single number is broken down by
subgroup (vascular territory, lesion-volume quartile, age band, sex). The
concern is whether accuracy degrades for any subpopulation; the plan's criterion
is that the worst-to-best ratio stays below 2x.
"""
import numpy as np

from .metrics import root_pehe


def stratified_pehe(cate_pred, cate_true, groups, max_ratio: float = 2.0) -> dict:
    """Root-PEHE overall and within each subgroup defined by groups.

    Returns a dict with the per-group root-PEHE, the overall value under 'all',
    the worst-to-best ratio under 'max_min_ratio', and a boolean 'passes' that is
    True when the ratio is below max_ratio.
    """
    cate_pred = np.asarray(cate_pred).ravel()
    cate_true = np.asarray(cate_true).ravel()
    groups = np.asarray(groups)
    out = {"all": root_pehe(cate_pred, cate_true)}
    per_group = {}
    for g in np.unique(groups):
        m = groups == g
        if m.sum() >= 1:
            per_group[str(g)] = root_pehe(cate_pred[m], cate_true[m])
    out.update(per_group)
    if per_group:
        values = np.array(list(per_group.values()), dtype=np.float64)
        lo = float(values.min())
        ratio = float(values.max() / lo) if lo > 0 else float("inf")
        out["max_min_ratio"] = ratio
        out["passes"] = bool(ratio < max_ratio)
    return out
