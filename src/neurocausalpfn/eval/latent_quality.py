"""Tier 3 evaluation: latent-space quality.

These operate on encoded representations (and, where noted, on ground-truth
latent factors), not on the model itself, so they can be run on exported latents.

- active_dimensions / per_dim_kl: how many latent axes the model actually uses
  (the ARD effective dimensionality, E4).
- ioss: a relative disentanglement diagnostic (independence of support, after
  Wang and Jordan 2024), estimated by the Hausdorff distance between the joint
  support and an independent recombination of the coordinates.
- mcc: mean correlation coefficient between learned latents and ground-truth
  factors (identifiability check), matched by the optimal assignment.
"""
import numpy as np


def per_dim_kl(mu, logvar, prior_var=None) -> np.ndarray:
    """Per-dimension KL of N(mu, exp(logvar)) against the prior, averaged over the
    sample axis. prior_var (per dimension) gives the ARD prior; None gives N(0, I).
    Returns a vector of length zdim."""
    mu = np.asarray(mu, dtype=np.float64)
    logvar = np.asarray(logvar, dtype=np.float64)
    var = np.exp(logvar)
    if prior_var is None:
        per = -0.5 * (1.0 + logvar - mu ** 2 - var)
    else:
        pv = np.asarray(prior_var, dtype=np.float64)
        per = -0.5 * (1.0 + logvar - np.log(pv) - (mu ** 2 + var) / pv)
    return per.mean(axis=0)


def active_dimensions(per_dim_kl_vector, threshold: float = 0.01) -> int:
    """Number of latent dimensions whose mean KL exceeds the threshold."""
    return int((np.asarray(per_dim_kl_vector) > threshold).sum())


def _standardize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return (X - X.mean(axis=0)) / sd


def ioss(Z, seed: int = 0) -> float:
    """Independence of support score (relative disentanglement diagnostic).

    Standardizes Z, builds an independent recombination by permuting each
    coordinate separately, and returns the symmetric Hausdorff distance between
    the two point sets. Values near 0 indicate near-independent support; larger
    values indicate stronger dependence between latent coordinates."""
    from scipy.spatial.distance import directed_hausdorff

    Zs = _standardize(Z)
    rng = np.random.default_rng(seed)
    Z_ind = np.column_stack([Zs[rng.permutation(Zs.shape[0]), j] for j in range(Zs.shape[1])])
    d1 = directed_hausdorff(Zs, Z_ind)[0]
    d2 = directed_hausdorff(Z_ind, Zs)[0]
    return float(max(d1, d2))


def mcc(Z, V) -> float:
    """Mean correlation coefficient between learned latents Z and ground-truth
    factors V, matched by the optimal (Hungarian) assignment. Returns a value in
    [0, 1]; 1 means each true factor is captured by a distinct latent dimension."""
    from scipy.optimize import linear_sum_assignment

    Zs, Vs = _standardize(Z), _standardize(V)
    n = Zs.shape[0]
    corr = np.abs(Zs.T @ Vs) / n          # [d_z, d_v]
    row, col = linear_sum_assignment(-corr)
    return float(corr[row, col].mean())
