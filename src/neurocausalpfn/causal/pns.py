"""Probability of Necessity and Sufficiency (PNS) for Arm B.

Following Wang and Jordan (2024), the PNS of a representation measures how
necessary and sufficient each latent dimension is for the outcome, after
deconfounding by the common cause C (here estimated with a factor model on the
latents, approximately the dominant vascular variation). Under monotonicity and
exogeneity given C, the PNS lower bound for a dimension reduces to the
deconfounded do-contrast between a high and a low value of that dimension.

Two estimators are provided:

- pns_lower_bound: the exact, non-differentiable estimator (logistic contrast
  conditional on C). This is the offline CRL metric used in Tier 3 and as an
  offline model-selection criterion.
- soft_pns_value / soft_pns_per_dim: a differentiable surrogate (deconfounded
  correlation passed through a ReLU) usable as the training-time auxiliary loss,
  since the exact estimator is not differentiable through the encoder.
"""
import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Exact offline estimator (numpy / scikit-learn)
# --------------------------------------------------------------------------- #
def _standardize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return (X - X.mean(axis=0)) / sd


def estimate_common_cause(Z, k: int = 5) -> np.ndarray:
    """Factor model estimate of the common cause C from the latents (k factors)."""
    from sklearn.decomposition import FactorAnalysis

    Z = _standardize(Z)
    k = max(1, min(int(k), Z.shape[1] - 1, Z.shape[0] - 1))
    return FactorAnalysis(n_components=k, random_state=0).fit_transform(Z)


def _binarize(Y: np.ndarray) -> np.ndarray:
    Y = np.asarray(Y, dtype=np.float64).ravel()
    if np.unique(Y).size > 2:
        Y = (Y > np.median(Y)).astype(np.float64)
    return Y


def pns_lower_bound(Z, Y, k: int = 5, C=None, delta: float = 1.0) -> np.ndarray:
    """Per-dimension PNS lower bound, conditional on the common cause C.

    For each standardized latent dimension a logistic model P(Y=1 | z_l, C) is
    fitted, and the contrast P(Y=1 | z_l=+delta, C) - P(Y=1 | z_l=-delta, C),
    averaged over the observed C, gives the deconfounded do-contrast. The PNS
    lower bound is the non-negative part of that contrast. Returns a vector of
    length zdim; values near 0 mean the dimension carries little causal signal.
    """
    from sklearn.linear_model import LogisticRegression

    Zs = _standardize(Z)
    Y = _binarize(Y)
    n, d = Zs.shape
    if np.unique(Y).size < 2:
        return np.zeros(d)
    if C is None:
        C = estimate_common_cause(Zs, k)
    C = np.asarray(C, dtype=np.float64)
    pns = np.zeros(d)
    for l in range(d):
        feat = np.column_stack([Zs[:, l], C])
        clf = LogisticRegression(max_iter=300).fit(feat, Y)
        hi = clf.predict_proba(np.column_stack([np.full(n, delta), C]))[:, 1]
        lo = clf.predict_proba(np.column_stack([np.full(n, -delta), C]))[:, 1]
        pns[l] = max(0.0, float((hi - lo).mean()))
    return pns


# --------------------------------------------------------------------------- #
# Differentiable soft surrogate (torch) for the training-time auxiliary loss
# --------------------------------------------------------------------------- #
def _topk_components(z: torch.Tensor, k: int) -> torch.Tensor:
    """Detached top-k principal component scores of z (the common cause C)."""
    zc = z - z.mean(0, keepdim=True)
    k = max(1, min(int(k), zc.shape[1] - 1, zc.shape[0] - 1))
    _, _, vh = torch.linalg.svd(zc, full_matrices=False)
    return (zc @ vh[:k].T).detach()


def soft_pns_per_dim(mu: torch.Tensor, y: torch.Tensor, k: int = 5, eps: float = 1e-6) -> torch.Tensor:
    """Differentiable per-dimension PNS surrogate: the ReLU of the deconfounded
    correlation between each latent dimension and the outcome. Differentiable with
    respect to the latents (hence the encoder)."""
    z = mu
    zc = z - z.mean(0, keepdim=True)
    y = y.float().view(-1)
    yc = y - y.mean()
    if float(yc.std()) < eps or zc.shape[1] == 0:
        return torch.zeros(zc.shape[1], device=z.device)
    if zc.shape[1] == 1 or zc.shape[0] <= 2:
        num = (zc * yc.unsqueeze(1)).mean(0)
        den = zc.std(0) * yc.std() + eps
        return torch.relu(num / den)
    C = _topk_components(z, k)                                  # [N, k], detached
    gram = C.T @ C + eps * torch.eye(C.shape[1], device=z.device)
    proj_z = C @ torch.linalg.solve(gram, C.T @ zc)            # projection of zc onto C
    z_res = zc - proj_z                                        # deconfounded latents
    proj_y = C @ torch.linalg.solve(gram, C.T @ yc.unsqueeze(1))
    y_res = (yc.unsqueeze(1) - proj_y).squeeze(1).detach()     # deconfounded outcome
    num = (z_res * y_res.unsqueeze(1)).mean(0)
    den = z_res.std(0) * y_res.std() + eps
    return torch.relu(num / den)


def soft_pns_value(mu: torch.Tensor, y: torch.Tensor, k: int = 5) -> torch.Tensor:
    """Scalar PNS surrogate (mean over dimensions). Larger is more causally
    informative; the training objective maximises it via -lambda * value."""
    return soft_pns_per_dim(mu, y, k).mean()
