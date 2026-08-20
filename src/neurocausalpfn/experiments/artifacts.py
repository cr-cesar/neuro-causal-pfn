"""Extraction of tier inputs from a trained Stage-1 encoder.

Given a trained :class:`ConvVAE3D` and its dataset, this module produces exactly
the quantities the tiered evaluation needs, and nothing more:

- Tier 1 : the held-out Dice and BCE of the reconstruction (lesion channel).
- Tier 2 : the frozen latent code Z (posterior mean mu), plus the clinical
           covariates and a clinical target for the probe.
- Tier 3 : mu and logvar over the cohort (for per-dimension KL and IOSS), and
           the ARD prior variance when the encoder has one.
- strata : lesion volume, age and sex, for the equity breakdown of Tiers 2 and 4.

The extraction rebuilds the dataset with the same seed the training used, so it
matches the exact cohort the encoder saw. It does not retrain anything.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

# NOTE: torch is imported lazily inside the functions that need it, so the
# numpy-only parts of the tier harness (registry, gates, estimators, report) can
# be imported and tested in environments without a torch install.


@dataclass
class VaeArtifacts:
    Z: np.ndarray                    # [N, zdim] posterior mean (the exported code)
    logvar: Optional[np.ndarray] = None        # [N, zdim] (None for non-VAE arms)
    dice: Optional[float] = None     # Tier-1 reconstruction overlap (lesion)
    bce: Optional[float] = None
    clinical: Optional[np.ndarray] = None      # [N, c] age, sex, indicators
    volume: Optional[np.ndarray] = None        # [N] lesion foreground fraction
    prior_var: Optional[np.ndarray] = None     # ARD prior variance, if any
    has_posterior: bool = True       # False for contrastive (C) / MAE (D): no KL
    meta: Dict = field(default_factory=dict)

    @property
    def per_dim_kl(self) -> Optional[np.ndarray]:
        if not self.has_posterior or self.logvar is None:
            return None
        from ..eval.latent_quality import per_dim_kl as _pdk
        return _pdk(self.Z, self.logvar, self.prior_var)


def latent_artifacts(Z: np.ndarray, volume: Optional[np.ndarray] = None,
                     clinical: Optional[np.ndarray] = None,
                     has_posterior: bool = False, meta: Optional[Dict] = None
                     ) -> "VaeArtifacts":
    """Wrap an exported latent code (e.g. from a contrastive or masked encoder,
    which have no posterior) as artifacts for the tier harness. Tier-3 KL-based
    diagnostics are skipped; IOSS and Tier-4 still apply."""
    Z = np.asarray(Z, dtype=np.float64)
    if volume is None:
        volume = np.zeros(len(Z))
    return VaeArtifacts(Z=Z, logvar=None, has_posterior=has_posterior,
                        volume=volume, clinical=clinical,
                        meta=meta or {"zdim": int(Z.shape[1]), "n": int(len(Z))})


def _binary_dice(logits, target) -> float:
    """Hard Dice at threshold 0.5, the reported Tier-1 overlap."""
    import torch
    prob = torch.sigmoid(logits)
    pred = (prob > 0.5).float()
    inter = (pred * target).sum()
    denom = pred.sum() + target.sum()
    if float(denom) == 0.0:
        return 1.0
    return float((2.0 * inter / denom).clamp(0.0, 1.0))


def vae_artifacts(model, dataset, device: str = "cpu", batch_size: int = 8,
                  representation: str = "lesion", use_daft: bool = False,
                  compute_recon: bool = True) -> VaeArtifacts:
    """Encode the whole cohort once and, for a binary lesion channel, measure the
    reconstruction Dice/BCE. ``model`` is a trained ConvVAE3D."""
    import torch
    from torch.utils.data import DataLoader

    from ..utils.runtime import resolve_device

    device = resolve_device({"device": device})   # accepts 'auto' from full-mode configs
    model = model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    lesion_channel = representation in ("lesion", "early_fusion")

    mus, lvs, vols = [], [], []
    dices, bces, n_seen = [], [], 0
    bce = torch.nn.BCEWithLogitsLoss()
    # Scoped no_grad: must NOT use the global torch.set_grad_enabled here, or it
    # would leak the disabled-grad state into any code that runs afterwards.
    with torch.no_grad():
        for batch in loader:
            items = list(batch) if isinstance(batch, (list, tuple)) else [batch]
            x = items[0].to(device)
            clin = items[1].to(device) if use_daft and len(items) > 1 else None
            mu, logvar = model.enc(x, clin) if use_daft else model.enc(x)
            mus.append(mu.cpu().numpy())
            lvs.append(logvar.cpu().numpy())
            # lesion foreground fraction as a cheap volume proxy / stratifier
            ch0 = x[:, 0:1]
            vols.append(ch0.flatten(1).mean(1).cpu().numpy())
            if compute_recon and lesion_channel:
                logits = model.dec(mu)
                l0 = logits[:, 0:1]
                t0 = (ch0 > 0.5).float()
                dices.append(_binary_dice(l0, t0))
                bces.append(float(bce(l0, t0)))
            n_seen += x.shape[0]

    Z = np.concatenate(mus, axis=0)
    logvar = np.concatenate(lvs, axis=0)
    volume = np.concatenate(vols, axis=0)
    prior_var = None
    if getattr(model, "use_ard", False) and hasattr(model, "ard_prior_var"):
        prior_var = model.ard_prior_var.detach().cpu().numpy()

    clinical = None
    if hasattr(dataset, "clinical_matrix"):
        try:
            clinical = np.asarray(dataset.clinical_matrix(), dtype=np.float64)[: len(Z)]
        except Exception:
            clinical = None

    dice = float(np.mean(dices)) if dices else None
    bce_v = float(np.mean(bces)) if bces else None
    return VaeArtifacts(Z=Z, logvar=logvar, dice=dice, bce=bce_v,
                        clinical=clinical, volume=volume, prior_var=prior_var,
                        meta={"n": int(n_seen), "zdim": int(Z.shape[1]),
                              "representation": representation})


def clinical_target(artifacts: VaeArtifacts, kind: str = "nihss",
                    seed: int = 0) -> np.ndarray:
    """A clinical score to probe against in Tier 2.

    When a real NIHSS/mRS column is available it should be passed directly. In
    the synthetic / real-mask-without-scores setting (section 22 note 2: NIHSS is
    absent from the Giles cohort) this builds a *non-circular* surrogate deficit
    from quantities the encoder never receives as a label: the lesion volume and,
    when present, age. Larger lesions and older patients carry more deficit, plus
    Gaussian noise. This exercises the probe and its gate honestly; it is not a
    substitute for a real clinical outcome.
    """
    rng = np.random.default_rng(seed)
    vol = artifacts.volume
    z = (vol - vol.mean()) / (vol.std() + 1e-8)
    score = 0.8 * z
    if artifacts.clinical is not None and artifacts.clinical.shape[1] >= 1:
        age = artifacts.clinical[:, 0]
        az = (age - age.mean()) / (age.std() + 1e-8)
        score = score + 0.2 * az
    score = score + 0.25 * rng.standard_normal(len(vol))
    return score.astype(np.float64)


def volume_quartiles(volume: np.ndarray) -> np.ndarray:
    """Quartile label (0..3) per sample, for the equity strata."""
    v = np.asarray(volume, dtype=np.float64)
    if v.size == 0:
        return v.astype(int)
    edges = np.quantile(v, [0.25, 0.5, 0.75])
    return np.digitize(v, edges).astype(int)
