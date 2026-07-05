"""Tier-4 causal evaluators.

Tier 4 asks the only question we ultimately care about: does the latent code Z
enable accurate individualised treatment-effect estimation (section 13)? Three
evaluators are provided, in increasing fidelity and cost:

- ``tier4_semisynthetic`` (default, representation-aware, cheap): build a
  semi-synthetic potential-outcomes problem *on the arm's own latents Z*, with a
  known CATE and a confounded treatment assignment, then estimate the CATE with a
  cross-fitted T-learner and report root-PEHE, ATE bias and prescriptive
  accuracy. Because the covariates are Z, the score reflects how causally usable
  that specific representation is. The InterSynth "Environment stress test" (9.6)
  is realised by shifting the propensity between environments and reporting the
  root-PEHE gap.

- ``tier4_pfn_synthetic`` (sanity, representation-agnostic): train a small
  prior-fitted network on the Neuro-Prior and evaluate root-PEHE, reusing the
  Stage-2 model exactly as :mod:`neurocausalpfn.train.curriculum` does. Used for
  E12 (the curriculum ablation is CausalPFN-only).

- ``tier4_pfn_real`` (production, full mode): route to
  :func:`neurocausalpfn.train.run_stage2_real.run_stage2_real`, which feeds the
  frozen encoder latents as the in-context cohort of the real CausalPFN. This is
  the authoritative Tier-4 path; it needs the real cohort and is wired but not
  run in prototype mode.

The Giles VAE-50 (disconnectome) baseline root-PEHE (0.349) is the stop/go threshold.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..eval.equity import stratified_pehe
from ..eval.metrics import prescriptive_accuracy, root_pehe


def _standardize(Z: np.ndarray) -> np.ndarray:
    Z = np.asarray(Z, dtype=np.float64)
    sd = Z.std(axis=0)
    sd[sd == 0] = 1.0
    return (Z - Z.mean(axis=0)) / sd


def make_semisynthetic_po(Z: np.ndarray, seed: int = 0, effect_scale: float = 0.25,
                          confound_strength: float = 1.0,
                          environment: int = 0) -> Dict[str, np.ndarray]:
    """A semi-synthetic potential-outcomes cohort on the latents Z.

    tau(Z) and the baseline mu0(Z) are fixed linear-plus-mild-nonlinear functions
    of a few standardized latent directions; the treatment is assigned with
    Z-dependent (observed) confounding. Outcomes are squashed to [0, 1] to match
    the bounded, mRS-like scale on which the 0.349 baseline is defined. The
    ``environment`` index flips the confounding direction to realise the A->B
    stress test.
    """
    rng = np.random.default_rng(seed)
    Zs = _standardize(Z)
    n, d = Zs.shape
    k = min(d, 5)

    w_tau = rng.normal(size=d) / np.sqrt(d)
    w_mu = rng.normal(size=d) / np.sqrt(d)
    # a mild nonlinearity so a linear probe cannot trivially be exact
    tau = effect_scale * (Zs @ w_tau + 0.3 * np.tanh(Zs[:, :k].sum(1)))
    mu0 = 0.5 + 0.3 * (Zs @ w_mu)

    gamma = rng.normal(size=d) / np.sqrt(d)
    sign = -1.0 if environment % 2 else 1.0
    logits = sign * confound_strength * (Zs @ gamma)
    e = 1.0 / (1.0 + np.exp(-logits))
    T = (rng.uniform(size=n) < e).astype(np.float64)

    noise = 0.05 * rng.standard_normal(n)
    Y = np.clip(mu0 + T * tau + noise, 0.0, 1.0)
    return {"Z": Zs, "T": T, "Y": Y, "tau": tau, "mu0": mu0, "e": e}


def _t_learner(Z, T, Y, estimator: str, seed: int):
    """Cross-fitted T-learner: two outcome models, one per arm, evaluated on a
    held-out half so root-PEHE reflects generalisation, not memorisation."""
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(Y))
    tr, te = train_test_split(idx, test_size=0.5, random_state=seed)

    def fit_predict(mask_val):
        if estimator == "gbm":
            from sklearn.ensemble import GradientBoostingRegressor
            m = GradientBoostingRegressor(random_state=seed, n_estimators=80,
                                          max_depth=2)
        else:
            from sklearn.linear_model import Ridge
            m = Ridge(alpha=1.0)
        sel = tr[T[tr] == mask_val]
        if sel.size < 3:                      # degenerate arm: fall back to mean
            from sklearn.dummy import DummyRegressor
            m = DummyRegressor(strategy="mean")
            sel = tr
        m.fit(Z[sel], Y[sel])
        return m

    m0, m1 = fit_predict(0.0), fit_predict(1.0)
    tau_hat = m1.predict(Z) - m0.predict(Z)
    return tau_hat, te


def tier4_semisynthetic(Z: np.ndarray, seed: int = 0, estimator: str = "ridge",
                        strata: Optional[Dict[str, np.ndarray]] = None,
                        effect_scale: float = 0.25,
                        confound_strength: float = 1.0,
                        with_ood: bool = False) -> Dict:
    """Representation-aware Tier-4 evaluation on Z. Returns root-PEHE, ATE bias,
    prescriptive accuracy, optional OOD gap and the equity breakdown."""
    po = make_semisynthetic_po(Z, seed=seed, effect_scale=effect_scale,
                               confound_strength=confound_strength, environment=0)
    tau_hat, te = _t_learner(po["Z"], po["T"], po["Y"], estimator, seed)
    tau_true = po["tau"]

    out = {
        "root_pehe": root_pehe(tau_hat[te], tau_true[te]),
        "ate_bias": float(np.mean(tau_hat[te]) - np.mean(tau_true[te])),
        "prescriptive_accuracy": prescriptive_accuracy(tau_hat[te], tau_true[te]),
        "n_eval": int(te.size),
        "estimator": estimator,
    }

    if strata:
        eq = {}
        for name, g in strata.items():
            g = np.asarray(g)[: len(tau_true)]
            eq[name] = stratified_pehe(tau_hat[te], tau_true[te], g[te])
        out["equity"] = eq

    if with_ood:
        po_b = make_semisynthetic_po(Z, seed=seed, effect_scale=effect_scale,
                                     confound_strength=confound_strength,
                                     environment=1)
        tau_hat_b, te_b = _t_learner(po_b["Z"], po_b["T"], po_b["Y"], estimator, seed)
        rp_b = root_pehe(tau_hat_b[te_b], po_b["tau"][te_b])
        out["ood_root_pehe_envB"] = rp_b
        out["ood_gap"] = float(rp_b - out["root_pehe"])
    return out


def tier4_pfn_synthetic(cfg: Dict, iters: int = 60, n_eval: int = 16) -> Dict:
    """Sanity Tier-4 via a small prior-fitted network on the Neuro-Prior.

    Reuses the Stage-2 model, prior and CATE evaluation. Representation-agnostic
    (the synthetic prior generates its own covariates), so it validates the
    causal head and the pipeline rather than a specific encoder. Used by E12.
    """
    import torch

    from ..pfn.tokens import to_tensors
    from ..pfn.inference import predict_cate
    from ..prior.cohort import NeuroPrior
    from ..train.train_pfn import build_model

    p = cfg["pfn"]
    device = cfg.get("device", "cpu")
    model = build_model(cfg, p["d_x"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=p.get("lr", 3e-4),
                            weight_decay=p.get("weight_decay", 0.01))
    model.train()
    n_ctx = int(p.get("context_max", 192))
    for step in range(iters):
        prior = NeuroPrior(d_x=p["d_x"], n_context=n_ctx, n_query=p["n_query"],
                           seed=cfg.get("seed", 0) + step)
        batch = to_tensors(prior.sample_batch(p.get("batch_size", 8)), device=device)
        logits = model(batch["Xc"], batch["Tc"], batch["Yc"], batch["Xq"], batch["Tq"])
        loss = model.head.loss(logits, batch["mu_q"])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), p.get("grad_clip", 1.0))
        opt.step()

    prior = NeuroPrior(d_x=p["d_x"], n_context=n_ctx, n_query=p["n_query"],
                       seed=10_000 + cfg.get("seed", 0))
    batch = to_tensors(prior.sample_batch(n_eval), device=device)
    pred = predict_cate(model, batch["Xc"], batch["Tc"], batch["Yc"], batch["Xq"])
    cate_true = batch["mu1"] - batch["mu0"]
    return {"root_pehe": float(root_pehe(pred["cate"], cate_true)),
            "prescriptive_accuracy": float(prescriptive_accuracy(pred["cate"], cate_true))}


def tier4_pfn_real(cfg: Dict) -> Dict:
    """Production Tier-4: the real CausalPFN over the frozen encoder latents.

    Delegates to run_stage2_real, which loads the frozen encoders, builds the
    anatomical Neuro-Prior with the exported latents as z_pool and evaluates
    root-PEHE on the real cohort. Requires the real data and the trained
    encoders, so it is wired for full mode and not exercised in prototype.
    """
    from ..train.run_stage2_real import run_stage2_real

    model, history = run_stage2_real(cfg)
    rp = None
    for h in reversed(history or []):
        if isinstance(h, dict) and "root_pehe" in h:
            rp = float(h["root_pehe"]); break
    return {"root_pehe": rp, "history_tail": (history or [])[-1:]}
