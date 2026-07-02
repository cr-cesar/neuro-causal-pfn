"""The tiered evaluation harness (section 13: Intermediate benchmarking).

The tiers are ordered by increasing cost and specificity; failure at an earlier
tier renders later evaluation uninformative, so a hard gate short-circuits the
rest of the pipeline:

    T1  Reconstruction   Dice, BCE                       gate   Dice >= 0.70
    T2  Clinical align.  linear/MLP probe R2, rho, MAE   soft   R2(NIHSS) >= 0.05
    T3  Latent quality   active dims, IOSS, KL, PNS      info   (no threshold)
    T4  Causal           root-PEHE, ATE bias, equity     gate   root-PEHE < 0.349

Only the tiers listed for an experiment (its Table-9 row) are evaluated, and they
are evaluated in the cost order above. A failing hard gate stops the pipeline; a
failing soft gate (T2) deprioritises the configuration but lets it continue.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..eval.latent_quality import active_dimensions, ioss, mcc
from ..eval.probes import linear_probe, mlp_probe, stratified_probe
from .registry import TIER_GATES, Experiment
from .estimators import tier4_semisynthetic

# canonical cost order; experiments may request any subset
TIER_ORDER = ("T1", "T2", "T3", "T4")


@dataclass
class TierResult:
    tier: str
    metrics: Dict
    gate_metric: Optional[str] = None
    gate_value: Optional[float] = None
    passed: Optional[bool] = None      # None = informational
    kind: str = "info"
    stopped: bool = False              # this tier halted the pipeline
    deprioritized: bool = False        # soft-gate failure


@dataclass
class TierReport:
    eid: str
    results: List[TierResult] = field(default_factory=list)
    stopped_at: Optional[str] = None
    deprioritized: bool = False

    @property
    def passed(self) -> bool:
        """True when no hard gate failed."""
        return self.stopped_at is None

    def metric(self, tier: str, name: str):
        for r in self.results:
            if r.tier == tier:
                return r.metrics.get(name)
        return None

    def summary(self) -> Dict:
        out = {"eid": self.eid, "passed": self.passed,
               "stopped_at": self.stopped_at, "deprioritized": self.deprioritized}
        for r in self.results:
            for k, v in r.metrics.items():
                if isinstance(v, (int, float)) or v is None:
                    out[f"{r.tier}.{k}"] = v
        return out


# --------------------------------------------------------------------------- #
# Per-tier evaluators
# --------------------------------------------------------------------------- #
def evaluate_t1(artifacts) -> TierResult:
    g = TIER_GATES["T1"]
    metrics = {"dice": artifacts.dice, "bce": artifacts.bce}
    passed = g.passes(artifacts.dice)
    return TierResult("T1", metrics, g.metric, artifacts.dice, passed, g.kind)


def evaluate_t2(Z, target, groups: Optional[np.ndarray] = None,
                seed: int = 0) -> TierResult:
    g = TIER_GATES["T2"]
    lin = linear_probe(Z, target, seed=seed)
    mlp = mlp_probe(Z, target, seed=seed)
    metrics = {"r2_nihss": lin["r2"], "spearman": lin["spearman"], "mae": lin["mae"],
               "mlp_r2": mlp["r2"]}
    if groups is not None:
        metrics["stratified"] = stratified_probe(Z, target, groups, kind="linear", seed=seed)
    passed = g.passes(lin["r2"])
    return TierResult("T2", metrics, g.metric, lin["r2"], passed, g.kind)


def evaluate_t3(artifacts, factors: Optional[np.ndarray] = None,
                pns_outcome: Optional[np.ndarray] = None, seed: int = 0) -> TierResult:
    g = TIER_GATES["T3"]
    pdk = artifacts.per_dim_kl
    metrics = {
        "zdim": int(artifacts.Z.shape[1]),
        "ioss": float(ioss(artifacts.Z, seed=seed)),
    }
    if pdk is not None:
        metrics["active_dims"] = active_dimensions(pdk, threshold=0.01)
        metrics["kl_mean"] = float(np.mean(pdk))
        metrics["kl_max"] = float(np.max(pdk))
    else:
        metrics["active_dims"] = None
        metrics["note"] = "no posterior (non-VAE arm): KL diagnostics skipped"
    if factors is not None:
        metrics["mcc"] = float(mcc(artifacts.Z, factors))
    if pns_outcome is not None:
        try:
            from ..causal.pns import pns_lower_bound
            pns = pns_lower_bound(artifacts.Z, pns_outcome)
            metrics["pns_lower_bound_max"] = float(np.max(pns))
            metrics["pns_lower_bound_mean"] = float(np.mean(pns))
        except Exception as exc:  # pragma: no cover - diagnostic only
            metrics["pns_error"] = str(exc)
    return TierResult("T3", metrics, None, None, None, g.kind)


def evaluate_t4(Z=None, *, t4_result: Optional[Dict] = None,
                strata: Optional[Dict[str, np.ndarray]] = None, seed: int = 0,
                estimator: str = "ridge", with_ood: bool = False) -> TierResult:
    g = TIER_GATES["T4"]
    if t4_result is None:
        t4_result = tier4_semisynthetic(Z, seed=seed, estimator=estimator,
                                        strata=strata, with_ood=with_ood)
    rp = t4_result.get("root_pehe")
    metrics = dict(t4_result)
    passed = g.passes(rp)
    return TierResult("T4", metrics, g.metric, rp, passed, g.kind)


# --------------------------------------------------------------------------- #
# Orchestration across the tiers of one experiment
# --------------------------------------------------------------------------- #
def run_tiers(experiment: Experiment, artifacts, *, seed: int = 0,
              t2_target: Optional[np.ndarray] = None,
              strata: Optional[Dict[str, np.ndarray]] = None,
              factors: Optional[np.ndarray] = None,
              pns_outcome: Optional[np.ndarray] = None,
              t4_result: Optional[Dict] = None,
              t4_estimator: str = "ridge") -> TierReport:
    """Evaluate exactly the tiers requested by the experiment, in cost order,
    stopping at the first failed hard gate."""
    report = TierReport(eid=experiment.eid)
    requested = [t for t in TIER_ORDER if t in experiment.tiers]
    with_ood = "ood_gap" in experiment.extra_metrics
    Z = artifacts.Z

    for tier in requested:
        if tier == "T1":
            res = evaluate_t1(artifacts)
        elif tier == "T2":
            tgt = t2_target
            if tgt is None:
                from .artifacts import clinical_target
                tgt = clinical_target(artifacts, seed=seed)
            grp = strata.get("volume_quartile") if strata else None
            res = evaluate_t2(Z, tgt, groups=grp, seed=seed)
        elif tier == "T3":
            res = evaluate_t3(artifacts, factors=factors, pns_outcome=pns_outcome, seed=seed)
        elif tier == "T4":
            res = evaluate_t4(Z, t4_result=t4_result, strata=strata, seed=seed,
                              estimator=t4_estimator, with_ood=with_ood)
        else:  # pragma: no cover
            continue

        # apply the gate semantics
        if res.kind == "gate" and res.passed is False:
            res.stopped = True
            report.results.append(res)
            report.stopped_at = tier
            break
        if res.kind == "deprioritize" and res.passed is False:
            res.deprioritized = True
            report.deprioritized = True
        report.results.append(res)

    return report
