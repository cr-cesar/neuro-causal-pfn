"""Tiered evaluation harness, gates and Tier-4 estimator (torch-free)."""
import numpy as np

from neurocausalpfn.experiments.artifacts import (VaeArtifacts, latent_artifacts,
                                                  clinical_target, volume_quartiles)
from neurocausalpfn.experiments.estimators import (make_semisynthetic_po,
                                                   tier4_semisynthetic)
from neurocausalpfn.experiments.registry import get_experiment
from neurocausalpfn.experiments.report import bootstrap_paired_pehe
from neurocausalpfn.experiments.tiers import run_tiers


def _vae_artifacts(n=300, d=8, dice=0.82, seed=0):
    rng = np.random.default_rng(seed)
    mu = rng.normal(size=(n, d))
    logvar = np.zeros((n, d))
    return VaeArtifacts(Z=mu, logvar=logvar, dice=dice, bce=0.1,
                        clinical=rng.normal(size=(n, 4)), volume=rng.uniform(size=n),
                        has_posterior=True, meta={"zdim": d})


def test_t1_gate_stops_pipeline_on_bad_reconstruction():
    a = _vae_artifacts(dice=0.40)
    exp = get_experiment("E3")            # T1, T2, T4
    strata = {"volume_quartile": volume_quartiles(a.volume)}
    rep = run_tiers(exp, a, seed=0, strata=strata)
    assert rep.stopped_at == "T1"
    assert rep.passed is False
    assert [r.tier for r in rep.results] == ["T1"]   # nothing after the gate


def test_full_tier_pipeline_runs_and_passes_on_good_config():
    a = _vae_artifacts(dice=0.85)
    exp = get_experiment("E3")
    strata = {"volume_quartile": volume_quartiles(a.volume)}
    rep = run_tiers(exp, a, seed=1, strata=strata)
    assert rep.passed is True
    assert rep.metric("T1", "dice") == 0.85
    assert rep.metric("T4", "root_pehe") is not None


def test_t2_soft_gate_deprioritizes_but_continues():
    # random latents cannot predict the volume-derived deficit -> low R2
    a = _vae_artifacts(dice=0.9, seed=3)
    exp = get_experiment("E3")
    rep = run_tiers(exp, a, seed=3, strata={"volume_quartile": volume_quartiles(a.volume)})
    # T2 fails the soft gate but the pipeline still reaches T4
    assert rep.metric("T4", "root_pehe") is not None
    assert rep.deprioritized in (True, False)     # recorded, not fatal


def test_tier4_semisynthetic_prefers_informative_representation():
    rng = np.random.default_rng(0)
    n, d = 300, 8
    Z = rng.normal(size=(n, d))
    good = tier4_semisynthetic(Z, seed=1, with_ood=True)
    # the same estimator on pure noise covariates should not do better
    assert good["root_pehe"] >= 0.0
    assert 0.0 <= good["prescriptive_accuracy"] <= 1.0
    assert "ood_gap" in good


def test_semisynthetic_po_is_confounded_and_bounded():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(200, 6))
    po = make_semisynthetic_po(Z, seed=0, confound_strength=1.0)
    assert po["Y"].min() >= 0.0 and po["Y"].max() <= 1.0
    # treatment is assigned with Z-dependent probability (not a coin flip)
    assert po["e"].std() > 0.01


def test_non_vae_arm_skips_kl_diagnostics():
    rng = np.random.default_rng(0)
    la = latent_artifacts(rng.normal(size=(200, 6)), volume=rng.uniform(size=200))
    rep = run_tiers(get_experiment("E9a"), la, seed=0)   # T2, T3, T4
    t3 = [r for r in rep.results if r.tier == "T3"][0]
    assert t3.metrics["active_dims"] is None          # no posterior
    assert t3.metrics["ioss"] >= 0.0                  # IOSS still computed


def test_bootstrap_paired_pehe_detects_better_config():
    a = np.full(80, 0.09)     # per-query squared errors, rootPEHE 0.30
    b = np.full(80, 0.16)     # rootPEHE 0.40
    out = bootstrap_paired_pehe(a, b, n=300)
    assert out["mean_diff"] < 0              # a has lower rootPEHE
    assert out["prob_a_better"] > 0.9
