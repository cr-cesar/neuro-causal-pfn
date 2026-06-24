import numpy as np

from neurocausalpfn.prior.cohort import NeuroPrior
from neurocausalpfn.prior.intersynth import SyntheticDGP
from neurocausalpfn.train.curriculum import (evaluate_pehe, prototype_config,
                                             run_curriculum,
                                             run_curriculum_ablation,
                                             stages_for_variant)


# --------------------------- curriculum variants --------------------------- #
def test_curriculum_variants():
    cfg = prototype_config()
    ref = stages_for_variant(cfg, "reference")
    two = stages_for_variant(cfg, "two_stage")
    one = stages_for_variant(cfg, "one_stage")
    assert len(ref) == 3 and len(two) == 2 and len(one) == 1
    total = sum(s["steps"] for s in ref)
    assert one[0]["steps"] == total                       # one-stage runs the full budget
    assert two == ref[1:]                                 # two-stage skips stage 1


# ------------------- effect scale / prior backward compatibility ----------- #
def test_effect_scale_backward_compatible():
    dgp_default = SyntheticDGP(8, np.random.default_rng(0))
    dgp_one = SyntheticDGP(8, np.random.default_rng(0), effect_scale=1.0)
    assert np.allclose(dgp_default.delta, dgp_one.delta)
    assert dgp_default.b1 == dgp_one.b1
    dgp_half = SyntheticDGP(8, np.random.default_rng(0), effect_scale=0.5)
    assert np.allclose(dgp_half.delta, dgp_default.delta * 0.5)


def test_neuroprior_default_deterministic():
    b1 = NeuroPrior(d_x=16, n_context=32, n_query=8, seed=3).sample_batch(2)
    b2 = NeuroPrior(d_x=16, n_context=32, n_query=8, seed=3).sample_batch(2)
    assert all(np.allclose(b1[k], b2[k]) for k in b1)


def test_neuroprior_with_ranges_runs():
    prior = NeuroPrior(d_x=16, n_context=32, n_query=8, seed=0,
                       confound_range=(0.0, 0.3), effect_range=(0.2, 0.4))
    batch = prior.sample_batch(2)
    for key in ("Xc", "Tc", "Yc", "Xq", "Tq", "mu_q", "mu0", "mu1"):
        assert key in batch and np.isfinite(batch[key]).all()


# ------------------------------- ablation ---------------------------------- #
def _tiny_cfg():
    cfg = prototype_config()
    cfg["curriculum"]["stage1"].update({"steps": 3, "n_context": 24})
    cfg["curriculum"]["stage2"].update({"steps": 2, "n_context": 32})
    cfg["curriculum"]["stage3"].update({"steps": 1, "n_context": 40})
    cfg["pfn"]["context_max"] = 40
    cfg["pfn"]["batch_size"] = 4
    return cfg


def test_run_curriculum_records_stage_boundaries():
    cfg = _tiny_cfg()
    _, hist = run_curriculum(cfg, "reference")
    assert len(hist) == 6                                  # 3 + 2 + 1
    assert {h["stage"] for h in hist} == {0, 1, 2}
    assert all(np.isfinite(h["loss"]) for h in hist)


def test_evaluate_pehe_finite():
    cfg = _tiny_cfg()
    model, _ = run_curriculum(cfg, "two_stage")
    ev = evaluate_pehe(model, cfg, n_eval=4)
    assert np.isfinite(ev["root_pehe"]) and np.isfinite(ev["prescriptive_accuracy"])


def test_run_curriculum_ablation():
    cfg = _tiny_cfg()
    res = run_curriculum_ablation(cfg, variants=("reference", "two_stage", "one_stage"))
    assert set(res) == {"reference", "two_stage", "one_stage"}
    assert res["reference"]["steps"] == 6
    assert res["two_stage"]["steps"] == 3
    assert res["one_stage"]["steps"] == 6
    for r in res.values():
        assert np.isfinite(r["root_pehe"])
        assert 1 <= r["steps_to_half"] <= r["steps"]
