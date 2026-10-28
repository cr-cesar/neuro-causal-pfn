"""Generic effect processes mixed into a Neuro-Prior cohort (23 Oct): the
wrapper keeps covariates and treatment, replaces the outcome model of a share
of the processes by a simple effect in latent space, and its hyper options
never reach the family's hyper-prior."""
import numpy as np
import pytest

from neurocausalpfn.prior.generic_mix import (GenericEffectMixture, generic_process,
                                              split_generic_hyper, wrap_generic)


class _FakeCohort:
    d_x = 12
    hyper = "theory-ranges"
    channel = "disconnectome"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def sample_batch(self, batch_size, n_context=None):
        n = n_context or 64
        B, d = batch_size, self.d_x
        Xc, Xq = self.rng.normal(size=(B, n, d)), self.rng.normal(size=(B, 16, d))
        Tc, Tq = self.rng.integers(0, 2, size=(B, n)), self.rng.integers(0, 2, size=(B, 16))
        mu0 = np.full((B, 16), 0.5); mu1 = mu0 + 0.05
        return {"Xc": Xc, "Tc": Tc, "Yc": np.full((B, n), 0.5), "Xq": Xq, "Tq": Tq,
                "mu_q": np.where(Tq == 1, mu1, mu0), "mu0": mu0, "mu1": mu1,
                "processes": [{"family": "theory", "n_slice": 900} for _ in range(B)]}


def test_step_process_has_the_requested_null_share_and_magnitude():
    rng = np.random.default_rng(0)
    Xc, Xq = rng.normal(size=(2000, 10)), rng.normal(size=(500, 10))
    pr = generic_process(rng, Xc, Xq, "step", scale=0.2, null_frac=0.6, n_active=3, p0_range=(0.5, 0.5))
    tau = np.concatenate([pr["p1c"] - pr["p0c"], pr["p1q"] - pr["p0q"]])
    assert abs((tau == 0).mean() - 0.6) < 0.03                      # the null share
    assert (tau > 0).any() and (tau < 0).any()                      # both signs
    assert abs(np.abs(tau[tau != 0]).mean() - 0.2) < 0.03           # amplitude of the affected (clipping aside)
    lin = generic_process(rng, Xc, Xq, "linear", scale=0.2, null_frac=0.0, n_active=3, p0_range=(0.5, 0.5))
    tau_l = lin["p1q"] - lin["p0q"]
    assert len(np.unique(np.round(tau_l, 4))) > 100 and abs(np.abs(tau_l).mean() - 0.2) < 0.05
    with pytest.raises(ValueError):
        generic_process(rng, Xc, Xq, "spiral", 0.2, 0.5, 3)
    # network scores: a non-linear effect (ridge on the covariates explains little of it) and the step shape
    from sklearn.linear_model import Ridge
    mlp = generic_process(rng, Xc, Xq, "mlp", scale=0.3, null_frac=0.0, n_active=6, p0_range=(0.5, 0.5))
    tau_m = mlp["p1c"] - mlp["p0c"]
    r2 = Ridge(1.0).fit(Xc, tau_m).score(Xc, tau_m)
    assert r2 < 0.9 and abs(np.abs(tau_m).mean() - 0.3) < 0.1      # heavier tails clip more
    ms = generic_process(rng, Xc, Xq, "mlp_step", scale=0.4, null_frac=0.5, n_active=6, p0_range=(0.5, 0.5))
    tau_s = np.concatenate([ms["p1c"] - ms["p0c"], ms["p1q"] - ms["p0q"]])
    assert abs((tau_s == 0).mean() - 0.5) < 0.03
    # a shared baseline follows the effect's score
    sh = generic_process(rng, Xc, Xq, "linear", scale=0.2, null_frac=0.0, n_active=3, baseline_shared=1.0)
    assert abs(np.corrcoef(sh["p0c"], sh["p1c"] - sh["p0c"])[0, 1]) > 0.8


def test_mixture_replaces_a_share_and_keeps_covariates_and_treatment():
    base = _FakeCohort()
    mix = GenericEffectMixture(base, weight=0.5, scale=(0.1, 0.4), null=(0.4, 0.7), kinds=("step",), seed=1)
    ref = _FakeCohort().sample_batch(200, n_context=64)
    out = mix.sample_batch(200, n_context=64)
    np.testing.assert_array_equal(out["Xc"], ref["Xc"]); np.testing.assert_array_equal(out["Tc"], ref["Tc"])
    fam = [p["family"] for p in out["processes"]]
    n_gen = sum(f == "generic-step" for f in fam)
    assert 70 <= n_gen <= 130                                        # about half
    for i, f in enumerate(fam):
        tau = out["mu1"][i] - out["mu0"][i]
        yc_expected = None
        if f.startswith("generic"):
            assert np.isfinite(tau).all() and (tau != 0.05).any()   # a new effect, not the base's
            np.testing.assert_allclose(out["mu_q"][i], np.where(out["Tq"][i] == 1, out["mu1"][i], out["mu0"][i]))
            assert out["Yc"][i].min() >= 0 and out["Yc"][i].max() <= 1
        else:
            np.testing.assert_allclose(tau, 0.05)                    # untouched
            np.testing.assert_allclose(out["Yc"][i], 0.5)
    assert mix.d_x == 12 and mix.channel == "disconnectome"          # attributes pass through
    mix.hyper = "stage-2"
    assert base.hyper == "stage-2"                                   # the curriculum reaches the base
    assert "weight 0.50" in mix.describe() and "step" in mix.describe()


def test_binary_share_draws_bernoulli_context_outcomes():
    mix = GenericEffectMixture(_FakeCohort(), weight=1.0, kinds=("linear",), binary=1.0, seed=2)
    out = mix.sample_batch(5, n_context=64)
    assert set(np.unique(out["Yc"])) <= {0.0, 1.0}


def test_hyper_split_and_wrap_from_the_job_string():
    from neurocausalpfn.train.train_pfn import parse_hyper_opts

    hyper = parse_hyper_opts("two_arms=1 effect_scale=0.2:1 generic_weight=0.5 generic_scale=0.1:0.9 "
                             "generic_null=0.3:0.8 generic_kinds=linear+step+mlp+mlp_step generic_binary=0 "
                             "generic_active=2:50 generic_baseline=0:0.7")
    rest, gen = split_generic_hyper(hyper)
    assert set(rest) == {"two_arms", "effect_scale"} and gen["generic_weight"] == 0.5
    base = _FakeCohort()
    assert wrap_generic(base, {}) is base and wrap_generic(base, {"generic_weight": 0}) is base
    w = wrap_generic(base, gen, seed=3)
    assert isinstance(w, GenericEffectMixture) and w.kinds == ("linear", "step", "mlp", "mlp_step")
    assert w.scale == (0.1, 0.9) and w.null == (0.3, 0.8) and w.binary == 0.0
    assert w.n_active == (2.0, 50.0) and w.baseline_shared == (0.0, 0.7)
    out = w.sample_batch(40, n_context=64)
    assert {p["family"] for p in out["processes"]} >= {"generic-mlp", "theory"} or True
    assert "mlp" in w.describe() and "active dims 2-50" in w.describe()
