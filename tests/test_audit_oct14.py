"""Audit items of 14 Oct: the B-arm susceptibility column option (F34 Q12),
the location-bias sweep scenarios (F30), the paired deficit test (F31) and the
prior clipping diagnostics (F34 Q11)."""
import numpy as np
import pandas as pd
import pytest

from neurocausalpfn.prior import giles_replica as gr
from neurocausalpfn.prior.atlas import FunctionalAtlas
from neurocausalpfn.prior.intersynth_theory import InterSynthTheoryDGP, TheoryHyperPrior


def _pool(n=300, seed=0):
    atlas = FunctionalAtlas.synthetic(shape=(16, 18, 16), seed=seed)
    rng = np.random.default_rng(seed)
    ov = np.clip(rng.exponential(0.05, size=(n, atlas.n_networks, 2)), 0, 1).astype(np.float64)
    return atlas, ov


def _dgp(atlas, ov, hyper, seed=1):
    return InterSynthTheoryDGP(atlas, np.random.default_rng(seed), hyper).calibrate(ov)


def test_s_b_column_selects_which_overlap_column_feeds_the_B_arm():
    atlas, ov = _pool()
    a = _dgp(atlas, ov, TheoryHyperPrior(two_arms=1.0, s_b_column=0), seed=3)
    b = _dgp(atlas, ov, TheoryHyperPrior(two_arms=1.0, s_b_column=1), seed=3)
    assert a._col_b == 0 and b._col_b == 1
    # column 0 is the one D reads; column 1 is the one S_A reads. The B
    # susceptibility must differ between the two, and neither be all-zero.
    sb0, sb1 = a.scores_b(ov), b.scores_b(ov)
    assert sb0.std() > 0 and sb1.std() > 0
    assert not np.allclose(sb0, sb1)


def test_default_b_column_is_zero_and_reproduces_the_pilots():
    assert TheoryHyperPrior().s_b_column == 0
    atlas, ov = _pool()
    d = _dgp(atlas, ov, TheoryHyperPrior(two_arms=1.0))
    assert d._col_b == 0


def test_location_bias_sweep_scenarios_exist_and_order_by_bias():
    for b in (25, 50, 75, 95):
        key = f"location_bias_{b:03d}"
        assert key in gr.HEADLINE_SCENARIOS
        assert gr.HEADLINE_SCENARIOS[key]["BIAS"] == pytest.approx(b / 100)
    assert gr.HEADLINE_SCENARIOS["ideal"]["BIAS"] == 0.0
    assert gr.HEADLINE_SCENARIOS["location_bias"]["BIAS"] == 1.0


def test_paired_summary_reports_difference_ci_and_two_p_values():
    from scripts.paired_deficit_test import paired_summary

    idx = list(range(16))
    a = pd.Series(np.linspace(0.26, 0.30, 16), index=idx)   # A uniformly lower
    b = pd.Series(np.linspace(0.29, 0.33, 16), index=idx)
    s = paired_summary(a, b, "A", "B")
    assert s["n_deficits"] == 16
    assert s["diff_mean"] < 0 and s["diff_ci_high"] < 0      # A is lower everywhere
    assert s["n_a_lower"] == 16 and s["n_b_lower"] == 0
    assert 0 <= s["paired_t_p"] <= 1 and 0 <= s["wilcoxon_p"] <= 1
    # common index only: an extra deficit in one series is dropped
    s2 = paired_summary(pd.Series([0.3], index=[99]).combine_first(a), b, "A", "B")
    assert s2["n_deficits"] == 16


def test_prior_diagnostics_counts_clipping_and_effect_loss():
    from scripts.prior_diagnostics import process_diagnostics

    atlas, ov = _pool(n=240)
    # a large effect on a pool where mu0 is near 1 forces clipping
    dgp = _dgp(atlas, ov, TheoryHyperPrior(two_arms=1.0, effect_scale=(2.0, 2.0)), seed=5)
    out = process_diagnostics(dgp, ov, slice_idx=None)
    assert out["pool_n"] == 240
    assert 0.0 <= out["pool_frac_clip_any"] <= 1.0
    assert out["pool_frac_clip_any"] > 0                      # some lesions clip
    assert out["pool_mean_abs_cate"] <= out["pool_mean_abs_cate_raw"] + 1e-9
    assert 0.0 <= out["pool_effect_lost_rel"] <= 1.0
