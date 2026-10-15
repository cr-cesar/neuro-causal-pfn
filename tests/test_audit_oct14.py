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


def test_score_continuous_prior_scores_a_fold_on_drawn_processes(tmp_path):
    from scripts.score_continuous_prior import draw_processes, score_fold

    atlas, ov = _pool(n=200)
    rng = np.random.default_rng(0)
    files = np.asarray([f"img{i:04d}.nii.gz" for i in range(200)])
    cache = {"overlaps": ov, "centroids": rng.uniform(0, 16, (200, 3)),
             "volumes": rng.uniform(100, 5000, 200), "files": files[::-1].copy()}   # shuffled order: alignment by name
    tr, te = np.arange(160), np.arange(160, 200)
    Z = rng.normal(size=(200, 6))
    np.savez(tmp_path / "fold0.npz", Ztr=Z[tr], Zte=Z[te], tr_idx=tr, te_idx=te, files=files, fold=0)
    dgps = draw_processes(cache, atlas, {"two_arms": 1.0, "effect_scale": [0.2, 1.0]}, n=2, seed=1)
    rows = score_fold(str(tmp_path / "fold0.npz"), cache, dgps, volume=False, seed=0)
    assert len(rows) == 2 * 2                      # 2 processes x 2 estimators
    assert all(np.isfinite(r["pehe"]) and r["pehe"] >= 0 for r in rows)
    vol = score_fold(str(tmp_path / "fold0.npz"), cache, dgps, volume=True, seed=0)
    assert len(vol) == 4


def test_score_continuous_prior_refuses_misaligned_cache(tmp_path):
    from scripts.score_continuous_prior import _align

    with pytest.raises(ValueError):
        _align(np.asarray(["a.nii", "b.nii"]), ["a.nii", "c.nii"])


def test_plot_bias_sweep_reads_the_scenario_roots(tmp_path):
    pytest.importorskip("matplotlib")
    from scripts.plot_bias_sweep import CELLS, main

    for bias, suf in CELLS:
        for t4, base in (("", 0.30), ("_t4", 0.28)):
            d = tmp_path / f"outputs_perfold_pub{suf}{t4}"
            d.mkdir()
            pd.DataFrame([{"eid": "E1", "budget": "published", "channel": "disconnectome",
                           "pehe_paper_mean": base + 0.07 * bias, "pehe_paper_ci_low": base + 0.07 * bias - 0.02,
                           "pehe_paper_ci_high": base + 0.07 * bias + 0.02, "balacc_mean": 0.84 - 0.08 * bias}]
                         ).to_csv(d / "leaderboard.csv", index=False)
    out = tmp_path / "sweep.png"
    df = main(["--perfold", str(tmp_path), "--out", str(out)])
    assert out.exists() and (tmp_path / "sweep.csv").exists()
    assert df["pehe"].notna().all() and len(df) == 2 * len(CELLS)


def test_paired_test_loads_label_folders_with_brackets(tmp_path):
    from scripts.paired_deficit_test import _load

    d = tmp_path / "E3[backbone=cnn]" / "seed0" / "x-singles"
    d.mkdir(parents=True)
    pd.DataFrame({"deficit": ["a", "a", "b"], "fold": [0, 1, 0], "classifier": ["causalpfn"] * 3,
                  "pehe_paper": [0.2, 0.4, 0.5]}).to_csv(d / "replica_results.csv", index=False)
    s = _load([str(tmp_path / "E3[backbone=cnn]" / "seed*" / "*-singles" / "replica_results.csv")],
              {"classifier": "causalpfn"}, "pehe_paper")
    assert s.loc["a"] == pytest.approx(0.3) and s.loc["b"] == pytest.approx(0.5)


def test_compare_atlas_splits_detects_a_hemispheric_cut(tmp_path):
    nib = pytest.importorskip("nibabel")
    from scripts.compare_atlas_splits import main

    base = tmp_path / "2mm_parcellations"
    (base / "receptor").mkdir(parents=True)
    (base / "genetics").mkdir(parents=True)
    # 2 mm grid whose world x crosses zero at the middle voxel
    aff = np.diag([2.0, 2.0, 2.0, 1.0]); aff[0, 3] = -10.0
    shape = (10, 6, 6)
    rec = np.zeros(shape); rec[:, :3, :] = 1; rec[:, 3:, :] = 2      # anterior / posterior cut
    gen = np.zeros(shape); gen[:5] = 1; gen[5:] = 2                  # left / right cut
    for name, vol in (("receptor", rec), ("genetics", gen)):
        nib.save(nib.Nifti1Image(vol, aff), str(base / name / "1_net.nii.gz"))
    df = main(["--atlas-dir", str(tmp_path), "--out", str(tmp_path / "splits.csv")])
    assert len(df) == 1
    # the genetics halves separate by hemisphere, the receptor halves do not
    gsep = abs(df["genetics_left_frac_1"][0] - df["genetics_left_frac_2"][0])
    rsep = abs(df["receptor_left_frac_1"][0] - df["receptor_left_frac_2"][0])
    assert gsep == pytest.approx(1.0) and rsep == pytest.approx(0.0)
    assert df["dice_best_pairing"][0] < 0.8                           # unrelated cuts
    assert (tmp_path / "splits.csv").exists()
