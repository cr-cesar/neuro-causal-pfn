"""Audit items of 14 Oct: the location-bias sweep scenarios (F30), the paired
test over deficits (F31), the B-arm column option and the clipping
diagnostics of the extended Neuro-Prior (F34)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

from neurocausalpfn.prior import giles_replica as gr
from neurocausalpfn.prior.intersynth_theory import TheoryHyperPrior
from tests.test_theory_extensions import _dgp, _pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import paired_deficit_test as pdt      # noqa: E402
import prior_diagnostics as pdiag      # noqa: E402


def test_bias_sweep_scenarios_exist_and_keep_the_other_knobs():
    for b, name in ((0.25, "location_bias_025"), (0.5, "location_bias_050"), (0.75, "location_bias_075"), (0.95, "location_bias_095")):
        sc = gr.HEADLINE_SCENARIOS[name]
        assert sc["BIAS"] == pytest.approx(b) and sc["TE"] == 1.0 and sc["RE"] == 0.0 and sc["BIASTYPE"] == "observed"
    assert gr.HEADLINE_SCENARIOS["location_bias"]["BIAS"] == 1.0 and gr.HEADLINE_SCENARIOS["ideal"]["BIAS"] == 0.0


def test_paired_summary_over_deficits(tmp_path):
    rng = np.random.default_rng(0)
    deficits = list(range(1, 17))
    rows = []
    for d in deficits:
        for fold in range(3):
            base = 0.3 + 0.01 * d
            rows.append({"deficit": d, "fold": fold, "classifier": "lr", "learner": "two", "pehe_paper": base + rng.normal(0, 0.005)})
            rows.append({"deficit": d, "fold": fold, "classifier": "pfn", "learner": "in_context", "pehe_paper": base - 0.02 + rng.normal(0, 0.005)})
    f = tmp_path / "replica_results.csv"
    pd.DataFrame(rows).to_csv(f, index=False)
    s = pdt.main(["--a", str(f), "--a-filter", "classifier=pfn", "--b", str(f), "--b-filter", "classifier=lr",
                  "--label-a", "PFN", "--label-b", "LR", "--out", str(tmp_path / "out.csv")])
    assert s["n_deficits"] == 16
    assert s["diff_mean"] == pytest.approx(-0.02, abs=0.005)
    assert s["diff_ci_high"] < 0 and s["paired_t_p"] < 1e-3 and s["wilcoxon_p"] < 1e-3
    assert s["n_a_lower"] == 16
    assert (tmp_path / "out_per_deficit.csv").exists()
    with pytest.raises(ValueError):
        pdt.main(["--a", str(f), "--a-filter", "classifier=nope", "--b", str(f)])


def test_s_b_column_selects_the_subnetwork_column_of_arm_b():
    atlas, ov, _, _ = _pool()
    h0 = TheoryHyperPrior(two_arms=1.0, s_b_column=0)
    h1 = TheoryHyperPrior(two_arms=1.0, s_b_column=1)
    d0, d1 = _dgp(atlas, ov, h0, seed=3), _dgp(atlas, ov, h1, seed=3)
    assert np.allclose(d0.w_s2, d1.w_s2)                       # same draw, other column
    # the B susceptibility of the default reads the column D reads
    sb0 = ov[:, :, 0] @ d0.w_s2 / d0.scale_b
    sb1 = ov[:, :, 1] @ d1.w_s2 / d1.scale_b
    assert np