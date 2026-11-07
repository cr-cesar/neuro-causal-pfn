"""Brief 8 audit (7 Nov): scoring the noisy cells against the CATE they imply,
and the allocation-by-responsiveness cells."""
import numpy as np
import pytest

from neurocausalpfn.prior import giles_replica as gr


def test_implied_tau_matches_the_trial_potential_outcomes():
    # mu_t = 1 - (1 - TE r_t)(1 - RE) for a W1-only, a W0-only and a both patient
    y = np.array([1.0, 0.0, 0.5])
    for te, re in ((1.0, 0.0), (0.75, 0.25), (0.5, 0.5), (0.25, 0.75)):
        r1, r0 = np.array([1, 0, 1]), np.array([0, 1, 1])
        mu1 = 1 - (1 - te * r1) * (1 - re)
        mu0 = 1 - (1 - te * r0) * (1 - re)
        assert np.allclose(gr.implied_tau(y, te, re), mu1 - mu0)


def test_implied_equals_paper_scale_in_the_ideal_cell():
    rng = np.random.default_rng(0)
    y = rng.choice([0.0, 0.5, 1.0], size=200)
    p1, p0 = rng.uniform(size=200), rng.uniform(size=200)
    s = gr.score_predictions(p1, p0, y)
    assert gr.pehe(p1 - p0, gr.implied_tau(y, 1.0, 0.0)) == pytest.approx(s["pehe_paper"])


def test_a_model_that_ignores_te_is_flattered_by_the_paper_scale():
    # predicting +-1 in a TE = 0.5, RE = 0.5 cell is perfect on the paper scale
    # and wrong by 0.75 on the implied scale for every decidable patient
    y = np.array([1.0, 0.0, 1.0, 0.0])
    tau_hat = 2 * y - 1
    assert gr.pehe(tau_hat, 2 * y - 1) == 0.0
    assert gr.pehe(tau_hat, gr.implied_tau(y, 0.5, 0.5)) == pytest.approx(0.75)


def test_responsiveness_cells_exist_and_use_the_unobserved_type():
    for b in (60, 75, 90):
        sc = gr.HEADLINE_SCENARIOS[f"responsiveness_{b:03d}"]
        assert sc["BIASTYPE"] == "unobserved" and sc["TE"] == 1.0 and sc["RE"] == 0.0
        assert sc["BIAS"] == pytest.approx(b / 100)


def test_responsiveness_allocation_favours_the_optimal_arm():
    import pandas as pd
    n = 4000
    y = np.r_[np.ones(n // 2), np.zeros(n // 2)]
    train = pd.DataFrame({"y_true": y})
    trial = gr.GilesTrial(1.0, 0.0, 0.9, "unobserved", k=0)
    W, _ = trial.simulate(train, roi_centroids=None)
    assert W[y == 1].mean() == pytest.approx(0.9, abs=0.03)
    assert W[y == 0].mean() == pytest.approx(0.1, abs=0.03)
