"""The Giles virtual-trial replica: labelling, trial simulation, PEHE scale."""
import numpy as np
import pandas as pd
import pytest

from neurocausalpfn.prior import giles_replica as gr


def _fake_pair(shape=(20, 22, 20)):
    arr = np.zeros(shape)
    arr[2:8, 2:8, 2:8] = 1      # subnetwork 1 (larger -> treatment "1")
    arr[12:16, 12:16, 12:16] = 2
    masks = [(arr == 1).astype(int), (arr == 2).astype(int)]
    vols = [int(m.sum()) for m in masks]
    treatments = ["1", "0"] if int(np.argmax(vols)) == 0 else ["0", "1"]
    cents = ([5.0, 14.0], [5.0, 14.0], [5.0, 14.0])
    return gr.RoiPair(masks, vols, treatments, cents)


def _labels(n=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        w1, w0 = rng.random() < 0.45, rng.random() < 0.45
        y = 0.5 if (w1 and w0) else (1.0 if w1 else (0.0 if w0 else np.nan))
        if np.isnan(y):
            w1 = True
            y = 1.0
        rows.append({"filename": f"lesion{i:04d}.nii.gz", "vol": int(rng.integers(50, 500)),
                     "centroid_x": float(rng.uniform(0, 20)),
                     "centroid_y": float(rng.uniform(0, 22)),
                     "centroid_z": float(rng.uniform(0, 20)),
                     "hearing_W1": int(w1), "hearing_W0": int(w0)})
    return pd.DataFrame(rows)


def test_deficit_slice_y_true_encoding():
    df = _labels()
    sl = gr.deficit_slice(df, 1)
    assert set(np.unique(sl["y_true"])) <= {0.0, 0.5, 1.0}
    both = (sl["hearing_W0"] == 1) & (sl["hearing_W1"] == 1)
    assert (sl.loc[both, "y_true"] == 0.5).all()


def test_trial_is_deterministic_per_fold():
    df = gr.deficit_slice(_labels(), 1)
    pair = _fake_pair()
    W1, Y1 = gr.GilesTrial(0.75, 0.25, 0.5, "observed", k=3).simulate(df, pair.centroids)
    W2, Y2 = gr.GilesTrial(0.75, 0.25, 0.5, "observed", k=3).simulate(df, pair.centroids)
    W3, _ = gr.GilesTrial(0.75, 0.25, 0.5, "observed", k=4).simulate(df, pair.centroids)
    assert (W1 == W2).all() and (Y1 == Y2).all()
    assert not (W1 == W3).all()          # a different fold reseeds


def test_random_allocation_and_full_te():
    df = gr.deficit_slice(_labels(200), 1)
    pair = _fake_pair()
    W, Y = gr.GilesTrial(1.0, 0.0, 0.0, "observed", k=0).simulate(df, pair.centroids)
    # TE=1, RE=0: exactly the treated-susceptible respond
    susceptible = ((W == df["y_true"].to_numpy()) | (df["y_true"].to_numpy() == 0.5))
    assert (Y[~susceptible] == 0).all()
    assert Y[susceptible].mean() == 1.0
    assert 0.3 < W.mean() < 0.7          # roughly balanced allocation


def test_score_predictions_scale():
    true = np.array([1.0, 0.0, 0.5, 1.0])
    perfect = gr.score_predictions(np.array([9, -9, 0, 9.0]),
                                   np.array([-9, 9, 0, -9.0]), true)
    poor = gr.score_predictions(np.zeros(4), np.zeros(4), true)
    assert perfect["pehe"] < 0.01 and perfect["prescriptive_balacc"] == 1.0
    assert poor["pehe"] == pytest.approx(np.sqrt(3 * 0.25 / 4), abs=1e-6)


def test_evaluate_representation_end_to_end():
    df = _labels(120)
    pair = _fake_pair()
    Z = np.hstack([df[["centroid_x", "centroid_y", "centroid_z"]].to_numpy(),
                   df["y_true_hint"].to_numpy()[:, None]]) if "y_true_hint" in df else \
        np.random.default_rng(0).normal(size=(120, 4))
    sims = []
    res = gr.evaluate_representation(Z, df, {1: pair},
                                     gr.HEADLINE_SCENARIOS["ideal"], n_folds=4,
                                     deficits=[1], classifiers=["logistic_regression"],
                                     collect_sims=sims)
    assert len(res) > 0 and np.isfinite(res["pehe"]).all()
    assert len(sims) > 0 and {"filename", "W", "Y", "y_true"} <= set(sims[0])
    agg = gr.headline_aggregate(res)
    assert np.isfinite(agg["pehe_mean"])
