"""In-context estimators inside the Phase 1 replica: registry plumbing with a
numpy estimator (no torch) and, when torch is available, the PFN wrapper on a
prototype checkpoint."""
import os

import numpy as np
import pytest

from neurocausalpfn.prior import giles_replica as gr
from tests.test_giles_replica import _fake_pair, _labels


def _knn_estimator(k=5):
    """Nearest-neighbour potential outcomes: the simplest in-context estimator."""
    def fn(Xtr, W, Y, Xte):
        out = []
        for w in (1, 0):
            Xw, Yw = Xtr[W == w], Y[W == w]
            if len(Xw) == 0:
                out.append(np.full(len(Xte), 0.5)); continue
            d = ((Xte[:, None, :] - Xw[None, :, :]) ** 2).sum(-1)
            nn = np.argsort(d, axis=1)[:, :min(k, len(Xw))]
            out.append(Yw[nn].mean(1))
        return out[0], out[1]
    return fn


@pytest.fixture(autouse=True)
def _clean_registry():
    yield
    gr.ESTIMATORS.clear()


def test_registered_estimator_is_scored_once_per_deficit_and_fold():
    df = _labels(120)
    Z = np.random.default_rng(0).normal(size=(120, 4))
    gr.register_estimator("knn", _knn_estimator())
    res = gr.evaluate_representation(Z, df, {1: _fake_pair()}, gr.HEADLINE_SCENARIOS["ideal"],
                                     n_folds=4, deficits=[1],
                                     classifiers=["logistic_regression", "knn"])
    knn = res[res["classifier"] == "knn"]
    lr = res[res["classifier"] == "logistic_regression"]
    assert set(knn["learner"]) == {"in_context"} and set(lr["learner"]) == {"one", "two"}
    assert len(knn) == knn[["deficit", "fold"]].drop_duplicates().shape[0] == len(lr) / 2
    assert np.isfinite(knn["pehe"]).all() and np.isfinite(knn["pehe_paper"]).all()
    agg = gr.headline_aggregate(res)
    assert agg["classifier"] in ("knn", "logistic_regression")


def test_builtin_names_cannot_be_overridden():
    with pytest.raises(ValueError):
        gr.register_estimator("logistic_regression", _knn_estimator())


def test_pfn_estimator_on_prototype_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    from neurocausalpfn.pfn.estimator import PFNEstimator
    from neurocausalpfn.train.train_pfn import prototype_config, run_pfn

    cfg = prototype_config()
    cfg["out_dir"] = str(tmp_path); cfg["pfn"]["iters"] = 3; cfg["pfn"]["batch_size"] = 2
    cfg["pfn"]["d_x"] = 4; cfg["pfn"]["context_max"] = 32; cfg["pfn"]["n_query"] = 4
    cfg["device"] = "cpu"
    run_pfn(cfg)
    est = PFNEstimator(os.path.join(str(tmp_path), "pfn.pt"), device="cpu", max_queries=7)
    assert est.d_x == 4
    rng = np.random.default_rng(0)
    Xtr, Xte = rng.normal(size=(40, 4)), rng.normal(size=(20, 4))
    W, Y = rng.integers(0, 2, 40).astype(float), rng.integers(0, 2, 40).astype(float)
    p1, p0 = est(Xtr, W, Y, Xte)
    assert p1.shape == p0.shape == (20,) and (0 <= p1).all() and (p1 <= 1).all()
    with pytest.raises(ValueError):
        est(Xtr[:, :3], W, Y, Xte[:, :3])
    # and it slots into the replica like any other estimator
    df = _labels(80); Z = rng.normal(size=(80, 4))
    gr.register_estimator("pfn", est)
    res = gr.evaluate_representation(Z, df, {1: _fake_pair()}, gr.HEADLINE_SCENARIOS["ideal"],
                                     n_folds=3, deficits=[1], classifiers=["pfn"])
    assert len(res) > 0 and set(res["learner"]) == {"in_context"} and np.isfinite(res["pehe"]).all()


def test_off_the_shelf_causalpfn_wrapper_returns_potential_outcomes():
    pytest.importorskip("causalpfn")
    from neurocausalpfn.pfn.estimator import CausalPFNEstimator
    est = CausalPFNEstimator(device="cpu")
    rng = np.random.default_rng(0)
    Xtr, Xte = rng.normal(size=(60, 4)), rng.normal(size=(10, 4))
    W = rng.integers(0, 2, 60).astype(float); Y = (Xtr[:, 0] + W > 0.5).astype(float)
    p1, p0 = est(Xtr, W, Y, Xte)
    assert p1.shape == p0.shape == (10,) and (0 <= p0).all() and (p1 <= 1).all()
