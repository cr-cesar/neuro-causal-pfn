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


def test_pfn_estimator_caps_the_context_at_the_training_length(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from neurocausalpfn.pfn.estimator import PFNEstimator
    from neurocausalpfn.train.train_pfn import prototype_config, run_pfn

    cfg = prototype_config()
    cfg["out_dir"] = str(tmp_path); cfg["pfn"]["iters"] = 2; cfg["pfn"]["batch_size"] = 2
    cfg["pfn"]["d_x"] = 4; cfg["pfn"]["context_max"] = 32; cfg["pfn"]["n_query"] = 4
    cfg["device"] = "cpu"
    run_pfn(cfg)
    ck = os.path.join(str(tmp_path), "pfn.pt")
    rng = np.random.default_rng(1)
    Xtr, Xte = rng.normal(size=(100, 4)).astype(np.float32), rng.normal(size=(9, 4)).astype(np.float32)
    W = (np.arange(100) < 30).astype(float); Y = rng.integers(0, 2, 100).astype(float)

    est = PFNEstimator(ck, device="cpu")                    # default: the checkpoint's context_max
    assert est.context_cap == 32
    Xc, Wc, Yc = est._cap_context(Xtr, W, Y)
    assert len(Xc) == len(Wc) == len(Yc) == 32
    assert int(Wc.sum()) == round(32 * 0.3)                 # treated share preserved
    Xc2, _, _ = est._cap_context(Xtr, W, Y)
    assert np.array_equal(Xc, Xc2)                          # deterministic for the same slice
    before = est.n_capped
    p1, p0 = est(Xtr, W, Y, Xte)
    assert p1.shape == p0.shape == (9,) and est.n_capped == before + 1
    # a slice within the cap is untouched
    assert est._cap_context(Xtr[:20], W[:20], Y[:20])[0].shape == (20, 4)

    monkeypatch.setenv("PFN_CONTEXT_CAP", "0")              # 0 disables the cap
    est0 = PFNEstimator(ck, device="cpu")
    assert est0.context_cap == 0 and est0._cap_context(Xtr, W, Y)[0].shape == (100, 4)
    assert PFNEstimator(ck, device="cpu", context_cap=10)._cap_context(Xtr, W, Y)[0].shape == (10, 4)


def test_off_the_shelf_causalpfn_wrapper_returns_potential_outcomes():
    pytest.importorskip("causalpfn")
    from neurocausalpfn.pfn.estimator import CausalPFNEstimator
    est = CausalPFNEstimator(device="cpu")
    rng = np.random.default_rng(0)
    Xtr, Xte = rng.normal(size=(60, 4)), rng.normal(size=(10, 4))
    W = rng.integers(0, 2, 60).astype(float); Y = (Xtr[:, 0] + W > 0.5).astype(float)
    p1, p0 = est(Xtr, W, Y, Xte)
    assert p1.shape == p0.shape == (10,) and (0 <= p0).all() and (p1 <= 1).all()


def test_causalpfn_wrapper_records_the_feature_compression(monkeypatch, capsys):
    """F08: wider representations are compressed by CausalPFN's TruncatedSVD,
    narrower ones enter raw; the wrapper records and prints which."""
    import sys
    import types

    class _FakeEst:
        def __init__(self, **kw):
            self.max_feature_size = 80
            self.num_neighbours = 1
            self.temperature = 1.0
            self.x_dim_transformer = types.SimpleNamespace(transform=lambda X: X[:, :80])

        def fit(self, X, t, y):
            self.X_train, self.t_train, self.y_train = X, t, y

        def _predict_cepo(self, X, t, y, Xq, tq, temperature=1.0):
            return np.full(len(Xq), 0.5)

    fake = types.ModuleType("causalpfn"); fake.CATEEstimator = _FakeEst
    monkeypatch.setitem(sys.modules, "causalpfn", fake)
    pytest.importorskip("torch")
    from neurocausalpfn.pfn.estimator import CausalPFNEstimator

    est = CausalPFNEstimator(device="cpu")
    rng = np.random.default_rng(0)
    W = np.r_[np.ones(10), np.zeros(10)]; Y = rng.integers(0, 2, 20).astype(float)
    est(rng.normal(size=(20, 50)), W, Y, rng.normal(size=(5, 50)))
    info = dict(est.last_info); v = info.pop("version")
    assert info == {"d_in": 50, "max_feature_size": 80, "truncated_svd": False,
                    "context_cap": 0, "n_context": 20}
    assert isinstance(v, str) and v                          # the installed causalpfn version, or "unknown"
    est(rng.normal(size=(20, 100)), W, Y, rng.normal(size=(5, 100)))
    assert est.last_info["truncated_svd"] is True
    est(rng.normal(size=(20, 100)), W, Y, rng.normal(size=(5, 100)))
    out = capsys.readouterr().out
    assert out.count(f"causalpfn {v}: d_in=50") == 1 and out.count(f"causalpfn {v}: d_in=100") == 1   # once per width


def test_cap_context_is_shared_and_deterministic(monkeypatch):
    from neurocausalpfn.pfn.estimator import cap_context

    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4)); W = (rng.random(300) < 0.3).astype(int); Y = rng.integers(0, 2, 300)
    Xa, Wa, Ya = cap_context(X, W, Y, 100)
    Xb, Wb, Yb = cap_context(X, W, Y, 100)
    assert len(Xa) == 100 and np.array_equal(Xa, Xb) and np.array_equal(Wa, Wb)
    assert abs(Wa.mean() - W.mean()) < 0.02                 # treated share kept
    assert cap_context(X, W, Y, 0)[0] is X and len(cap_context(X, W, Y, 500)[0]) == 300
    # CausalPFN's wrapper reads the same cap from its own variable
    monkeypatch.setenv("CAUSALPFN_CONTEXT_CAP", "1024")
    import inspect
    from neurocausalpfn.pfn import estimator as est
    src = inspect.getsource(est.CausalPFNEstimator.__init__)
    assert "CAUSALPFN_CONTEXT_CAP" in src


def test_causalpfn_wrapper_caps_its_context_like_the_pfn(monkeypatch):
    import sys
    import types

    seen = {}

    class _FakeEst:
        def __init__(self, **kw):
            self.max_feature_size = 80; self.num_neighbours = 1; self.temperature = 1.0

        def fit(self, X, t, y):
            self.X_train, self.t_train, self.y_train = X, t, y; seen["n"] = len(X)

        def _predict_cepo(self, X, t, y, Xq, tq, temperature=1.0):
            return np.full(len(Xq), 0.5)

    fake = types.ModuleType("causalpfn"); fake.CATEEstimator = _FakeEst
    monkeypatch.setitem(sys.modules, "causalpfn", fake)
    pytest.importorskip("torch")
    monkeypatch.setenv("CAUSALPFN_CONTEXT_CAP", "64")
    from neurocausalpfn.pfn.estimator import CausalPFNEstimator, cap_context

    est = CausalPFNEstimator(device="cpu")
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 10)); W = (rng.random(200) < 0.4).astype(float); Y = rng.integers(0, 2, 200).astype(float)
    est(X, W, Y, rng.normal(size=(5, 10)))
    assert seen["n"] == 64 and est.n_capped == 1 and est.last_info["n_context"] == 64
    np.testing.assert_array_equal(est.est.X_train, cap_context(X.astype(np.float32), W, Y, 64)[0])


def test_effect_strata_pick_neighbours_by_estimated_effect_within_the_cap():
    from neurocausalpfn.pfn.estimator import effect_strata

    rng = np.random.default_rng(0)
    e_ctx = rng.uniform(-1, 1, 400); W = (rng.random(400) < 0.5).astype(int)
    e_qry = np.array([-0.9, -0.85, 0.0, 0.9, 0.95])
    strata = list(effect_strata(e_ctx, W, e_qry, k=20, cap=30))
    covered = np.concatenate([q for q, _ in strata])
    assert sorted(covered.tolist()) == list(range(5))             # every query scored once
    for q_idx, c_idx in strata:
        assert len(np.unique(c_idx)) == len(c_idx)
        for w in (0, 1):
            assert (W[c_idx] == w).sum() <= 30                    # per-arm cap
        # the chosen rows are the ones nearest in estimated effect to the stratum's queries
        for q in q_idx:
            for w in (0, 1):
                arm = np.flatnonzero(W == w)
                nearest = arm[np.argsort(np.abs(e_ctx[arm] - e_qry[q]))[:20]]
                assert set(nearest) <= set(c_idx)
    # queries far apart in effect never share a stratum when the cap is tight
    assert len(strata) >= 2


def test_pfn_estimator_effect_context_scores_every_query(tmp_path, monkeypatch):
    """PFN_CONTEXT_SELECT=effect: the context of every query stratum is chosen
    by estimated effect (CausalPFN's rule) and every query is scored once."""
    pytest.importorskip("torch")
    from neurocausalpfn.pfn.estimator import PFNEstimator
    from neurocausalpfn.train.train_pfn import prototype_config, run_pfn

    cfg = prototype_config()
    cfg["out_dir"] = str(tmp_path); cfg["pfn"]["iters"] = 2; cfg["pfn"]["batch_size"] = 2
    cfg["pfn"]["d_x"] = 4; cfg["pfn"]["context_max"] = 32; cfg["pfn"]["n_query"] = 4
    cfg["device"] = "cpu"
    run_pfn(cfg)
    est = PFNEstimator(os.path.join(str(tmp_path), "pfn.pt"), device="cpu", max_queries=5)
    rng = np.random.default_rng(1)
    Xtr, Xte = rng.normal(size=(300, 4)), rng.normal(size=(23, 4))
    W = (rng.random(300) < 0.5).astype(float)
    Y = np.clip(0.5 + 0.3 * Xtr[:, 0] * W + 0.1 * rng.normal(size=300), 0, 1)
    monkeypatch.setenv("PFN_CONTEXT_SELECT", "effect")
    monkeypatch.setenv("PFN_CONTEXT_K", "40")
    monkeypatch.setenv("PFN_CONTEXT_ARM_CAP", "60")
    p1, p0 = est(Xtr, W, Y, Xte)
    assert p1.shape == p0.shape == (23,) and np.isfinite(p1).all() and (0 <= p1).all() and (p1 <= 1).all()
    assert est.n_strata >= 2                                    # the cap split the queries
    monkeypatch.delenv("PFN_CONTEXT_SELECT")
    q1, q0 = est(Xtr, W, Y, Xte)
    assert q1.shape == (23,) and not np.allclose(p1 - p0, q1 - q0)   # a different context, different answers


def test_rotation_ensembling_averages_rotated_predictions(monkeypatch):
    """PFN_ROTATIONS / PFN_ARM_FLIP: the estimator averages _predict over
    rotated copies (and the arm swap, outputs swapped back)."""
    import numpy as np
    from neurocausalpfn.pfn import estimator as E

    class _Est(E.PFNEstimator):
        def __init__(self):
            self.d_x, self.standardize, self.context_cap, self.n_capped = 3, False, 0, 0
            self.calls = []
        def _predict(self, Xtr, W, Y, Xte):
            self.calls.append((Xtr.copy(), np.asarray(W).copy()))
            return np.full(len(Xte), 0.8), np.full(len(Xte), 0.2 + 0.1 * float(np.asarray(W)[0]))

    rng = np.random.default_rng(0)
    Xtr, Xte = rng.normal(size=(10, 3)).astype(np.float32), rng.normal(size=(4, 3)).astype(np.float32)
    W, Y = np.array([1, 0] * 5, dtype=np.float32), np.zeros(10, dtype=np.float32)
    monkeypatch.setenv("PFN_ROTATIONS", "3"); monkeypatch.setenv("PFN_ARM_FLIP", "1")
    e = _Est(); mu1, mu0 = e(Xtr, W, Y, Xte)
    assert len(e.calls) == 8                                        # (identity + 3 rotations) x 2 arm orders
    assert any(np.allclose(c[0], Xtr) for c in e.calls) and any(np.allclose(c[1], 1 - W) for c in e.calls)
    # rotations keep norms
    assert all(np.allclose(np.linalg.norm(c[0], axis=1), np.linalg.norm(Xtr, axis=1), atol=1e-4) for c in e.calls)
    # with the arms swapped, outputs are swapped back: mu1 averages 0.8 and the flipped mu0
    assert np.allclose(mu0, (0.3 + 0.8) / 2) and np.allclose(mu1, (0.8 + 0.2) / 2)
    monkeypatch.delenv("PFN_ROTATIONS"); monkeypatch.delenv("PFN_ARM_FLIP")
    e2 = _Est(); e2(Xtr, W, Y, Xte)
    assert len(e2.calls) == 1
