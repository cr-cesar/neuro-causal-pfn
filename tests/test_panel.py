"""The tabular foundation-model panel as replica classifiers (19 Oct)."""
import sys
import types

import numpy as np
import pytest

from neurocausalpfn.pfn import panel
from neurocausalpfn.prior import giles_replica as gr


class _FakeModel:
    """sklearn-style classifier whose classes_ are in reverse order, to check
    the probability column is picked by label and not by position."""
    calls = 0

    def __init__(self, **kw):
        self.kw = kw

    def fit(self, X, y):
        _FakeModel.calls += 1
        self.classes_ = np.array([1, 0])
        self.m = float(np.mean(y))
        return self

    def predict_proba(self, X):
        p1 = np.full(len(X), self.m)
        return np.stack([p1, 1 - p1], axis=1)       # column 0 is class 1 here


@pytest.fixture
def fake_packages(monkeypatch):
    tp = types.ModuleType("tabpfn"); tp.__version__ = "9.9"
    tp.TabPFNClassifier = _FakeModel
    tps = types.ModuleType("tabpfn.settings")
    tps.settings = types.SimpleNamespace(tabpfn=types.SimpleNamespace(model_version="v3.5"))
    ti = types.ModuleType("tabicl")

    class _FakeICL(_FakeModel):
        checkpoint_version = "tabicl-test.ckpt"

    ti.TabICLClassifier = _FakeICL
    tpf = types.ModuleType("tabpfn.finetuning")

    class _FakeFT:
        """like tabpfn's FinetunedTabPFNClassifier: no classes_ of its own,
        predictions and classes on finetuned_inference_classifier_"""
        def __init__(self, **kw):
            self.kw = kw

        def fit(self, X, y):
            self.finetuned_inference_classifier_ = _FakeModel().fit(X, y)
            return self

        def predict_proba(self, X):
            return self.finetuned_inference_classifier_.predict_proba(X)

    tpf.FinetunedTabPFNClassifier = _FakeFT
    monkeypatch.setitem(sys.modules, "tabpfn.finetuning", tpf)
    monkeypatch.setitem(sys.modules, "tabpfn", tp)
    monkeypatch.setitem(sys.modules, "tabpfn.settings", tps)
    monkeypatch.setitem(sys.modules, "tabicl", ti)
    monkeypatch.setattr(panel, "_version", lambda dist: "9.9")
    monkeypatch.setattr(panel, "REPORTED", {})
    monkeypatch.setattr(gr, "CLASSIFIERS", {})
    _FakeModel.calls = 0
    yield


def test_register_panel_makes_replica_classifiers_with_class1_column(fake_packages, capsys):
    made = panel.register_panel(["tabpfn", "tabicl"], device="cpu", warm_up=True)
    assert set(made) == {"tabpfn", "tabicl"} and set(gr.CLASSIFIERS) == {"tabpfn", "tabicl"}
    assert made["tabpfn"].info["model"] == "v3.5" and made["tabicl"].info["model"] == "tabicl-test.ckpt"
    assert "warm-up fit ok" in capsys.readouterr().out
    # the factory hands back the same instance; the replica refits it in turn
    assert gr._classifier("tabpfn") is made["tabpfn"]
    X = np.zeros((6, 3)); y = np.array([1, 1, 1, 0, 0, 0])
    p = gr._classifier("tabpfn").fit(X, y).predict_proba(X)
    assert p.shape == (6, 2) and np.allclose(p[:, 1], 0.5) and np.allclose(p.sum(1), 1)
    out = capsys.readouterr().out
    assert "tabpfn 9.9: model=v3.5 device=cpu d_in=3 n_fit=6" in out
    gr._classifier("tabpfn").fit(X, y)
    assert "tabpfn 9.9" not in capsys.readouterr().out   # reported once
    with pytest.raises(ValueError):
        panel.register_panel(["lightgbm"])
    with pytest.raises(ValueError):
        gr.register_estimator("tabpfn", lambda *a: None)


def test_panel_member_scores_inside_the_replica_like_a_built_in(fake_packages):
    from sklearn.linear_model import LogisticRegression

    class _LR(LogisticRegression):
        pass

    gr.register_classifier("lr_panel", lambda: _LR(max_iter=200))
    rng = np.random.default_rng(0)
    Xtr, Xte = rng.normal(size=(60, 3)), rng.normal(size=(10, 3))
    W = (rng.uniform(size=60) < 0.5).astype(int)
    Y = (rng.uniform(size=60) < 0.5).astype(int)
    p1, p0 = gr._fit_predict_two_model(Xtr, W, Y, Xte, "lr_panel")
    assert p1.shape == (10,) and p0.shape == (10,)
    p1, p0 = gr._fit_predict_one_model(Xtr, W, Y, Xte, "lr_panel")
    assert p1.shape == (10,) and np.all((0 <= p1) & (p1 <= 1))
    with pytest.raises(ValueError):
        gr._classifier("nope")


def test_fine_tuned_members(fake_packages, monkeypatch, tmp_path):
    """TABPFN_MODEL_PATH makes the tabpfn member load a fine-tuned checkpoint
    (route A); tabpfn_ft fine-tunes per slice with the env knobs (route B)."""
    ck = tmp_path / "kch_gen05" / "tabpfn_ft.pth"; ck.parent.mkdir(); ck.write_text("x")
    monkeypatch.setenv("TABPFN_MODEL_PATH", str(ck))
    monkeypatch.setenv("TABPFN_FT_EPOCHS", "3"); monkeypatch.setenv("TABPFN_FT_TIME_LIMIT", "30")
    made = panel.register_panel(["tabpfn", "tabpfn_ft"], device="cpu", warm_up=True)
    assert made["tabpfn"].model.kw["model_path"] == str(ck) and made["tabpfn"].info["model"] == "kch_gen05/tabpfn_ft.pth"
    ft = made["tabpfn_ft"].model
    assert ft.kw["epochs"] == 3 and ft.kw["time_limit"] == 30 and ft.kw["extra_classifier_kwargs"] == {"model_path": str(ck)}
    assert "per-slice fine-tune, 3 epochs, 30 s" == made["tabpfn_ft"].info["model"]
    # the class-1 column is found on the inner classifier (reversed order there)
    X, y = np.zeros((5, 2)), np.array([1, 1, 1, 0, 0])
    assert np.allclose(made["tabpfn_ft"].fit(X, y).predict_proba(X)[:, 1], 0.6)
    assert made["tabpfn_ft"].model is ft and made["tabpfn_ft"].n_untuned == 0
    # a class with one case cannot be split for validation: untuned TabPFN,
    # same starting checkpoint, counted
    member = made["tabpfn_ft"]
    p = member.fit(X, np.array([0, 0, 0, 0, 1])).predict_proba(X)
    assert member.model is member.base and member.n_untuned == 1 and np.allclose(p[:, 1], 0.2)
    assert member.base.kw["model_path"] == str(ck)
    member.fit(X, y)
    assert member.model is ft and member.n_untuned == 1
    monkeypatch.delenv("TABPFN_MODEL_PATH")
    plain = panel.make_tabpfn(device="cpu")
    assert "model_path" not in plain.model.kw and plain.info["model"] == "v3.5"
    monkeypatch.setenv("TABPFN_N_ESTIMATORS", "4")
    same = panel.make_tabpfn(device="cpu")
    assert same.model.kw["n_estimators"] == 4 and same.info["n_estimators"] == 4


def test_per_fold_tabpfn_follows_the_scorer_fold(fake_packages, monkeypatch, tmp_path):
    """{fold} in TABPFN_MODEL_PATH: the member loads fold k's checkpoint when
    the scorer is on fold k, warms up on fold 0's, and refuses a missing one."""
    for k in (0, 1):
        (tmp_path / f"fold{k}").mkdir()
        (tmp_path / f"fold{k}" / "tabpfn_ft.pth").write_bytes(b"")
    monkeypatch.setenv("TABPFN_MODEL_PATH", str(tmp_path / "fold{fold}" / "tabpfn_ft.pth"))
    monkeypatch.setitem(panel.CURRENT_FOLD, "fold", -1)
    clf = panel.make_tabpfn(device="cpu")
    assert isinstance(clf, panel.PerFoldTabPFN)
    panel.warm_up_fit(clf)
    assert "fold0" in clf.model.kw["model_path"]
    assert panel.CURRENT_FOLD["fold"] == -1
    X, y = np.zeros((10, 3)), np.array([0, 1] * 5)
    panel.CURRENT_FOLD["fold"] = 1
    clf.fit(X, y)
    assert "fold1" in clf.model.kw["model_path"]
    panel.CURRENT_FOLD["fold"] = 2
    with pytest.raises(FileNotFoundError):
        clf.fit(X, y)
    panel.CURRENT_FOLD["fold"] = -1
    with pytest.raises(RuntimeError):
        clf.fit(X, y)


def test_classes_fall_back_to_the_fit_labels():
    """a wrapped model with no classes_ anywhere: the sorted labels of the fit"""
    class _Bare:
        def fit(self, X, y):
            self.m = float(np.mean(y)); return self

        def predict_proba(self, X):
            return np.stack([np.full(len(X), 1 - self.m), np.full(len(X), self.m)], axis=1)

    clf = panel.LoggedClassifier("bare", _Bare(), {"version": "0", "model": "m", "device": "cpu"})
    X, y = np.zeros((4, 2)), np.array([1, 0, 0, 0])
    p = clf.fit(X, y).predict_proba(X)
    assert list(clf.classes_) == [0, 1] and np.allclose(p[:, 1], 0.25)


def test_route_b_convergence_knobs(fake_packages, monkeypatch):
    """a converging fine-tune: many epochs, no time limit, real patience,
    log-loss as the validation metric (defined with one class)"""
    for k, v in {"TABPFN_FT_EPOCHS": "50", "TABPFN_FT_TIME_LIMIT": "0", "TABPFN_FT_LR": "1e-4",
                 "TABPFN_FT_PATIENCE": "5", "TABPFN_FT_METRIC": "log_loss", "TABPFN_FT_LOG": "1"}.items():
        monkeypatch.setenv(k, v)
    m = panel.make_tabpfn_ft(device="cpu")
    kw = m.ft.kw
    assert kw["epochs"] == 50 and kw["time_limit"] is None and kw["learning_rate"] == 1e-4
    assert kw["early_stopping_patience"] == 5 and kw["eval_metric"] == "log_loss"
    assert m.info["model"] == "per-slice fine-tune, 50 epochs, 0 s, lr 0.0001, patience 5, log_loss"
    import logging
    assert logging.getLogger("tabpfn").level == logging.INFO
