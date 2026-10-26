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

    class _FakeFT(_FakeModel):
        def __init__(self, **kw):
            super().__init__(**{k: v for k, v in kw.items() if k in ("device", "random_state")})
            self.kw = kw

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
    monkeypatch.delenv("TABPFN_MODEL_PATH")
    plain = panel.make_tabpfn(device="cpu")
    assert "model_path" not in plain.model.kw and plain.info["model"] == "v3.5"
