"""Tabular foundation models as the replica's classifiers (panel of 19 Oct).

The Phase-1 classifiers are a logistic regression and an ExtraTrees. The
panel puts two pretrained tabular foundation models in the very same role:

- ``tabpfn``: Prior Labs' TabPFN (``pip install tabpfn``; the package picks
  v3.5 unless ``TABPFN_MODEL_VERSION`` says otherwise);
- ``tabicl``: TabICL (``pip install tabicl``).

Each is an sklearn-style classifier fitted on the train slice and asked for
``predict_proba`` on the test slice, one-model and two-model, on the same
folds, simulated trials and PEHE scale as the built-ins, so the gap between
an off-the-shelf tabular model and the trained Neuro-Causal-PFN is read on
one scale. One instance per name is kept and refitted slice after slice (the
replica fits and predicts in turn, never two models at once); TabPFN's
built-model cache keeps the weights in memory across refits. The first fit
of each member prints the package version, model file, device and input
width, so every scoring log states what ran.

The packages need torch >= 2.5, a separate venv on Myriad
(``scripts/panel_venv_myriad.sh``). TabPFN v2.5+ weights sit behind a
one-time licence acceptance (``TABPFN_TOKEN`` from https://ux.priorlabs.ai
and a Hugging Face token with access to the Prior-Labs repository); TabICL's
download freely. Download both once on a login node::

    python -c "from neurocausalpfn.pfn.panel import register_panel; register_panel(['tabpfn', 'tabicl'], warm_up=True)"
"""
import os
import time
from typing import Dict, Iterable, Optional

import numpy as np

REPORTED: Dict[str, dict] = {}


def _device(device: Optional[str]) -> str:
    if device:
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _version(dist: str) -> str:
    try:
        from importlib.metadata import version

        return version(dist)
    except Exception:                                   # not installed as a distribution
        return "unknown"


class LoggedClassifier:
    """``fit`` / ``predict_proba`` of the wrapped classifier; the first fit
    prints one line with version, model, device and input width (kept in
    ``REPORTED``). ``predict_proba`` returns two columns, the second one the
    probability of outcome 1 whatever the wrapped model's class order."""

    def __init__(self, name: str, model, info: dict):
        self.name, self.model, self.info = name, model, dict(info)
        self.n_fits = 0
        self.fit_seconds = 0.0

    def fit(self, X, y):
        t = time.time()
        self.model.fit(np.asarray(X, dtype=np.float32), np.asarray(y))
        self.fit_seconds += time.time() - t
        self.n_fits += 1
        if self.name not in REPORTED:
            d = dict(self.info, d_in=int(np.shape(X)[1]), n_fit=int(len(X)))
            REPORTED[self.name] = d
            print(f"{self.name} {d['version']}: model={d['model']} device={d['device']} "
                  f"d_in={d['d_in']} n_fit={d['n_fit']} ({self.fit_seconds:.1f} s)", flush=True)
        return self

    @property
    def classes_(self):
        return np.asarray(self.model.classes_)

    def predict_proba(self, X):
        p = np.asarray(self.model.predict_proba(np.asarray(X, dtype=np.float32)), dtype=float)
        classes = self.classes_
        hits = np.flatnonzero(classes == 1)
        j = int(hits[0]) if len(hits) else p.shape[1] - 1
        out = np.empty((len(p), 2))
        out[:, 1] = np.clip(p[:, j], 0, 1)
        out[:, 0] = 1 - out[:, 1]
        return out


def make_tabpfn(device: Optional[str] = None, n_estimators="auto", seed: int = 0) -> LoggedClassifier:
    # keep the built model in memory across the replica's refits (off by default)
    os.environ.setdefault("TABPFN_MODEL_CACHE_SIZE", "2")
    from tabpfn import TabPFNClassifier
    from tabpfn.settings import settings

    dev = _device(device)
    kw = dict(device=dev, n_estimators=n_estimators, random_state=seed, ignore_pretraining_limits=True)
    # TABPFN_MODEL_PATH: a checkpoint written by scripts/finetune_tabpfn_prior.py
    # (route A); the member is then the neuro fine-tuned model, scored on the
    # same path as the untuned one
    model_path = os.environ.get("TABPFN_MODEL_PATH")
    if model_path:
        kw["model_path"] = model_path
    # TABPFN_N_ESTIMATORS: the ensemble size at scoring, so a fine-tuned and
    # an untuned member are compared with the same settings
    if os.environ.get("TABPFN_N_ESTIMATORS", "").strip():
        kw["n_estimators"] = int(os.environ["TABPFN_N_ESTIMATORS"])
    clf = TabPFNClassifier(**kw)
    version = settings.tabpfn.model_version
    label = os.path.basename(os.path.dirname(model_path.rstrip("/"))) + "/" + os.path.basename(model_path) \
        if model_path else getattr(version, "value", str(version))
    return LoggedClassifier("tabpfn", clf, {"version": _version("tabpfn"), "model": label, "device": dev,
                                            "n_estimators": kw["n_estimators"]})


def make_tabpfn_ft(device: Optional[str] = None, n_estimators=None, seed: int = 0) -> LoggedClassifier:
    """Route B: TabPFN fine-tuned briefly on each slice's own training
    patients before predicting (Prior Labs' per-dataset fine-tuning; arXiv
    2506.08982). Every ``fit`` starts from the base weights again, so slices
    never share information. Knobs: TABPFN_FT_EPOCHS (default 5),
    TABPFN_FT_TIME_LIMIT seconds (default 60), TABPFN_FT_LR (default 1e-5);
    TABPFN_MODEL_PATH sets the starting checkpoint."""
    from tabpfn.finetuning import FinetunedTabPFNClassifier

    dev = _device(device)
    epochs = int(os.environ.get("TABPFN_FT_EPOCHS", "5"))
    time_limit = int(os.environ.get("TABPFN_FT_TIME_LIMIT", "60"))
    lr = float(os.environ.get("TABPFN_FT_LR", "1e-5"))
    extra = {}
    if os.environ.get("TABPFN_MODEL_PATH"):
        extra["model_path"] = os.environ["TABPFN_MODEL_PATH"]
    clf = FinetunedTabPFNClassifier(device=dev, epochs=epochs, time_limit=time_limit, learning_rate=lr,
                                    validation_split_ratio=0.1, early_stopping=True, random_state=seed,
                                    n_estimators_finetune=2, n_estimators_validation=2,
                                    n_estimators_final_inference=4, save_checkpoint_interval=None,
                                    extra_classifier_kwargs=extra or None)
    return LoggedClassifier("tabpfn_ft", clf, {"version": _version("tabpfn"),
                                               "model": f"per-slice fine-tune, {epochs} epochs, {time_limit} s",
                                               "device": dev})


def make_tabicl(device: Optional[str] = None, n_estimators: int = 8, seed: int = 0) -> LoggedClassifier:
    from tabicl import TabICLClassifier

    dev = _device(device)
    clf = TabICLClassifier(device=dev, n_estimators=n_estimators, random_state=seed)
    return LoggedClassifier("tabicl", clf, {"version": _version("tabicl"),
                                            "model": str(clf.checkpoint_version), "device": dev})


PANEL = {"tabpfn": make_tabpfn, "tabicl": make_tabicl, "tabpfn_ft": make_tabpfn_ft}


def warm_up_fit(clf: LoggedClassifier, d: int = 4, n: int = 40, seed: int = 0) -> float:
    """A toy fit and prediction: downloads or loads the weights now, so a
    missing package, licence or weight file fails before a scoring job
    starts. Returns the seconds taken."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = (X[:, 0] + 0.5 * rng.normal(size=n) > 0).astype(int)
    t = time.time()
    p = clf.fit(X, y).predict_proba(X[:8])
    if p.shape != (8, 2) or not np.all(np.isfinite(p)):
        raise RuntimeError(f"{clf.name}: warm-up prediction has shape {p.shape}")
    REPORTED.pop(clf.name, None)                  # the real first fit reports the real width
    return time.time() - t


def register_panel(names: Iterable[str], device: Optional[str] = None, warm_up: bool = False,
                   seed: int = 0) -> Dict[str, LoggedClassifier]:
    """Register the named panel members as replica classifiers (one instance
    each, returned by the factory on every ``_classifier`` call). With
    ``warm_up`` each member is fitted once on toy data first."""
    from neurocausalpfn.prior import giles_replica as gr

    made = {}
    for name in names:
        if name not in PANEL:
            raise ValueError(f"unknown panel member {name!r}; choose from {sorted(PANEL)}")
        inst = PANEL[name](device=device, seed=seed)
        if warm_up:
            dt = warm_up_fit(inst)
            print(f"{name}: warm-up fit ok ({dt:.1f} s, {inst.info['model']}, {inst.info['device']})", flush=True)
        gr.register_classifier(name, (lambda inst=inst: inst))
        made[name] = inst
    return made
