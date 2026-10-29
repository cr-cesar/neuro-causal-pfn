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

# the fold being scored: the per-fold scorer (ncpfold.score) shares this dict
# and sets it whenever it serves a fold's latents, so a member with one
# checkpoint per fold ({fold} in TABPFN_MODEL_PATH) follows it
CURRENT_FOLD: Dict[str, int] = {"fold": -1}


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
        self.fit_classes = np.unique(np.asarray(y))
        self.n_fits += 1
        if self.name not in REPORTED:
            d = dict(self.info, d_in=int(np.shape(X)[1]), n_fit=int(len(X)))
            REPORTED[self.name] = d
            print(f"{self.name} {d['version']}: model={d['model']} device={d['device']} "
                  f"d_in={d['d_in']} n_fit={d['n_fit']} ({self.fit_seconds:.1f} s)", flush=True)
        return self

    @property
    def classes_(self):
        # FinetunedTabPFNClassifier (route B) has no classes_ of its own: they
        # sit on the inner classifier it predicts with; failing both, the
        # sorted labels of the last fit (the sklearn column order)
        for m in (self.model, getattr(self.model, "finetuned_inference_classifier_", None)):
            c = getattr(m, "classes_", None)
            if c is not None:
                return np.asarray(c)
        return self.fit_classes

    def predict_proba(self, X):
        p = np.asarray(self.model.predict_proba(np.asarray(X, dtype=np.float32)), dtype=float)
        classes = self.classes_
        hits = np.flatnonzero(classes == 1)
        j = int(hits[0]) if len(hits) else p.shape[1] - 1
        out = np.empty((len(p), 2))
        out[:, 1] = np.clip(p[:, j], 0, 1)
        out[:, 0] = 1 - out[:, 1]
        return out


class PerFoldTabPFN(LoggedClassifier):
    """TabPFN with one fine-tuned checkpoint per fold (route A per fold): the
    checkpoint of fold k was tuned on a prior built from a pool without fold
    k's test patients, and scores fold k only. ``{fold}`` in the path is
    filled with ``CURRENT_FOLD`` at every fit; the classifier is rebuilt when
    the fold changes."""

    def __init__(self, pattern: str, kw: dict, info: dict):
        super().__init__("tabpfn", None, info)
        self.pattern, self.kw, self.fold = pattern, dict(kw), None

    def path(self, fold: int) -> str:
        return self.pattern.replace("{fold}", str(fold))

    def fit(self, X, y):
        k = CURRENT_FOLD["fold"]
        if k < 0:
            raise RuntimeError("per-fold TabPFN fitted before the scorer served any fold")
        if k != self.fold:
            from tabpfn import TabPFNClassifier

            path = self.path(k)
            if not os.path.exists(path):
                raise FileNotFoundError(f"no TabPFN checkpoint for fold {k}: {path}")
            self.model, self.fold = TabPFNClassifier(**dict(self.kw, model_path=path)), k
        return super().fit(X, y)


class PerSliceFineTune(LoggedClassifier):
    """Route B: fine-tunes on each slice. FinetunedTabPFNClassifier holds out
    a stratified validation split, impossible when a class has a single case
    (a small two-model arm); such a slice is fitted with the untuned TabPFN
    from the same starting weights instead, and counted in ``n_untuned``."""

    def __init__(self, ft, base, info: dict):
        super().__init__("tabpfn_ft", ft, info)
        self.ft, self.base, self.n_untuned = ft, base, 0

    def fit(self, X, y):
        _, counts = np.unique(np.asarray(y), return_counts=True)
        self.model = self.ft if counts.min() >= 2 else self.base
        if self.model is self.base:
            self.n_untuned += 1
            print(f"tabpfn_ft: slice n={len(y)} has a class with {counts.min()} case(s); "
                  f"untuned TabPFN used ({self.n_untuned} so far)", flush=True)
        return super().fit(X, y)


def make_tabpfn(device: Optional[str] = None, n_estimators="auto", seed: int = 0) -> LoggedClassifier:
    # keep the built model in memory across the replica's refits (off by default)
    os.environ.setdefault("TABPFN_MODEL_CACHE_SIZE", "2")
    from tabpfn import TabPFNClassifier
    from tabpfn.settings import settings

    dev = _device(device)
    kw = dict(device=dev, n_estimators=n_estimators, random_state=seed, ignore_pretraining_limits=True)
    # TABPFN_MODEL_PATH: a checkpoint written by scripts/finetune_tabpfn_prior.py
    # (route A); the member is then the neuro fine-tuned model, scored on the
    # same path as the untuned one; with {fold} in the path, one checkpoint
    # per fold (PerFoldTabPFN)
    model_path = os.environ.get("TABPFN_MODEL_PATH")
    # TABPFN_N_ESTIMATORS: the ensemble size at scoring, so a fine-tuned and
    # an untuned member are compared with the same settings
    if os.environ.get("TABPFN_N_ESTIMATORS", "").strip():
        kw["n_estimators"] = int(os.environ["TABPFN_N_ESTIMATORS"])
    version = settings.tabpfn.model_version
    label = os.path.basename(os.path.dirname(model_path.rstrip("/"))) + "/" + os.path.basename(model_path) \
        if model_path else getattr(version, "value", str(version))
    info = {"version": _version("tabpfn"), "model": label, "device": dev, "n_estimators": kw["n_estimators"]}
    if model_path and "{fold}" in model_path:
        return PerFoldTabPFN(model_path, kw, info)
    if model_path:
        kw["model_path"] = model_path
    return LoggedClassifier("tabpfn", TabPFNClassifier(**kw), info)


def make_tabpfn_ft(device: Optional[str] = None, n_estimators=None, seed: int = 0) -> LoggedClassifier:
    """Route B: TabPFN fine-tuned briefly on each slice's own training
    patients before predicting (Prior Labs' per-dataset fine-tuning; arXiv
    2506.08982). Every ``fit`` starts from the base weights again, so slices
    never share information. Knobs: TABPFN_FT_EPOCHS (default 5),
    TABPFN_FT_TIME_LIMIT seconds (default 60), TABPFN_FT_LR (default 1e-5);
    TABPFN_MODEL_PATH sets the starting checkpoint. A slice too small for the
    stratified validation split gets the untuned model (PerSliceFineTune)."""
    from tabpfn import TabPFNClassifier
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
    # the fallback predicts like the fine-tuned one's final model, untuned
    base = TabPFNClassifier(device=dev, n_estimators=4, random_state=seed, ignore_pretraining_limits=True,
                            **extra)
    return PerSliceFineTune(clf, base, {"version": _version("tabpfn"),
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
    # a per-fold member warms up on fold 0's checkpoint, so a missing file
    # fails now; the scorer sets the real fold before the first real fit
    before = CURRENT_FOLD["fold"]
    if isinstance(clf, PerFoldTabPFN) and before < 0:
        CURRENT_FOLD["fold"] = 0
    try:
        p = clf.fit(X, y).predict_proba(X[:8])
    finally:
        CURRENT_FOLD["fold"] = before
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
