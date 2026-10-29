"""Faithful replica of the Giles et al. (2025) virtual-trial evaluation.

Ports the ground-truth simulation and the PEHE scale of
``high-dimensional/individualized_prescriptive_inference`` (Nature
Communications 16:8968; code CC BY-NC-SA 4.0, adapted here with citation) so
that any lesion representation — theirs or ours — can be scored on the SAME
scale as the published headline (disconnectome VAE-50: PEHE 0.349, 95% CI
[0.33, 0.37], random allocation, averaged over the 16 functional networks,
classifier not stated in the main text; Supplementary Table 8g, "ideal
experimental conditions", gives 0.305 for VAE-50 with logistic regression
two-model, i.e. the ideal cell, not the confounded one. Under location bias
the paper publishes no single cell: Supplementary Table 5 gives grid means
over TE x RE x bias, and Supplementary Figs. 77-78 the heatmaps).

The mechanism, exactly as in their code:

- Each of the 16 functional networks has two subnetworks (labels 1 and 2 in
  the parcellation nifti). A lesion is "susceptible" to the treatment mapped
  to a subnetwork when it covers > 5% of that subnetwork's volume (binarised
  at 0.5 for disconnectomes). The LARGER subnetwork maps to treatment 1.
- Per network, the true label y_true is 1 (W1-susceptible), 0
  (W0-susceptible) or 0.5 (both) — this y_true IS the ground-truth ITE.
- A virtual trial assigns treatment W with allocation bias BIAS in [0, 1]:
  'observed' bias sorts patients along the spatial axis that best separates
  the two subnetwork centroids; 'unobserved' bias allocates towards y_true
  directly. Treated-susceptible patients respond with probability TE
  (population fraction), and an RE fraction of everyone responds
  spontaneously. Seeds are deterministic: seed = fold * 10000, incrementing.
- Estimation fits classifiers (one-model / two-model, a.k.a. S-/T-learner)
  on the train fold and predicts both counterfactuals on the test fold;
  pred_ITE = sigmoid(p1 - p0); PEHE = RMSE(pred_ITE, y_true_test).

Folds. ``image_folds`` splits by image (the original evaluation; two images of
one patient can straddle the split). ``group_folds`` splits by patient with
the lesion->patient table received from Giles (``load_group_table``): only
the earliest image of each patient is tested, and the ``paper`` table marks
his single-acquisition patients, which gives his test set (2,828 patients).

Encoders. A frozen encoder trained on the full lesion set has seen the test
images (anatomical leakage; outcomes are simulated after the fact, so there is
no outcome leakage). The per-fold study (neuro-causal-pfn-perfold) retrains
the encoder inside each fold and removes it. Estimator fitting is always
train-fold-only.

Headline selection. ``headline_aggregate`` picks the classifier x learner cell
by the released-code PEHE (sigmoid scale, column ``pehe``) and reports that
cell on every scale; the paper-scale value of the cell chosen on the paper
scale is reported alongside as ``pehe_paper_selected``.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFICIT_NAMES = ['hearing', 'language', 'introspection', 'cognition', 'mood',
                 'memory', 'aversion', 'coordination', 'interoception', 'sleep',
                 'reward', 'visual recognition', 'visual perception',
                 'spatial reasoning', 'motor', 'somatosensory']

ROI_THRESH = 0.05        # lesion must cover >5% of the subnetwork volume
DISCO_THRESH = 0.5       # binarisation threshold for continuous inputs

# The headline cells of the paper (Figure 6 text): "ideal" is large treatment
# effect, infrequent spontaneous recovery, random allocation; "location_bias"
# is the strongest observed (centroid-axis) allocation bias.
HEADLINE_SCENARIOS = {
    "ideal": {"TE": 1.0, "RE": 0.0, "BIAS": 0.0, "BIASTYPE": "observed"},
    "location_bias": {"TE": 1.0, "RE": 0.0, "BIAS": 1.0, "BIASTYPE": "observed"},
}
# the location-bias sweep (F30): BIAS = 1 is deterministic allocation with
# positivity violated, the worst case; the intermediate cells give the curve
for _b in (0.25, 0.5, 0.75, 0.95):
    HEADLINE_SCENARIOS[f"location_bias_{int(round(_b * 100)):03d}"] = {
        "TE": 1.0, "RE": 0.0, "BIAS": float(_b), "BIASTYPE": "observed"}
del _b
# the paper's noisy cells (27 Oct): lower treatment efficacy with spontaneous
# recovery, TE + RE = 1 as in its grid, randomised allocation. The true effect
# stays y_true in {0, 0.5, 1} (tau in {-1, 0, +1}) in the paper's code and
# here: these cells measure robustness to noisy outcomes, not calibration to
# the smaller identifiable effect TE * (1 - RE).
for _te in (0.75, 0.5, 0.25):
    HEADLINE_SCENARIOS[f"noisy_te{int(round(_te * 100)):03d}"] = {
        "TE": float(_te), "RE": float(round(1.0 - _te, 2)), "BIAS": 0.0, "BIASTYPE": "observed"}
del _te


# --------------------------------------------------------------------------- #
# Ground truth: atlas pairs, susceptibility labels, per-lesion features
# --------------------------------------------------------------------------- #
@dataclass
class RoiPair:
    masks: List[np.ndarray]          # [roi1, roi2] binary masks
    vols: List[int]
    treatments: List[str]            # "1"/"0" per mask; larger subnetwork -> "1"
    centroids: Tuple[List[float], List[float], List[float]]   # x, y, z per mask


def load_atlas_pairs(atlas_dir: str, modality: str = "receptor") -> Dict[int, RoiPair]:
    """The 16 subnetwork pairs of ``atlas_dir/2mm_parcellations/<modality>``,
    with volumes, treatment mapping and centroids, as in deficit_modelling."""
    import nibabel as nib

    roipath = os.path.join(atlas_dir, "2mm_parcellations", modality)
    pairs: Dict[int, RoiPair] = {}
    files = os.listdir(roipath)
    for idx in range(1, 17):
        match = [f for f in files if f.startswith(f"{idx}_") and "nii" in f]
        if not match:
            raise FileNotFoundError(f"no parcellation file for pair {idx} in {roipath}")
        arr = nib.load(os.path.join(roipath, match[0])).get_fdata()
        masks = [(arr == 1).astype(int), (arr == 2).astype(int)]
        vols = [int(m.sum()) for m in masks]
        treatments = ["1", "0"] if int(np.argmax(vols)) == 0 else ["0", "1"]
        cx, cy, cz = [], [], []
        for m in masks:
            inds = np.where(m == 1)
            cx.append(float(inds[0].mean()))
            cy.append(float(inds[1].mean()))
            cz.append(float(inds[2].mean()))
        pairs[idx] = RoiPair(masks, vols, treatments, (cx, cy, cz))
    return pairs


def label_images(files: Sequence[str], pairs: Dict[int, RoiPair],
                 roi_thresh: float = ROI_THRESH,
                 disco_thresh: float = DISCO_THRESH,
                 progress: bool = True):
    """Susceptibility labels and spatial features for a list of nifti files.

    Returns a pandas DataFrame with one row per file: filename, vol,
    centroid_x/y/z (of the binarised image) and, per deficit,
    ``{name}_W0`` / ``{name}_W1`` hits — the exact columns the Giles trial
    simulator consumes.
    """
    import nibabel as nib
    import pandas as pd

    rows = []
    iterator = files
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(files, desc="labelling lesions")
        except ImportError:
            pass
    for path in iterator:
        img = nib.load(path).get_fdata()
        binary = (img > disco_thresh).astype(int)
        inds = np.where(binary == 1)
        row = {"filename": os.path.basename(path),
               "vol": int(binary.sum()),
               "centroid_x": float(inds[0].mean()) if binary.sum() else 0.0,
               "centroid_y": float(inds[1].mean()) if binary.sum() else 0.0,
               "centroid_z": float(inds[2].mean()) if binary.sum() else 0.0}
        for idx, pair in pairs.items():
            name = DEFICIT_NAMES[idx - 1]
            for mask, vol, treatment in zip(pair.masks, pair.vols, pair.treatments):
                thresh = int(np.round(vol * roi_thresh))
                row[f"{name}_W{treatment}"] = 1 if int((binary * mask).sum()) > thresh else 0
        rows.append(row)
    return pd.DataFrame(rows)


def deficit_slice(df, deficit_idx: int):
    """The sub-cohort susceptible to network ``deficit_idx`` (1..16), with the
    ground-truth y_true in {0, 0.5, 1}; None when the slice is empty."""
    name = DEFICIT_NAMES[deficit_idx - 1]
    sl = df.loc[(df[f"{name}_W0"] == 1) | (df[f"{name}_W1"] == 1)].reset_index(drop=True).copy()
    if not len(sl):
        return None
    sl.loc[sl[f"{name}_W1"] == 1, "y_true"] = 1.0
    sl.loc[sl[f"{name}_W0"] == 1, "y_true"] = 0.0
    sl.loc[(sl[f"{name}_W0"] == 1) & (sl[f"{name}_W1"] == 1), "y_true"] = 0.5
    return sl


# --------------------------------------------------------------------------- #
# The virtual trial (port of ground_truth_simulation)
# --------------------------------------------------------------------------- #
class GilesTrial:
    """Port of ``ground_truth_simulation``: allocation bias + treatment and
    recovery effects, with the original deterministic seeding."""

    def __init__(self, TE: float, RE: float, BIAS: float, BIASTYPE: str, k: int):
        self.TE, self.RE, self.BIAS, self.BIASTYPE = TE, RE, BIAS, BIASTYPE
        self.seed = k * 10000

    def _bias_by_axis(self, train, axis: str) -> list:
        train = train.sort_values(by=f"centroid_{axis}")
        inds = train.index
        n = len(train)
        if self.BIAS == 0:
            np.random.seed(self.seed); self.seed += 1
            sel = np.random.choice([0, 1], n)
            return [inds[i] for i in range(n) if sel[i] == 1]
        if self.BIAS <= 0.5:
            probs = np.linspace(0.5 - self.BIAS, 0.5 + self.BIAS, n)
            alloc = []
            for i in range(n):
                np.random.seed(self.seed); self.seed += 1
                if np.random.choice([0, 1], p=[1 - probs[i], probs[i]]) == 1:
                    alloc.append(inds[i])
            return alloc
        # BIAS > 0.5: a fixed deterministic block plus graded probabilities
        alloc = []
        median = int(np.round(n / 2))
        lowhalf, uphalf = train.iloc[:median], train.iloc[median:]
        fix = 2 * self.BIAS - 1
        alloc.append(list(uphalf.iloc[int(np.round(len(uphalf) * (1 - fix))):].index))
        up_iter = uphalf.iloc[:int(np.round(len(uphalf) * (1 - fix)))]
        probs = np.linspace(0.5, 1, len(up_iter))
        for i, ind in enumerate(up_iter.index):
            np.random.seed(self.seed); self.seed += 1
            if np.random.choice([0, 1], p=[1 - probs[i], probs[i]]) == 1:
                alloc.append([ind])
        low_iter = lowhalf.iloc[-1:] if fix == 1 else lowhalf.iloc[-int(np.round(len(lowhalf) * (1 - fix))):]
        probs = np.linspace(0, 0.5, len(low_iter))
        for i, ind in enumerate(low_iter.index):
            np.random.seed(self.seed); self.seed += 1
            if np.random.choice([0, 1], p=[1 - probs[i], probs[i]]) == 1:
                alloc.append([ind])
        return [x for sub in alloc for x in (sub if isinstance(sub, list) else [sub])]

    def _bias_agnostic(self, train) -> list:
        inds = train.index
        towards = train["y_true"].to_numpy()
        if self.BIAS == 0:
            np.random.seed(self.seed); self.seed += 1
            sel = np.random.choice([0, 1], len(train))
            return [inds[i] for i in range(len(train)) if sel[i] == 1]
        alloc = []
        for i in range(len(train)):
            p1 = self.BIAS if towards[i] == 1 else (1 - self.BIAS if towards[i] == 0 else 0.5)
            np.random.seed(self.seed); self.seed += 1
            if np.random.choice([0, 1], 1, p=[1 - p1, p1]):
                alloc.append(inds[i])
        return alloc

    def simulate(self, train, roi_centroids) -> Tuple[np.ndarray, np.ndarray]:
        """W, Y for the train slice, mutating a copy (port of simulate_trial)."""
        train = train.reset_index(drop=True).copy()
        if self.BIASTYPE == "unobserved":
            train["group"] = 0
            train.loc[self._bias_agnostic(train), "group"] = 1
        else:
            cx, cy, cz = roi_centroids
            span = [abs(cx[0] - cx[1]), abs(cy[0] - cy[1]), abs(cz[0] - cz[1])]
            axis = "xyz"[int(np.argmax(span))]
            centres = {"x": cx, "y": cy, "z": cz}[axis]
            alloc = self._bias_by_axis(train, axis)
            train["group"] = 1 - int(np.argmax(centres))
            train.loc[alloc, "group"] = int(np.argmax(centres))

        susceptible = train.loc[(train["group"] == train["y_true"]) | (train["y_true"] == 0.5)]
        np.random.seed(self.seed); self.seed += 1
        respond = np.random.choice(susceptible.index,
                                   int(np.round(len(susceptible) * self.TE)), replace=False)
        train["respond"] = 0
        train.loc[respond, "respond"] = 1
        if self.RE != 0:
            np.random.seed(self.seed); self.seed += 1
            spont = np.random.choice(train.index, int(round(len(train) * self.RE)), replace=False)
            train.loc[spont, "respond"] = 1
        return np.array(train["group"]), np.array(train["respond"])


# --------------------------------------------------------------------------- #
# The PEHE scale (port of estimate_PEHE / process_results)
# --------------------------------------------------------------------------- #
def pehe(pred_ITE: np.ndarray, true_ITE: np.ndarray) -> float:
    return float((((pred_ITE - true_ITE) ** 2).sum() / len(true_ITE)) ** 0.5)


def score_predictions(p1: np.ndarray, p0: np.ndarray, true_ITE: np.ndarray) -> Dict[str, float]:
    """PEHE and prescriptive balanced accuracy for one representation.

    Three PEHE variants are reported; ``pehe_paper`` is the headline.

    - ``pehe_paper`` (headline; the paper's Methods definition): tau_hat is the
      RAW probability difference p1 - p0 in [-1, 1], scored against
      tau in {-1, 0, +1} (both-susceptible = 0) over ALL participants, i.e.
      sqrt(mean((tau_hat - tau)^2)). This is the convention that places the
      six published anchors in reachable neighbourhoods and is the one the
      leaderboard and every selection decision use.
    - ``pehe`` (released-code convention, diagnostic): sigmoid(p1 - p0) against
      y_true in {0, 0.5, 1}; the sigmoid imposes a floor of ~0.27, so it is
      kept for comparison with the released scripts, not for ranking.
    - ``pehe_xor`` (diagnostic): the released-code scale restricted to the
      decidable cases (y_true != 0.5). It excludes the tau = 0 participants
      that the paper explicitly keeps, so it is NOT the published headline;
      it survives from an early calibration pass and is retained only as a
      diagnostic. Balanced accuracy is, as in the paper, decidable-only."""
    pred_ITE = 1.0 / (1.0 + np.exp(-(p1 - p0)))
    out = {"pehe": pehe(pred_ITE, true_ITE)}
    # The paper's own convention (Methods, "Observed PEHE" + Problem setup):
    # tau_hat = P(Y=1|A,x) - P(Y=1|B,x), the RAW probability difference in
    # [-1, 1], against tau in {-1, 0, +1} (both-susceptible = 0), over ALL
    # participants. The released code applies a sigmoid and a {0, 0.5, 1}
    # encoding instead; the published numbers live on this scale, so both are
    # reported.
    out["pehe_paper"] = pehe(p1 - p0, 2.0 * true_ITE - 1.0)
    xor = true_ITE != 0.5
    if xor.sum() > 0:
        t, p = true_ITE[xor].astype(int), (pred_ITE[xor] > 0.5).astype(int)
        tp = int(((t == 1) & (p == 1)).sum()); tn = int(((t == 0) & (p == 0)).sum())
        fp = int(((t == 0) & (p == 1)).sum()); fn = int(((t == 1) & (p == 0)).sum())
        sens = tp / (tp + fn) if (tp + fn) else np.nan
        spec = tn / (tn + fp) if (tn + fp) else np.nan
        out["prescriptive_balacc"] = 0.5 * (sens + spec)
        out["pehe_xor"] = pehe(pred_ITE[xor], true_ITE[xor])
    else:
        out["prescriptive_balacc"] = np.nan
        out["pehe_xor"] = np.nan
    return out


VOLUME_STRATA = 4


def volume_strata(vols_all, vols) -> np.ndarray:
    """Cohort-level lesion-volume quartile (0 = smallest .. 3 = largest) of
    each entry of ``vols``. The cut points come from the WHOLE cohort so every
    fold and deficit shares the same strata (the equity gate of E11a uses the
    same convention on the proxy scale)."""
    edges = np.quantile(np.asarray(vols_all, dtype=float), [0.25, 0.5, 0.75])
    return np.searchsorted(edges, np.asarray(vols, dtype=float), side="right")


def stratified_pehe_paper(p1: np.ndarray, p0: np.ndarray, true_ITE: np.ndarray,
                          strata: np.ndarray, n_strata: int = VOLUME_STRATA) -> Dict[str, float]:
    """``pehe_paper`` inside each stratum (nan when the stratum is empty in
    this test slice) plus the member counts. Same estimand as the headline,
    restricted to the participants of one stratum, so a ratio between strata
    is an equity statement on the paper scale rather than on the proxy."""
    tau_hat = np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)
    tau = 2.0 * np.asarray(true_ITE, dtype=float) - 1.0
    strata = np.asarray(strata)
    out: Dict[str, float] = {}
    for s in range(n_strata):
        m = strata == s
        out[f"pehe_paper_vq{s}"] = pehe(tau_hat[m], tau[m]) if m.any() else float("nan")
        out[f"n_vq{s}"] = int(m.sum())
    return out


def effect_decomposition(p1: np.ndarray, p0: np.ndarray, true_ITE: np.ndarray) -> Dict[str, float]:
    """Where ``pehe_paper`` comes from in one test slice (audit H5: does an
    estimator rank the participants well but predict effects that are too
    small?). tau_hat = p1 - p0 against tau = 2 * true_ITE - 1 in {-1, 0, +1}.

    - ``tau_hat_pos`` / ``tau_hat_zero`` / ``tau_hat_neg``: mean tau_hat among
      the participants with tau = +1, 0 and -1 (``n_pos`` / ``n_zero`` /
      ``n_neg`` of them; nan when a group is empty). A perfect estimator gives
      +1, 0 and -1.
    - ``tau_slope``: least-squares slope of tau_hat on tau (1 = the full
      effect, 0 = no effect predicted); nan when tau is constant in the slice.
    - ``mse_between`` + ``mse_within`` = ``pehe_paper`` ** 2: the error of the
      group means (magnitude and offset) and the spread of tau_hat inside each
      group (ranking noise).
    - ``pehe_paper_rescaled``: ``pehe_paper`` after the single factor
      ``rescale_factor`` that best maps tau_hat onto tau in this slice. The
      factor is fitted on the test labels, so this is a bound on what fixing
      the magnitude alone could reach, not a score.

    Aggregates of the slice only; no participant-level values."""
    tau_hat = np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)
    tau = 2.0 * np.asarray(true_ITE, dtype=float) - 1.0
    n = len(tau)
    out: Dict[str, float] = {}
    between = within = 0.0
    for key, g in (("pos", 1.0), ("zero", 0.0), ("neg", -1.0)):
        m = np.isclose(tau, g)
        out[f"n_{key}"] = int(m.sum())
        if m.any():
            mu = float(tau_hat[m].mean())
            out[f"tau_hat_{key}"] = mu
            between += m.sum() * (mu - g) ** 2
            within += float(((tau_hat[m] - mu) ** 2).sum())
        else:
            out[f"tau_hat_{key}"] = float("nan")
    out["mse_between"], out["mse_within"] = float(between / n), float(within / n)
    dt = tau - tau.mean()
    var = float((dt ** 2).sum())
    out["tau_slope"] = float((dt * (tau_hat - tau_hat.mean())).sum() / var) if var > 0 else float("nan")
    ss = float((tau_hat ** 2).sum())
    b = float((tau_hat * tau).sum() / ss) if ss > 0 else 0.0
    out["rescale_factor"] = b
    out["pehe_paper_rescaled"] = pehe(b * tau_hat, tau)
    out["pehe_paper_isotonic"] = isotonic_pehe(tau_hat, tau)
    return out


def isotonic_pehe(tau_hat: np.ndarray, tau: np.ndarray, n_folds: int = 5, seed: int = 0) -> float:
    """``pehe_paper`` after the best monotone (isotonic) mapping of tau_hat
    onto tau, cross-fitted over ``n_folds`` random folds of the slice so the
    mapping is never fitted on the patients it is applied to. The monotone
    mapping keeps the estimator's ordering of patients and lets zeros stay at
    zero while large effects stretch, so this is the PEHE the ranking alone
    could reach; the gap between it and ``pehe_paper_rescaled`` is what a
    non-linear reshaping of the effects could recover, and the gap between it
    and a competitor's is ranking. nan when the slice is too small."""
    from sklearn.isotonic import IsotonicRegression

    tau_hat, tau = np.asarray(tau_hat, dtype=float), np.asarray(tau, dtype=float)
    n = len(tau)
    if n < 2 * n_folds or tau.std() == 0:
        return float("nan")
    perm = np.random.default_rng(seed).permutation(n)
    out = np.empty(n)
    for k in range(n_folds):
        te = perm[k::n_folds]
        tr = np.setdiff1d(perm, te)
        iso = IsotonicRegression(out_of_bounds="clip").fit(tau_hat[tr], tau[tr])
        out[te] = iso.predict(tau_hat[te])
    return pehe(out, tau)


# --------------------------------------------------------------------------- #
# Estimators (one-model / two-model, as in prescription.py)
# --------------------------------------------------------------------------- #
BUILTIN_CLASSIFIERS = ("logistic_regression", "extra_trees")

# Extra classifiers (the tabular foundation-model panel of 19 Oct: TabPFN,
# TabICL in ``neurocausalpfn.pfn.panel``): name -> factory returning an
# sklearn-style classifier (fit / predict_proba / classes_). They take the
# one-model and two-model learners exactly like the built-in logistic
# regression and extra trees and compete in the same headline cell selection,
# unlike the in-context ESTIMATORS below, which read the whole slice at once.
CLASSIFIERS: Dict[str, Callable] = {}


def register_classifier(name: str, factory: Callable) -> None:
    if name in BUILTIN_CLASSIFIERS:
        raise ValueError(f"{name!r} is a built-in classifier")
    CLASSIFIERS[name] = factory


def _classifier(name: str):
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegression
    if name == "logistic_regression":
        # the released code uses LogisticRegression() as is (sklearn's
        # max_iter=100), and so does this replica from 29 Oct; every score
        # before that used 1000, which GILES_LR_MAX_ITER=1000 reproduces
        return LogisticRegression(max_iter=int(os.environ.get("GILES_LR_MAX_ITER", "100") or 100))
    if name == "extra_trees":
        return ExtraTreesClassifier()
    if name in CLASSIFIERS:
        return CLASSIFIERS[name]()
    raise ValueError(f"unknown classifier {name!r}")


def _fit_predict_two_model(Xtr, W, Y, Xte, name):
    p = []
    for w in (1, 0):
        Xw, Yw = Xtr[W == w], Y[W == w]
        if len(set(Yw)) <= 1:
            p.append(np.full(len(Xte), float(Yw[0]) if len(Yw) else 0.0))
        else:
            m = _classifier(name)
            m.fit(Xw, Yw)
            p.append(m.predict_proba(Xte)[:, 1])
    return p[0], p[1]


def _fit_predict_one_model(Xtr, W, Y, Xte, name):
    Xtr_w = np.hstack([Xtr, W.reshape(-1, 1)])
    if len(set(Y)) <= 1:
        return (np.full(len(Xte), float(Y[0])),) * 2
    m = _classifier(name)
    m.fit(Xtr_w, Y)
    p1 = m.predict_proba(np.hstack([Xte, np.ones((len(Xte), 1))]))[:, 1]
    p0 = m.predict_proba(np.hstack([Xte, np.zeros((len(Xte), 1))]))[:, 1]
    return p1, p0


MIN_N = 15   # prescription_processor.min_n: slices smaller than this are skipped


# --------------------------------------------------------------------------- #
# In-context estimators (Phase 2): scored inside the same replica
# --------------------------------------------------------------------------- #
# name -> fn(Xtr, W, Y, Xte) -> (p1, p0). An in-context estimator takes the
# whole train slice as its context and predicts both potential outcomes of the
# test slice in one call, so it has no one-/two-model learner variant: it is
# scored once per (deficit, fold) with learner "in_context" and competes with
# the classifiers in the headline. Register with ``register_estimator``; the
# PFN wrapper lives in ``neurocausalpfn.pfn.estimator``.
ESTIMATORS: Dict[str, Callable] = {}


def register_estimator(name: str, fn: Callable) -> None:
    if name in BUILTIN_CLASSIFIERS or name in CLASSIFIERS:
        raise ValueError(f"{name!r} is a classifier")
    ESTIMATORS[name] = fn


# --------------------------------------------------------------------------- #
# Evaluation driver
# --------------------------------------------------------------------------- #
def image_folds(n: int, n_folds: int = 10, seed: int = 0):
    """The default folds: KFold over images, shuffled, fixed seed."""
    from sklearn.model_selection import KFold

    return [(tr, te) for tr, te in
            KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(np.arange(n))]


def load_group_table(path: str) -> Dict[str, Tuple[str, int]]:
    """``filename -> (group, rank)`` from a CSV with columns filename, group,
    rank (rank 0 = the group's earliest acquisition)."""
    import csv

    table: Dict[str, Tuple[str, int]] = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            table[os.path.basename(r["filename"])] = (str(r["group"]), int(r["rank"]))
    return table


def group_folds(files: Sequence[str], groups: Dict[str, Tuple[str, int]],
                n_folds: int = 10, seed: int = 0, mode: str = "giles",
                allow_missing: bool = False):
    """Group-aware folds over a file listing.

    Only the rank-0 image of each group (its earliest acquisition) is ever
    tested: the rank-0 images are split with the same KFold as image_folds and
    the later acquisitions are added to the training side. Two placements:

    - ``giles``: every later acquisition trains in EVERY fold (the paper's
      protocol; a group may then sit in train through a repeat while its
      earliest image is in test).
    - ``strict``: a later acquisition trains only in the folds where its
      group's earliest image trains, so no group ever straddles the split.

    A group with no rank-0 image is train-only in every fold (both modes).
    Files absent from the table are treated as single-image groups when
    ``allow_missing`` is set, else raise.
    """
    from sklearn.model_selection import KFold

    names = [os.path.basename(f) for f in files]
    missing = [nm for nm in names if nm not in groups]
    if missing and not allow_missing:
        raise KeyError(f"{len(missing)} files missing from the group table, e.g. {missing[:3]}")
    primary, extras_by_group = [], {}
    for i, nm in enumerate(names):
        g, rank = groups.get(nm, (f"__solo__{nm}", 0))
        if rank == 0:
            primary.append((i, g))
        else:
            extras_by_group.setdefault(g, []).append(i)
    seen = {}
    for i, g in primary:
        if g in seen:
            raise ValueError(f"group {g} has two rank-0 images ({names[seen[g]]}, {names[i]})")
        seen[g] = i
    # a group with no rank-0 image is never tested: all its images train in
    # every fold (this is how a table can mark whole groups as train-only,
    # e.g. the paper's multi-acquisition subjects)
    train_only = [i for g, lst in extras_by_group.items() if g not in seen for i in lst]
    prim_idx = np.array([i for i, _ in primary])
    prim_grp = [g for _, g in primary]
    all_extras = [i for lst in extras_by_group.values() for i in lst]
    folds = []
    for tr_p, te_p in KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(prim_idx):
        te = prim_idx[te_p]
        if mode == "giles":
            tr = list(prim_idx[tr_p]) + all_extras
        elif mode == "strict":
            tr = list(prim_idx[tr_p]) + train_only
            for k in tr_p:
                tr += extras_by_group.get(prim_grp[k], [])
        else:
            raise ValueError("mode must be 'giles' or 'strict'")
        folds.append((np.array(sorted(tr)), np.array(sorted(te))))
    return folds


def straddling_groups(files: Sequence[str], groups: Dict[str, Tuple[str, int]], folds) -> int:
    """Number of (fold, group) pairs where a tested image's group also has an
    image on the training side of that fold. Zero under ``strict``; under
    ``giles`` it counts the groups whose earliest image is tested while a
    repeat trains (28 Oct review: the logs now state it). Counts only."""
    names = [os.path.basename(f) for f in files]
    grp = [groups.get(nm, (f"__solo__{nm}", 0))[0] for nm in names]
    n = 0
    for tr, te in folds:
        train_groups = {grp[i] for i in tr}
        n += sum(grp[i] in train_groups for i in te)
    return n


def evaluate_representation(Z: np.ndarray, labels_df, pairs: Dict[int, RoiPair],
                            scenario: Dict, n_folds: int = 10,
                            deficits: Optional[Sequence[int]] = None,
                            classifiers: Sequence[str] = ("logistic_regression", "extra_trees"),
                            learners: Sequence[str] = ("one", "two"),
                            collect_sims: Optional[list] = None,
                            folds: Optional[Sequence[Tuple[np.ndarray, np.ndarray]]] = None,
                            fold_ids: Optional[Sequence[int]] = None):
    """Score a representation on the Giles scale.

    Z is [n, d], row-aligned with labels_df (one row per lesion file). For
    each deficit (default all 16), each fold and the given scenario, fits the
    classifiers on the train slice and scores PEHE on the test slice. Returns
    a DataFrame of per-(deficit, fold, classifier, learner) results; the
    headline aggregate is the mean over deficits of the best
    classifier x learner cell, as in the paper's Figure 6.

    Z may instead be a callable ``(tr_idx, te_idx) -> (Z_tr, Z_te)`` that is
    invoked once per fold with the file-level indices. This is how the paper
    fits its reductions (NMF/PCA on the train side of each fold only), and it
    is the honest anchor when comparing against published numbers: a fixed Z
    computed from the full cohort leaks the test lesions' anatomy into the
    embedding, which flatters PEHE.

    When ``collect_sims`` is a list, every simulated train slice is appended
    to it as records (filename, deficit, fold, y_true, W, Y, scenario) so the
    raw simulations with outcomes can be exported.

    ``folds`` overrides the default image-level KFold with an explicit list of
    (train_idx, test_idx) pairs, e.g. from :func:`group_folds`. ``fold_ids``
    names those folds when they are a subset of the full split (the scorer's
    --only-folds): fold ``fold_ids[i]`` seeds its virtual trial and fills the
    fold column as in the full run, instead of its position ``i``.
    """
    import pandas as pd

    deficits = list(deficits) if deficits is not None else list(range(1, 17))
    results = []
    n = len(labels_df)
    if folds is None:
        folds = image_folds(n, n_folds)
    fold_ids = list(range(len(folds))) if fold_ids is None else [int(k) for k in fold_ids]
    if len(fold_ids) != len(folds):
        raise ValueError(f"{len(fold_ids)} fold ids for {len(folds)} folds")
    # equity on the paper scale: cohort-level volume quartile of every image
    strata_all = (volume_strata(labels_df["vol"], labels_df["vol"])
                  if "vol" in labels_df.columns else None)
    # in-context estimators are the slow part of a scoring job (one forward
    # pass per deficit, fold and arm): report each finished fold so that a
    # job killed at its time limit shows how far it got
    progress = any(c in ESTIMATORS or c in CLASSIFIERS for c in classifiers)
    t_start = time.time()
    for i, (k, (tr_idx, te_idx)) in enumerate(zip(fold_ids, folds)):
        if progress and i:
            print(f"  fold {i}/{len(folds)} done, {time.time() - t_start:.0f} s elapsed", flush=True)
        df_tr, df_te = labels_df.iloc[tr_idx], labels_df.iloc[te_idx]
        if callable(Z):
            Z_tr_all, Z_te_all = Z(tr_idx, te_idx)
        else:
            Z_tr_all, Z_te_all = Z[tr_idx], Z[te_idx]
        for d in deficits:
            sl_tr = deficit_slice(df_tr.reset_index(drop=True), d)
            sl_te = deficit_slice(df_te.reset_index(drop=True), d)
            if sl_tr is None or sl_te is None or len(sl_tr) < MIN_N or not len(sl_te):
                continue
            trial = GilesTrial(scenario["TE"], scenario["RE"], scenario["BIAS"],
                               scenario["BIASTYPE"], k)
            W, Y = trial.simulate(sl_tr, pairs[d].centroids)
            true_ITE = sl_te["y_true"].to_numpy()
            # row-align the latents with the slices via the membership masks
            name = DEFICIT_NAMES[d - 1]
            mask_tr = ((df_tr[f"{name}_W0"] == 1) | (df_tr[f"{name}_W1"] == 1)).to_numpy()
            mask_te = ((df_te[f"{name}_W0"] == 1) | (df_te[f"{name}_W1"] == 1)).to_numpy()
            Xtr, Xte = Z_tr_all[mask_tr], Z_te_all[mask_te]
            if collect_sims is not None:
                for fn, yt, w, y in zip(sl_tr["filename"], sl_tr["y_true"], W, Y):
                    collect_sims.append({"filename": fn, "deficit": name, "fold": k,
                                         "y_true": yt, "W": int(w), "Y": int(y), **scenario})
            strata_te = strata_all[te_idx][mask_te] if strata_all is not None else None
            for cname in classifiers:
                in_context = cname in ESTIMATORS
                for learner in (("in_context",) if in_context else learners):
                    if in_context:
                        p1, p0 = ESTIMATORS[cname](Xtr, W, Y, Xte)
                        p1, p0 = np.asarray(p1, float).ravel(), np.asarray(p0, float).ravel()
                    else:
                        fit = _fit_predict_one_model if learner == "one" else _fit_predict_two_model
                        p1, p0 = fit(Xtr, W, Y, Xte, cname)
                    row = {"deficit": name, "fold": k, "classifier": cname,
                           "learner": learner, "n_train": len(Xtr), "n_test": len(Xte),
                           **scenario, **score_predictions(p1, p0, true_ITE),
                           **effect_decomposition(p1, p0, true_ITE)}
                    if strata_te is not None:
                        row.update(stratified_pehe_paper(p1, p0, true_ITE, strata_te))
                    results.append(row)
    return pd.DataFrame(results)


def _deficit_ci(best) -> Dict[str, float]:
    mean = float(best.mean())
    if len(best) > 1:
        se = float(best.std(ddof=1) / np.sqrt(len(best)))
        ci = (mean - 1.96 * se, mean + 1.96 * se)
    else:
        ci = (float("nan"), float("nan"))
    return {"pehe_mean": mean, "ci_low": ci[0], "ci_high": ci[1],
            "n_deficits": int(len(best))}


def headline_aggregate(results) -> Dict[str, float]:
    """The paper's aggregation (its supplementary tables report one global
    configuration, e.g. "VAE 50 / logistic regression / two-model"): pick the
    single classifier x learner with the lowest PEHE averaged over deficits,
    then report that configuration's mean and 95% CI over deficits.

    Also returns ``pehe_per_deficit_best`` — letting each deficit pick its own
    best cell — as a diagnostic; it is optimistic by selection and must not be
    compared against the published numbers."""
    if len(results) == 0:
        # no deficit reached MIN_N susceptible lesions (small cohorts / smokes)
        return {"pehe_mean": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n_deficits": 0,
                "classifier": "", "learner": "",
                "pehe_per_deficit_best": float("nan")}
    per = (results.groupby(["deficit", "classifier", "learner"])["pehe"]
           .mean().reset_index())
    overall = per.groupby(["classifier", "learner"])["pehe"].mean()
    cbest, lbest = overall.idxmin()
    fixed = per[(per["classifier"] == cbest) & (per["learner"] == lbest)]
    out = _deficit_ci(fixed.set_index("deficit")["pehe"])
    out["classifier"], out["learner"] = cbest, lbest
    out["pehe_per_deficit_best"] = float(per.groupby("deficit")["pehe"].min().mean())
    return out


def headline_row(results) -> Dict[str, float]:
    """One complete headline entry for a representation: the paper-style
    aggregate plus, for the SAME classifier x learner the PEHE headline picked,
    the companion metrics — balanced accuracy (the paper's other calibration
    anchor), the code-convention pehe-xor, and the Methods-convention
    pehe-paper, which is the externally comparable number."""
    agg = headline_aggregate(results)
    if agg["classifier"]:
        cfg = results[(results["classifier"] == agg["classifier"]) &
                      (results["learner"] == agg["learner"])]
        for col, key in (("prescriptive_balacc", "balacc_mean"),
                         ("pehe_xor", "pehe_xor_mean"),
                         ("pehe_paper", "pehe_paper_mean")):
            if col in cfg.columns:
                agg[key] = float(cfg.groupby("deficit")[col].mean().mean())
        if "pehe_paper" in cfg.columns:
            # 95% CI over deficits of the paper-scale PEHE of the selected cell
            # (Giles's own interval), and the cell the paper scale itself would
            # select: both are reported so that the selection scale is explicit
            ci = _deficit_ci(cfg.groupby("deficit")["pehe_paper"].mean())
            agg["pehe_paper_ci_low"], agg["pehe_paper_ci_high"] = ci["ci_low"], ci["ci_high"]
            per_p = results.groupby(["deficit", "classifier", "learner"])["pehe_paper"].mean().reset_index()
            cb, lb = per_p.groupby(["classifier", "learner"])["pehe_paper"].mean().idxmin()
            sel = per_p[(per_p["classifier"] == cb) & (per_p["learner"] == lb)].set_index("deficit")["pehe_paper"]
            ci2 = _deficit_ci(sel)
            agg["pehe_paper_selected"] = ci2["pehe_mean"]
            agg["pehe_paper_selected_ci_low"], agg["pehe_paper_selected_ci_high"] = ci2["ci_low"], ci2["ci_high"]
            agg["classifier_paper"], agg["learner_paper"] = cb, lb
        # equity by lesion-volume quartile on the paper scale (same estimand,
        # same configuration; folds where a quartile is empty are skipped by
        # the nan-aware means). volume_ratio = worst / best quartile.
        vq_cols = sorted(c for c in cfg.columns if c.startswith("pehe_paper_vq"))
        if vq_cols:
            means = [float(cfg.groupby("deficit")[c].mean().mean()) for c in vq_cols]
            for c, m in zip(vq_cols, means):
                agg[f"{c}_mean"] = m
            finite = [m for m in means if np.isfinite(m)]
            agg["volume_ratio"] = (float(max(finite) / min(finite))
                                   if finite and min(finite) > 0 else float("nan"))
    return agg
