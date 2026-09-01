"""Faithful replica of the Giles et al. (2025) virtual-trial evaluation.

Ports the ground-truth simulation and the PEHE scale of
``high-dimensional/individualized_prescriptive_inference`` (Nature
Communications 16:8968; code CC BY-NC-SA 4.0, adapted here with citation) so
that any lesion representation — theirs or ours — can be scored on the SAME
scale as the published headline (disconnectome VAE-50: PEHE 0.349, 95% CI
[0.33, 0.37], random allocation, averaged over the 16 functional networks;
0.305 under location bias).

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

Known deviation (documented): our frozen encoders are trained unsupervised on
the full lesion set, whereas Giles refits each representation inside every
train fold. Outcomes are simulated after the fact, so no outcome leakage is
possible; the residual anatomical leakage is stated, not hidden. Estimator
fitting remains strictly train-fold-only. Patient-level fold separation
requires the lesion->patient mapping (requested; not yet available), so folds
are per-lesion.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

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
    """Observed PEHE and prescriptive balanced accuracy on the Giles scale:
    pred_ITE = sigmoid(p1 - p0) against y_true in {0, 0.5, 1}.

    Calibration note (full-cohort runs, 4,119 disconnectomes): ``pehe_xor``
    (decidable cases only, y_true != 0.5) is the quantity that matches the
    published headline — our NMF-50 scores 0.343-0.352 across receptor,
    genetics and the nimfa variant, against the paper's VAE-50 0.349
    [0.33, 0.37]; balanced accuracy matches too (0.80-0.89 vs 0.875, volume
    ~0.52 vs the vascular baseline 0.523-0.546). The all-cases ``pehe`` sits
    lower because both-susceptible patients (sigmoid(0) = 0.5, error-free)
    dilute the mean; compare representations against the paper on pehe_xor."""
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


# --------------------------------------------------------------------------- #
# Estimators (one-model / two-model, as in prescription.py)
# --------------------------------------------------------------------------- #
def _classifier(name: str):
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegression
    if name == "logistic_regression":
        return LogisticRegression(max_iter=1000)
    if name == "extra_trees":
        return ExtraTreesClassifier()
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
# Evaluation driver
# --------------------------------------------------------------------------- #
def evaluate_representation(Z: np.ndarray, labels_df, pairs: Dict[int, RoiPair],
                            scenario: Dict, n_folds: int = 10,
                            deficits: Optional[Sequence[int]] = None,
                            classifiers: Sequence[str] = ("logistic_regression", "extra_trees"),
                            learners: Sequence[str] = ("one", "two"),
                            collect_sims: Optional[list] = None):
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
    """
    import pandas as pd
    from sklearn.model_selection import KFold

    deficits = list(deficits) if deficits is not None else list(range(1, 17))
    results = []
    n = len(labels_df)
    for k, (tr_idx, te_idx) in enumerate(
            KFold(n_splits=n_folds, shuffle=True, random_state=0).split(np.arange(n))):
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
            for cname in classifiers:
                for learner in learners:
                    fit = _fit_predict_one_model if learner == "one" else _fit_predict_two_model
                    p1, p0 = fit(Xtr, W, Y, Xte, cname)
                    row = {"deficit": name, "fold": k, "classifier": cname,
                           "learner": learner, "n_train": len(Xtr), "n_test": len(Xte),
                           **scenario, **score_predictions(p1, p0, true_ITE)}
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
