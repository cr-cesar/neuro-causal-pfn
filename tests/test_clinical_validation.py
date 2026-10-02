"""Clinical validation: outcome parsing, covariate encoding, id matching, CV scores."""
import importlib.util
import os

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    spec = importlib.util.spec_from_file_location(
        "clinical_validation", os.path.join(ROOT, "scripts", "clinical_validation.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parse_outcome_and_binarise_keep_missing_as_nan():
    cv = _load()
    col, rule, fn = cv.parse_outcome("mRS 3 months:<=2")
    assert col == "mRS 3 months" and rule == "<=2"
    y = cv.binarise(pd.Series([0, 2, 3, None, "6"]), fn)
    assert np.array_equal(np.isnan(y), [False, False, False, True, False])
    assert list(y[~np.isnan(y)]) == [1.0, 1.0, 0.0, 0.0]
    with pytest.raises(SystemExit):
        cv.parse_outcome("mRS 3 months")


def test_encode_covariates_handles_sex_and_missing_age():
    cv = _load()
    df = pd.DataFrame({"Age": [60, None, 80], "Sex": ["M", "F", "M"]})
    X = cv.encode_covariates(df, ["Age", "Sex"])
    assert X.shape == (3, 2)
    assert X[1, 0] == pytest.approx(70.0)          # mean-imputed
    assert list(X[:, 1]) == [1.0, 0.0, 1.0]        # F=0, M=1 (sorted levels)


def test_match_rows_uses_id_prefix_of_file_name():
    cv = _load()
    ids = ["sub-stroke0002", "sub-stroke0001"]
    files = ["/d/sub-stroke0001_lesion.nii.gz", "/d/sub-stroke0002_lesion.nii.gz"]
    assert list(cv.match_rows(ids, files)) == [1, 0]
    with pytest.raises(SystemExit):
        cv.match_rows(["sub-stroke0001"], ["/d/sub-stroke0009_lesion.nii.gz"])


def test_cv_scores_separable_signal_scores_high_and_noise_near_chance():
    cv = _load()
    rng = np.random.default_rng(0)
    y = np.array([0, 1] * 40)
    X_signal = y[:, None] * 3.0 + rng.normal(scale=0.5, size=(80, 4))
    X_noise = rng.normal(size=(80, 4))
    s1 = cv.cv_scores(X_signal, y, n_splits=5, n_repeats=2)
    s0 = cv.cv_scores(X_noise, y, n_splits=5, n_repeats=2)
    assert s1["bal_acc"] > 0.95 and s1["auc"] > 0.95
    assert 0.3 < s0["bal_acc"] < 0.7
    assert s1["folds"] == 5
