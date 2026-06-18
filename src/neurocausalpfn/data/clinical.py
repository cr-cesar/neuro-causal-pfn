"""Clinical covariates: filename parsing and vector construction.

In the Giles data, age and sex do not come in a separate table but encoded in
the filename itself, with the pattern lesion{arbitrary_id}_{age}_{sex}.nii.gz,
and with the literal NA when the data is not available. This module extracts
those fields, normalizes them and builds a fixed-dimension covariate vector that
includes a missing-data indicator for each variable, so that the model
distinguishes a real value from an imputation.

Conventions (documented and adjustable to the local cohort):
- age: normalized as (age - AGE_MEAN) / AGE_SD; missing to 0.0 with indicator 1.
- sex: male to +0.5, female to -0.5; missing to 0.0 with indicator 1.
Accepted sex tokens: M, F, MALE, FEMALE (and, for compatibility, 1 and 0,
assuming 1 = male and 0 = female). Any other token is treated as missing.
"""
import os
import re
from typing import Dict, List, Optional

import numpy as np

AGE_MEAN = 65.0   # approximate mean in stroke cohorts; adjust to the local data
AGE_SD = 15.0
CLINICAL_DIM = 4  # [age_norm, age_missing, sex_val, sex_missing]

_MALE = {"M", "MALE", "1"}
_FEMALE = {"F", "FEMALE", "0"}
_MISSING = {"NA", "NAN", "NONE", ""}


def _strip_nifti_ext(name: str) -> str:
    base = os.path.basename(name)
    low = base.lower()
    for ext in (".nii.gz", ".nii"):
        if low.endswith(ext):
            return base[: -len(ext)]
    return base


def parse_lesion_filename(name: str) -> Dict[str, object]:
    """Extracts id, age and sex from lesion{id}_{age}_{sex}.nii.gz.

    The last two underscore-separated fields are age and sex; everything before
    (without the 'lesion' prefix) is the id, so that the id can contain
    underscores without breaking the parsing. Returns age as float or None, and
    sex as 'M', 'F' or None."""
    stem = _strip_nifti_ext(name)
    parts = stem.split("_")
    age_raw = parts[-2] if len(parts) >= 2 else "NA"
    sex_raw = parts[-1] if len(parts) >= 1 else "NA"
    if len(parts) >= 3:
        id_part = "_".join(parts[:-2])
    else:
        id_part = parts[0] if parts else ""
    id_part = re.sub(r"^lesion", "", id_part, flags=re.IGNORECASE)

    try:
        age: Optional[float] = float(age_raw)
    except (ValueError, TypeError):
        age = None

    sex_token = str(sex_raw).strip().upper()
    if sex_token in _MALE:
        sex: Optional[str] = "M"
    elif sex_token in _FEMALE:
        sex = "F"
    else:
        sex = None

    return {"id": id_part, "age": age, "sex": sex,
            "age_raw": age_raw, "sex_raw": sex_raw}


def build_clinical_vector(age: Optional[float], sex: Optional[str]) -> np.ndarray:
    """Vector [CLINICAL_DIM] with missing-data indicators."""
    if age is None:
        age_norm, age_missing = 0.0, 1.0
    else:
        age_norm, age_missing = (float(age) - AGE_MEAN) / AGE_SD, 0.0
    if sex is None:
        sex_val, sex_missing = 0.0, 1.0
    elif sex == "M":
        sex_val, sex_missing = 0.5, 0.0
    else:
        sex_val, sex_missing = -0.5, 0.0
    return np.array([age_norm, age_missing, sex_val, sex_missing], dtype=np.float32)


def clinical_from_paths(paths: List[str]) -> np.ndarray:
    """Matrix [N, CLINICAL_DIM] parsed from a list of filenames."""
    rows = [build_clinical_vector(*(lambda m: (m["age"], m["sex"]))(parse_lesion_filename(p)))
            for p in paths]
    if not rows:
        return np.zeros((0, CLINICAL_DIM), dtype=np.float32)
    return np.stack(rows, axis=0)


def synthesize_clinical(n: int, d: int = CLINICAL_DIM, seed: int = 0) -> np.ndarray:
    """Synthetic covariates for prototype mode without real data."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, size=(n, d)).astype(np.float32)


def load_clinical(path: Optional[str], n: int, d: int = CLINICAL_DIM, seed: int = 0) -> np.ndarray:
    if path:
        import pandas as pd

        return pd.read_csv(path).to_numpy(dtype=np.float32)
    return synthesize_clinical(n, d, seed)
