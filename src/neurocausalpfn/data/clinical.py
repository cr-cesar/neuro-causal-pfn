"""Covariables clinicas: parseo del nombre de archivo y construccion del vector.

En los datos de Giles, la edad y el sexo no vienen en una tabla aparte sino
codificados en el propio nombre del archivo, con el patron
lesion{arbitrary_id}_{age}_{sex}.nii.gz, y con el literal NA cuando el dato no
esta disponible. Este modulo extrae esos campos, los normaliza y construye un
vector de covariables de dimension fija que incluye un indicador de dato
faltante por cada variable, de modo que el modelo distingue un valor real de una
imputacion.

Convenciones (documentadas y ajustables a la cohorte local):
- edad: normalizada como (edad - AGE_MEAN) / AGE_SD; faltante a 0.0 con indicador 1.
- sexo: varon a +0.5, mujer a -0.5; faltante a 0.0 con indicador 1.
Tokens de sexo aceptados: M, F, MALE, FEMALE (y, por compatibilidad, 1 y 0,
asumiendo 1 = varon y 0 = mujer). Cualquier otro token se trata como faltante.
"""
import os
import re
from typing import Dict, List, Optional

import numpy as np

AGE_MEAN = 65.0   # media aproximada en cohortes de ictus; ajustar a los datos locales
AGE_SD = 15.0
CLINICAL_DIM = 4  # [edad_norm, edad_faltante, sexo_val, sexo_faltante]

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
    """Extrae id, edad y sexo de lesion{id}_{age}_{sex}.nii.gz.

    Los dos ultimos campos separados por guion bajo son edad y sexo; todo lo
    anterior (sin el prefijo 'lesion') es el id, de modo que el id puede contener
    guiones bajos sin romper el parseo. Devuelve age como float o None, y sex como
    'M', 'F' o None."""
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
    """Vector [CLINICAL_DIM] con indicadores de dato faltante."""
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
    """Matriz [N, CLINICAL_DIM] parseada de una lista de nombres de archivo."""
    rows = [build_clinical_vector(*(lambda m: (m["age"], m["sex"]))(parse_lesion_filename(p)))
            for p in paths]
    if not rows:
        return np.zeros((0, CLINICAL_DIM), dtype=np.float32)
    return np.stack(rows, axis=0)


def synthesize_clinical(n: int, d: int = CLINICAL_DIM, seed: int = 0) -> np.ndarray:
    """Covariables sinteticas para el modo prototipo sin datos reales."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, size=(n, d)).astype(np.float32)


def load_clinical(path: Optional[str], n: int, d: int = CLINICAL_DIM, seed: int = 0) -> np.ndarray:
    if path:
        import pandas as pd

        return pd.read_csv(path).to_numpy(dtype=np.float32)
    return synthesize_clinical(n, d, seed)
