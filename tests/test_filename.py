import numpy as np

from neurocausalpfn.data.clinical import (build_clinical_vector, clinical_from_paths,
                                          parse_lesion_filename)


def test_parse_full():
    m = parse_lesion_filename("lesion42_67_M.nii.gz")
    assert m["id"] == "42"
    assert m["age"] == 67.0
    assert m["sex"] == "M"


def test_parse_age_missing():
    m = parse_lesion_filename("lesion7_NA_F.nii.gz")
    assert m["age"] is None
    assert m["sex"] == "F"


def test_parse_sex_missing():
    m = parse_lesion_filename("lesion9_55_NA.nii.gz")
    assert m["age"] == 55.0
    assert m["sex"] is None


def test_parse_both_missing():
    m = parse_lesion_filename("lesion3_NA_NA.nii.gz")
    assert m["age"] is None and m["sex"] is None


def test_parse_nii_only_and_word_sex():
    m = parse_lesion_filename("lesion12_80_female.nii")
    assert m["age"] == 80.0 and m["sex"] == "F"


def test_parse_id_with_underscore():
    # los dos ultimos campos son edad y sexo; el resto es el id
    m = parse_lesion_filename("lesionA_B_70_M.nii.gz")
    assert m["age"] == 70.0 and m["sex"] == "M" and m["id"] == "A_B"


def test_clinical_vector_missing_flags():
    v = build_clinical_vector(None, None)
    assert v.shape == (4,)
    assert v[1] == 1.0 and v[3] == 1.0   # indicadores de faltante encendidos
    assert v[0] == 0.0 and v[2] == 0.0   # imputados a la media


def test_clinical_vector_present():
    v = build_clinical_vector(65.0, "M")
    assert v[0] == 0.0 and v[1] == 0.0   # 65 == AGE_MEAN normaliza a 0
    assert v[2] == 0.5 and v[3] == 0.0


def test_clinical_from_paths():
    paths = ["lesion1_70_M.nii.gz", "lesion2_NA_F.nii.gz"]
    mat = clinical_from_paths(paths)
    assert mat.shape == (2, 4)
    assert mat[1, 1] == 1.0   # segunda fila: edad faltante
    assert mat[0, 2] == 0.5   # primera fila: varon
