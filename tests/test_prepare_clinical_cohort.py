"""prepare_clinical_cohort: separator sniffing, id matching by prefix, item totals."""
import importlib.util
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    spec = importlib.util.spec_from_file_location(
        "prep", os.path.join(ROOT, "scripts", "prepare_clinical_cohort.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def test_read_table_sniffs_comma_despite_tsv_extension(tmp_path):
    p = _load()
    f = tmp_path / "participants.tsv"
    f.write_text("participant_id,age,id\nsub-1,70,x1\nsub-2,65,x2\n")
    df = p.read_table(str(f))
    assert list(df.columns) == ["participant_id", "age", "id"] and len(df) == 2


def test_match_column_prefers_longest_prefix_and_best_column():
    p = _load()
    df = pd.DataFrame({"participant_id": ["sub-10", "sub-100", "sub-7"], "id": ["a", "b", "c"]})
    files = ["sub-100_lesion.nii.gz", "sub-10_lesion.nii.gz", "zzz.nii.gz"]
    col, rows, n = p.match_column(df, ["participant_id", "id"], files)
    assert col == "participant_id" and n == 2 and list(rows) == [1, 0, -1]


def test_item_total_is_nan_when_an_item_is_missing():
    p = _load()
    df = pd.DataFrame({f"S2NihssArrival{k}": [1, 2, None] for k in range(15)})
    df.iloc[1, 0] = 0
    t = p.item_total(df, "S2NihssArrival")
    assert t.iloc[0] == 15 and t.iloc[1] == 28 and np.isnan(t.iloc[2])
    df2 = pd.DataFrame({f"S2NihssArrival{k}": [1, 1, 1] for k in range(15)})
    df2.iloc[1, 3] = -1; df2.iloc[2, 5] = 9          # missing / untestable codes
    t2 = p.item_total(df2, "S2NihssArrival")
    assert t2.iloc[0] == 15 and np.isnan(t2.iloc[1]) and np.isnan(t2.iloc[2])


def test_diagnose_prints_shapes_not_ids(capsys):
    p = _load()
    df = pd.DataFrame({"participant_id": ["sub-1234", "sub-0007"], "id": ["k9", "k8"]})
    p.diagnose(df, ["participant_id", "id"], ["sub-1234_lesion.nii.gz", "sub-0007_lesion.nii.gz", "other.nii.gz"])
    out = capsys.readouterr().out
    assert "sub-####" in out and "1234" not in out and "files matched: 2/3" in out


def test_exact_match_with_stripped_prefix_is_unambiguous():
    p = _load()
    df = pd.DataFrame({"participant_id": ["K12", "K123", "U123"], "id": ["kch_12", "kch_123", "uclh_123"]})
    files = ["sub-K123.nii.gz", "sub-K12.nii.gz", "sub-U123.nii.gz", "sub-K9.nii.gz"]
    col, rows, n = p.match_column(df, ["participant_id", "id"], files, strip_prefix="sub-", exact=True)
    assert col == "participant_id" and list(rows) == [1, 0, 2, -1] and n == 3
