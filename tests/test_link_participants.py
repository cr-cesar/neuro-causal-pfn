import importlib.util
import os

_spec = importlib.util.spec_from_file_location(
    "link_participants",
    os.path.join(os.path.dirname(__file__), "..", "scripts", "link_participants.py"))
lp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lp)


def _fixture():
    cw = [{"a_file": "pub1.nii.gz", "b_file": "x_00000904_F_19350615_orig_a.nii.gz"},
          {"a_file": "pub2.nii.gz", "b_file": "x_00001200_M_19400101_orig_b.nii.gz"},
          {"a_file": "pub3.nii.gz", "b_file": "orig_missing.nii.gz"}]
    info = [{"seg": "L_PCA/x_00000904_F_19350615_orig_a.nii.gz", "pid": "S904"},
            {"seg": "R_MCA/x_00001200_M_19400101_orig_b.nii.gz", "pid": "S1200"}]
    parts = [{"participant_id": "U904", "age": "70", "n1": "1", "n2": "2"},
             {"participant_id": "K1200", "age": "65", "n1": "0", "n2": ""},
             {"participant_id": "U5", "age": "50", "n1": "0", "n2": "0"}]
    return cw, info, parts


def test_helpers():
    assert lp.numeric("uclh_0042") == 42 and lp.numeric("abc") is None
    assert lp.digit_runs("x_00000904_F_19350615_orig.nii.gz") == [904, 19350615]
    assert lp.site_of("kch_12") == "kch" and lp.site_of("U7") == "u"


def test_bridges_report_coverage_per_rule_and_emit_hides_identifiers():
    cw, info, parts = _fixture()
    p2o = lp.public_to_original(cw, info, "seg", "pid")
    assert set(p2o) == {"pub1.nii.gz", "pub2.nii.gz"}
    br = lp.bridges(p2o, parts, "participant_id")
    assert set(br["A"]) == {"pub1.nii.gz", "pub2.nii.gz"}        # numeric subject key == numeric participant key
    assert set(br["B:0"]) == {"pub1.nii.gz", "pub2.nii.gz"}      # first digit run of the path
    assert "B:1" not in br or not br["B:1"]                        # the date run matches nobody
    assert "C" not in br or not br["C"]                            # strings differ
    rows = lp.emit_rows(br["A"], "participant_id", ["age"], nihss_prefix="n")
    by = {r["filename"]: r for r in rows}
    assert by["pub1.nii.gz"] == {"filename": "pub1.nii.gz", "site": "u", "age": "70",
                                 "nihss_total": 3, "nihss_n_items": 2}
    assert by["pub2.nii.gz"]["nihss_total"] == "" and by["pub2.nii.gz"]["site"] == "k"
    blob = str(rows)
    assert "S904" not in blob and "19350615" not in blob and "orig_a" not in blob
