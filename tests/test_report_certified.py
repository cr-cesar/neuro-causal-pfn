"""The leaderboard's certified column: replica headline scores matched to
experiment variants, newest file winning per (variant, seed), certified rows
leading the sort."""
import csv
import os
import time

from neurocausalpfn.experiments.report import (_attach_certified,
                                               _read_certified,
                                               _stem_and_seed, build_report)


def _write_headline(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = ["representation", "pehe_mean", "pehe_paper_mean"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_runs(out_root, rows):
    os.makedirs(out_root, exist_ok=True)
    import json
    with open(os.path.join(out_root, "runs.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_stem_and_seed_parsing():
    assert _stem_and_seed("E2_E2_w_dice=0.1_seed0_disco.npz") == ("E2_E2_w_dice=0.1", 0, 0)
    assert _stem_and_seed("E1_seed2_lesion.npz") == ("E1", 2, 1)
    assert _stem_and_seed("E6_both_seed2.npz") == ("E6_both", 2, 0)
    assert _stem_and_seed("E7a_seed1.npz") == ("E7a", 1, 0)
    assert _stem_and_seed("nmf50") == (None, None, None)    # builtins have no seed
    assert _stem_and_seed("volume") == (None, None, None)


def test_certified_matches_variant_and_prefers_newest(tmp_path):
    out_root = str(tmp_path / "experiments")
    stale = str(tmp_path / "giles_replica_old" / "replica_headline.csv")
    clean = str(tmp_path / "giles_replica_new" / "replica_headline.csv")
    _write_headline(stale, [
        {"representation": "E2_E2_w_dice=0.5_seed0_disco.npz",
         "pehe_mean": 0.23, "pehe_paper_mean": 0.40},       # contaminated export
        {"representation": "nmf50", "pehe_mean": 0.215, "pehe_paper_mean": 0.32},
    ])
    _write_headline(clean, [
        {"representation": "E2_E2_w_dice=0.5_seed0_disco.npz",
         "pehe_mean": 0.21, "pehe_paper_mean": 0.31},       # clean re-run
        {"representation": "E2_E2_w_dice=0.5_seed1_disco.npz",
         "pehe_mean": 0.21, "pehe_paper_mean": 0.33},
        {"representation": "E2_E2_w_dice=0.1_seed0_disco.npz",
         "pehe_mean": 0.21, "pehe_paper_mean": 0.30},
    ])
    now = time.time()
    os.utime(stale, (now - 100, now - 100))
    os.utime(clean, (now, now))

    stems = _read_certified(out_root)
    assert stems["E2_E2_w_dice=0.5"] == [0.31, 0.33]        # newest wins seed 0
    assert stems["E2_E2_w_dice=0.1"] == [0.30]
    assert "nmf50" not in stems                             # no eid, no seed

    board = [
        {"arm": "A", "eid": "E2", "label": "E2[w_dice=0.5]", "n_seeds": 3,
         "passed_frac": 1.0, "T4.root_pehe.mean": 0.07},
        {"arm": "A", "eid": "E2", "label": "E2[w_dice=0.1]", "n_seeds": 3,
         "passed_frac": 1.0, "T4.root_pehe.mean": 0.08},
    ]
    _attach_certified(board, stems)
    assert abs(board[0]["certified.pehe_paper.mean"] - 0.32) < 1e-9
    assert board[0]["certified.n"] == 2
    assert abs(board[1]["certified.pehe_paper.mean"] - 0.30) < 1e-9


def test_disco_channel_beats_lesion_channel_even_when_older(tmp_path):
    # the failure seen on the real board: the same encoder exported on both
    # channels collapses to one stem, and the lesion-task headline (written
    # milliseconds later by a reaggregate sweep) silently replaced the
    # disco-task league values (E1 showed 0.485 instead of 0.320)
    out_root = str(tmp_path / "experiments")
    disco = str(tmp_path / "giles_replica_encoders" / "replica_headline.csv")
    lesion = str(tmp_path / "giles_replica_encoders_lesion" / "replica_headline.csv")
    _write_headline(disco, [{"representation": "E1_seed0_disco.npz",
                             "pehe_mean": 0.21, "pehe_paper_mean": 0.320}])
    _write_headline(lesion, [{"representation": "E1_seed0_lesion.npz",
                              "pehe_mean": 0.28, "pehe_paper_mean": 0.485}])
    now = time.time()
    os.utime(disco, (now - 100, now - 100))     # disco file is OLDER
    os.utime(lesion, (now, now))
    stems = _read_certified(out_root)
    assert stems["E1"] == [0.320]
    # a lesion-only export with no disco competitor still counts (E6 lesion arm)
    _write_headline(str(tmp_path / "giles_replica_e6" / "replica_headline.csv"),
                    [{"representation": "E6_E6_fusion_mode=lesion_seed0_lesion.npz",
                      "pehe_mean": 0.22, "pehe_paper_mean": 0.396}])
    stems = _read_certified(out_root)
    assert stems["E6_E6_fusion_mode=lesion"] == [0.396]


def test_single_variant_and_substring_matching():
    stems = {"E7b_dmvae": [0.335], "E6_both": [0.347],
             "E6_E6_fusion_mode=disconnectome": [0.317]}
    board = [
        {"arm": "A", "eid": "E7b", "label": "E7b", "T4.root_pehe.mean": 0.07},
        {"arm": "A", "eid": "E6", "label": "E6[fusion_mode=both]",
         "T4.root_pehe.mean": 0.07},
        {"arm": "A", "eid": "E6", "label": "E6[fusion_mode=disconnectome]",
         "T4.root_pehe.mean": 0.07},
    ]
    _attach_certified(board, stems)
    assert board[0]["certified.pehe_paper.mean"] == 0.335   # only-row fallback
    assert board[1]["certified.pehe_paper.mean"] == 0.347   # substring: both
    assert board[2]["certified.pehe_paper.mean"] == 0.317   # exact variant


def test_report_sorts_certified_first_and_survives_no_replica(tmp_path):
    out_root = str(tmp_path / "experiments")
    _write_runs(out_root, [
        {"kind": "E1/x", "label": "E1", "seed": 0, "T4.root_pehe": 0.09,
         "passed": True},
        {"kind": "E5/x", "label": "E5", "seed": 0, "T4.root_pehe": 0.05,
         "passed": True},
    ])
    # no giles_replica_* dirs at all: report must still build, all cells '-'
    paths = build_report(out_root)
    md = open(paths["md"]).read()
    assert "Certified rootPEHE" in md and "| - |" in md

    # E1 certified (0.320) but E5 not: E1 must lead despite the worse proxy
    _write_headline(str(tmp_path / "giles_replica_x" / "replica_headline.csv"),
                    [{"representation": "E1_seed0_disco.npz",
                      "pehe_mean": 0.21, "pehe_paper_mean": 0.320}])
    build_report(out_root)
    rows = list(csv.DictReader(open(paths["csv"])))
    assert [r["eid"] for r in rows] == ["E1", "E5"]
    assert rows[0]["certified.pehe_paper.mean"] == "0.32"
