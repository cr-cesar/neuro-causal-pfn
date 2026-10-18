"""H5: the effect-magnitude summary over replica_results files."""
import numpy as np
import pandas as pd
import pytest

from scripts import effect_magnitude_summary as ems


def _results(path, seed, slope, classifier="pfn", decomposed=True):
    rows = []
    for d in ("hearing", "language"):
        for k in range(2):
            r = {"representation": f"E1|E1|seed{seed}|chain|disconnectome|grp-giles-singles",
                 "deficit": d, "fold": k, "classifier": classifier, "learner": "in_context",
                 "pehe_paper": 0.4, "prescriptive_balacc": 0.8}
            if decomposed:
                r.update({"tau_hat_pos": slope, "tau_hat_zero": 0.0, "tau_hat_neg": -slope,
                          "tau_slope": slope, "mse_between": 0.12, "mse_within": 0.04,
                          "rescale_factor": 1 / slope, "pehe_paper_rescaled": 0.2})
            rows.append(r)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_run_label_and_encoder_seed():
    assert ems.run_label("a/outputs_perfold_h5/ckpt_s1/replica/E1/E1/seed2/x/replica_results.csv") == "ckpt_s1"
    assert ems.encoder_seed("E1|E1|seed2|chain|disconnectome|grp-giles-singles") == 2


def test_summary_averages_over_deficits_and_encoder_seeds(tmp_path, capsys):
    root = tmp_path / "outputs_perfold_h5"
    for seed, slope in ((0, 0.4), (1, 0.6)):
        _results(root / "ckpt" / "replica" / "E1" / "E1" / f"seed{seed}" / "d" / "replica_results.csv", seed, slope)
    _results(root / "cpfn" / "replica" / "E1" / "E1" / "seed0" / "d" / "replica_results.csv", 0, 0.9, "causalpfn")
    _results(root / "old" / "replica" / "E1" / "E1" / "seed0" / "d" / "replica_results.csv", 0, 0.5, decomposed=False)
    out = tmp_path / "status" / "h5.csv"
    runs = ems.main([str(root), "--out", str(out)])
    assert "1 skipped" in capsys.readouterr().out
    ck = runs[runs.run == "ckpt"].iloc[0]
    assert ck.n_seeds == 2 and ck.encoder_seeds == "0 1"
    assert ck.tau_slope == pytest.approx(0.5) and ck.tau_slope_sd == pytest.approx(0.1)
    assert ck.between_share == pytest.approx(0.75)
    assert set(runs.run) == {"ckpt", "cpfn"}
    per = pd.read_csv(out)
    assert len(per) == 3 and (tmp_path / "status" / "h5_by_run.csv").exists()
