"""Hashing of cache file names and the sequential job runner (bundle for a
machine outside the cluster, 20 Oct)."""
import os
import re
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import hash_cache_names as hcn  # noqa: E402


def test_hash_cache_names_replaces_names_and_keeps_the_rest(tmp_path, capsys):
    src = tmp_path / "outputs" / "prior_cache"; src.mkdir(parents=True)
    names = np.array(["lesion_01_F_67.nii.gz", "lesion_02_M_71.nii"])
    np.savez(src / "kch.npz", files=names, Z=np.ones((2, 3)), overlaps=np.zeros((2, 4, 2)),
             images_dir=np.array("/home/user/Scratch/kch_bcbdisco"), channel=np.array("disconnectome"))
    sub = src / "fold_pools" / "run"; sub.mkdir(parents=True)
    np.savez(sub / "fold0.npz", files=names[:1], Z=np.ones((1, 3)),
             fold_file=np.array("/abs/path/outputs_perfold/E1/seed0/folds/fold0.npz"))
    salt = tmp_path / "salt"
    hcn.main(["--src", str(src), "--dst", str(tmp_path / "dst"), "--salt-file", str(salt)])
    assert oct(os.stat(salt).st_mode)[-3:] == "600" and len(salt.read_text().strip()) >= 32
    with np.load(tmp_path / "dst" / "prior_cache" / "kch.npz", allow_pickle=True) as z:
        hashed = [str(x) for x in z["files"]]
        assert all(re.fullmatch(r"[0-9a-f]{16}", h) for h in hashed) and hashed[0] != hashed[1]
        assert "F_67" not in " ".join(hashed) and str(z["images_dir"]) == "kch_bcbdisco"
        np.testing.assert_array_equal(z["Z"], np.ones((2, 3))); assert str(z["channel"]) == "disconnectome"
    with np.load(tmp_path / "dst" / "prior_cache" / "fold_pools" / "run" / "fold0.npz", allow_pickle=True) as z:
        assert str(z["files"][0]) == hashed[0]          # same salt, same key: rows still match across files
        assert str(z["fold_file"]) == "fold0.npz"
    assert "2 npz files copied" in capsys.readouterr().out
    # a different salt gives different keys
    hcn.main(["--src", str(src / "kch.npz"), "--dst", str(tmp_path / "dst2"), "--salt-file", str(tmp_path / "salt2")])
    with np.load(tmp_path / "dst2" / "kch.npz", allow_pickle=True) as z:
        assert str(z["files"][0]) != hashed[0]


def test_run_jobs_runs_skips_and_reports(tmp_path):
    repo = tmp_path / "repo"; (repo / "scripts").mkdir(parents=True); (repo / ".venv" / "bin").mkdir(parents=True)
    (repo / ".venv" / "bin" / "python").write_text("#!/bin/sh\n"); os.chmod(repo / ".venv" / "bin" / "python", 0o755)
    (repo / "scripts" / "run_jobs.sh").write_text(open(os.path.join(ROOT, "scripts", "run_jobs.sh")).read())
    fake = repo / "scripts" / "fake_train.sh"
    fake.write_text('#!/bin/bash\nset -e\n[ "$FAIL" = 1 ] && { echo boom; exit 1; }\nmkdir -p "$OUT"; echo "$SEED $PRIOR_HYPER" > "$OUT/pfn.pt"\n')
    os.chmod(fake, 0o755)
    (repo / "outputs" / "done").mkdir(parents=True); (repo / "outputs" / "done" / "pfn.pt").write_text("x")
    jobs = repo / "jobs.txt"
    jobs.write_text('# comment\n\n'
                    'OUT=outputs/a SEED=1 PRIOR_HYPER="two_arms=1 shape_weights=0.4:0.4:0.2" scripts/fake_train.sh\n'
                    'OUT=outputs/done SEED=2 scripts/fake_train.sh\n'
                    'OUT=outputs/b SEED=3 FAIL=1 scripts/fake_train.sh\n')
    r = subprocess.run(["bash", "scripts/run_jobs.sh", "jobs.txt"], cwd=repo, capture_output=True, text=True)
    out = r.stdout
    assert r.returncode == 1 and "jobs: 3, ok 1, skipped 1, failed 1" in out
    assert (repo / "outputs" / "a" / "pfn.pt").read_text().strip() == "1 two_arms=1 shape_weights=0.4:0.4:0.2"
    assert "skip, done already" in out and "FAILED b" in out
    assert "boom" in (repo / "logs" / "3_b.log").read_text()
    # second pass: the finished job is skipped too
    r2 = subprocess.run(["bash", "scripts/run_jobs.sh", "jobs.txt"], cwd=repo, capture_output=True, text=True)
    assert "ok 0, skipped 2, failed 1" in r2.stdout


def test_run_jobs_in_another_checkout_with_job_names(tmp_path):
    """WORKDIR runs the lines in another repository (the per-fold one) with the
    given environment; JOBNAME names the log."""
    main = tmp_path / "main" / "scripts"; main.mkdir(parents=True)
    (main / "run_jobs.sh").write_text(open(os.path.join(ROOT, "scripts", "run_jobs.sh")).read())
    other = tmp_path / "perfold" / "qsub"; other.mkdir(parents=True)
    (other / "train_local.sh").write_text('#!/bin/bash\necho "fold $FOLD in $PWD with $(command -v python)"\n')
    env = tmp_path / "env" / "bin"; env.mkdir(parents=True)
    (env / "python").write_text("#!/bin/sh\n"); os.chmod(env / "python", 0o755)
    jobs = tmp_path / "jobs.txt"
    jobs.write_text("JOBNAME=e2r0s0f0 EID=E2 FOLD=0 bash qsub/train_local.sh   # E2[w_dice=0.1]\n"
                    "JOBNAME=e2r0s0f1 EID=E2 FOLD=1 bash qsub/train_local.sh\n")
    r = subprocess.run(["bash", str(main / "run_jobs.sh"), str(jobs)], cwd=tmp_path,
                       env={**os.environ, "WORKDIR": str(tmp_path / "perfold"), "VENV": str(tmp_path / "env")},
                       capture_output=True, text=True)
    assert r.returncode == 0 and "jobs: 2, ok 2, skipped 0, failed 0" in r.stdout, r.stdout + r.stderr
    log = (tmp_path / "perfold" / "logs" / "2_e2r0s0f1.log").read_text()
    assert "fold 1 in " + str(tmp_path / "perfold") in log and "ok    e2r0s0f1" in r.stdout
