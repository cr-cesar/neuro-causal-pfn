"""Job G (16 Nov): the launch's side of the new families. g4c_launch.sh is run
for real in a temporary checkout, with a stand-in ``python`` (the gate) and a
stand-in PFN trainer that record their arguments and the launch variables.
With two family weights the launch must do what the script before the change
did (commit 8fa4b74, taken from git history); with four it must pass the new
weights, the rule options and the families to keep absent."""
import os
import pathlib
import shutil
import subprocess
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_COMMIT = "8fa4b74ba1f0d17d7d5fa0d980ce969be6d8ac34"
BASH = shutil.which("bash")
NEEDS = [shutil.which(x) for x in ("bash", "git", "awk", "diff", "readlink")]
LAUNCH_VARS = ("CACHE", "SEED", "VARIANT", "FAMILY", "FAMILY_WEIGHTS", "CONTEXT_POOL", "PRIOR_HYPER", "GILES_HYPER",
               "CONTEXT_MAX", "N_QUERY", "STANDARDIZE", "ITERS", "BOTH_ARMS", "CATE_WEIGHT", "ARCH", "JOBNAME")
RECORD = '''{ echo "ARGS $2"; for v in %s; do
  if [ -n "${!v+x}" ]; then echo "$v=${!v}"; else echo "$v=<unset>"; fi; done; } > "$OUT/call_$1.txt"
''' % " ".join(LAUNCH_VARS)
INPUTS = ("FAMILY_W", "OBJECTIVE", "GENERIC_W", "RULE_GRADED", "RULE_ENROL", "RUN_ITERS", "EXCLUDE_NETWORKS",
          "PRIOR_REPAIR", "ALLOC", "S_SOURCE", "STD", "DISTINCT", "HEAD_RANGE", "CATE_W", "LAUNCHER", "ATLAS_SRC",
          "PRIOR_CHECK_ONLY", *LAUNCH_VARS)

pytestmark = pytest.mark.skipif(not all(NEEDS), reason="needs bash, git and the tools the launch script uses")


@pytest.fixture(scope="module")
def checkouts():
    """Two clean checkouts with the same stand-ins: one holds this tree's
    launch scripts, the other g4c_launch.sh as it was before the change."""
    d = pathlib.Path(tempfile.mkdtemp(prefix="ncp_launch_"))     # the launch splits CACHE on spaces
    if " " in str(d):
        shutil.rmtree(d, ignore_errors=True)
        pytest.skip("no temporary folder without spaces in its path")
    r = subprocess.run(["git", "show", f"{BASE_COMMIT}:scripts/audit_20261028/g4c_launch.sh"], cwd=ROOT,
                       capture_output=True)
    if r.returncode != 0:
        shutil.rmtree(d, ignore_errors=True)
        pytest.skip(f"commit {BASE_COMMIT[:7]} is not in this clone's history")
    venv = d / "venv" / "bin"; venv.mkdir(parents=True)
    (venv / "activate").write_text('export PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd):$PATH"\n',
                                   newline="\n")
    (venv / "python").write_text('#!/bin/bash\nname=$(basename "$1" .py); args="$*"; set -- "$name" "$args"\n'
                                 + RECORD, newline="\n")
    os.chmod(venv / "python", 0o755)
    pool = d / "pool.npz"; pool.write_bytes(b"synthetic pool")
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.org", "-c", "core.autocrlf=false"]
    repos = {}
    for name in ("new", "old"):
        repo = d / name
        (repo / "scripts" / "audit_20261028").mkdir(parents=True)
        (repo / "data" / "atlases").mkdir(parents=True)
        (repo / "data" / "atlases" / "functional_parcellation_2mm.nii.gz").write_bytes(b"synthetic atlas")
        for script in ("g4c_launch.sh", "launch_myriad.qsub.sh"):
            shutil.copy(os.path.join(ROOT, "scripts", "audit_20261028", script),
                        repo / "scripts" / "audit_20261028" / script)
        if name == "old":
            (repo / "scripts" / "audit_20261028" / "g4c_launch.sh").write_bytes(r.stdout)
        (repo / "scripts" / "train_pfn_myriad.qsub.sh").write_text(
            '#!/bin/bash\nset -- pfn ""\n' + RECORD + 'echo weights > "$OUT/pfn.pt"\n', newline="\n")
        for cmd in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "launch scripts"]):
            subprocess.run(git + cmd, cwd=repo, check=True, capture_output=True)
        repos[name] = repo
    yield {"dir": d, "venv": (d / "venv").as_posix(), "pool": pool.as_posix(), **repos}
    shutil.rmtree(d, ignore_errors=True)


def _launch(co, which, name, script="g4c_launch.sh", **env):
    """Run the launch of one checkout; returns (process, gate arguments, the
    variables the gate saw, the variables the trainer saw)."""
    out = co["dir"] / f"out_{which}_{name}"
    full = {k: v for k, v in os.environ.items() if k not in INPUTS}
    full.update(VENV=co["venv"], CACHE=co["pool"], OUT=out.as_posix(), **{k: str(v) for k, v in env.items()})
    r = subprocess.run([BASH, f"scripts/audit_20261028/{script}"], cwd=co[which], env=full, capture_output=True,
                       text=True)

    def call(stem):
        path = out / f"call_{stem}.txt"
        if not path.exists():
            return None, None
        lines = path.read_text().splitlines()
        return lines[0][5:], dict(line.split("=", 1) for line in lines[1:])

    gate_args, gate_env = call("prior_gate")
    _, train_env = call("pfn")
    return r, gate_args, gate_env, train_env


def _said(r, start):
    return [line for line in r.stdout.splitlines() if line.startswith(start)]


@pytest.mark.parametrize("name,env", [
    ("final", dict(OBJECTIVE="new", FAMILY_W="0 1", SEED=1, RUN_ITERS=250000)),      # final1m_launch.sh's call
    ("default", dict()),                                                             # G4c: 0.5 0.5
    ("quarter", dict(FAMILY_W="0.25 0.75", GENERIC_W=0, OBJECTIVE="new")),
    ("lno", dict(OBJECTIVE="new", FAMILY_W="0 1", EXCLUDE_NETWORKS="1+2", RUN_ITERS=125000)),
])
def test_two_family_weights_launch_exactly_as_before(checkouts, name, env):
    new = _launch(checkouts, "new", name, **env)
    old = _launch(checkouts, "old", name, **env)
    assert new[0].returncode == 0, new[0].stdout + new[0].stderr
    assert old[0].returncode == 0, old[0].stdout + old[0].stderr
    assert new[1] == old[1]                                  # the gate's command line, word for word
    assert new[2] == old[2] and new[3] == old[3]             # every launch variable, at the gate and at the trainer
    assert new[3] is not None and new[2] == new[3]
    for start in ("launch tag=", "effective composition"):
        assert _said(new[0], start) == _said(old[0], start) and len(_said(new[0], start)) == 1
    assert "--absent" not in new[1] and "rule_" not in new[1] and "rule_" not in new[3]["GILES_HYPER"]


def test_four_family_weights_reach_the_gate_and_the_trainer(checkouts):
    r, gate, genv, tenv = _launch(checkouts, "new", "four", OBJECTIVE="new", FAMILY_W="0 1 1 0")
    assert r.returncode == 0, r.stdout + r.stderr
    assert tenv["FAMILY_WEIGHTS"] == "0 1 1 0" == genv["FAMILY_WEIGHTS"] and tenv["FAMILY"] == "mixture"
    assert "rule_" not in tenv["GILES_HYPER"]                                # options unset: the generator's defaults
    assert " prior.family_weights=0:1:1:0 " in gate + " " and "--absent theory centroid --expect" in gate
    assert "prior.hyper_giles.rule_graded=UNSET prior.hyper_giles.rule_enrol=UNSET" in gate
    assert _said(r, "effective composition") == [
        "effective composition: theory 0.000, virtual-trial 0.250, volume 0.250, centroid 0.000, generic 0.500"]
    assert "launch tag=G4cobj_fw0-1-1-0 " in r.stdout and "family_w=0:1:1:0 " in r.stdout
    assert tenv["JOBNAME"] == "G4cobj_fw0-1-1-0-f0-s0"

    r, gate, genv, tenv = _launch(checkouts, "new", "opts", OBJECTIVE="new", FAMILY_W="0 1 0.5 0.5",
                                  RULE_GRADED="0.25", RULE_ENROL="0.2:1")
    assert r.returncode == 0, r.stdout + r.stderr
    assert tenv["GILES_HYPER"] == ("random_partitions=1 threshold=0.03:0.08 p_te=0.9:1 p_re=0:0.05 label_noise=0:0 "
                                   "rule_graded=0.25 rule_enrol=0.2:1") == genv["GILES_HYPER"]
    assert "prior.hyper_giles.rule_graded=0.25 prior.hyper_giles.rule_enrol=0.2:1" in gate
    assert "--absent theory --expect" in gate and "prior.family_weights=0:1:0.5:0.5" in gate
    assert "launch tag=G4cobj_fw0-1-0.5-0.5_rg0.25_re0.2-1 " in r.stdout

    # the latent-coordinate share left out; the two new families left out
    _, gate, _, tenv = _launch(checkouts, "new", "nogeneric", FAMILY_W="1 1 1 1", GENERIC_W=0)
    assert "--absent generic --expect" in gate and "generic_weight=0 " in tenv["PRIOR_HYPER"]
    _, gate, _, tenv = _launch(checkouts, "new", "zeros", OBJECTIVE="new", FAMILY_W="0 1 0 0")
    assert "--absent theory volume centroid --expect" in gate and tenv["FAMILY_WEIGHTS"] == "0 1 0 0"
    _, gate, _, _ = _launch(checkouts, "new", "all", FAMILY_W="1 1 1 1")
    assert "--absent" not in gate


@pytest.mark.parametrize("name,env,message", [
    ("three", dict(FAMILY_W="0 1 1"), "FAMILY_W must be two numbers"),
    ("five", dict(FAMILY_W="0 1 1 1 1"), "FAMILY_W must be two numbers"),
    ("word", dict(FAMILY_W="0 1 one 1"), "FAMILY_W must be two numbers"),
    ("zeros", dict(FAMILY_W="0 0 0 0"), "all zeros"),
    ("graded2", dict(FAMILY_W="0 1", RULE_GRADED="0.5"), "need four numbers"),
    ("enrol2", dict(RULE_ENROL="0.2:1"), "need four numbers"),
    ("graded", dict(FAMILY_W="0 1 1 1", RULE_GRADED="2"), "RULE_GRADED"),
    ("enrol", dict(FAMILY_W="0 1 1 1", RULE_ENROL="0.5"), "RULE_ENROL"),
    ("enrol_big", dict(FAMILY_W="0 1 1 1", RULE_ENROL="0.5:1.5"), "RULE_ENROL"),
])
def test_bad_values_stop_the_launch_before_the_gate(checkouts, name, env, message):
    r, gate, _, tenv = _launch(checkouts, "new", "bad_" + name, **env)
    assert r.returncode != 0 and message in r.stderr
    assert gate is None and tenv is None


def test_the_myriad_job_passes_four_weights_and_the_rule_options(checkouts):
    r, gate, genv, tenv = _launch(checkouts, "new", "myriad", script="launch_myriad.qsub.sh",
                                  LAUNCHER="g4c_launch.sh", OBJECTIVE="new", FAMILY_W="0:1:1:1", RULE_ENROL="0.2:1")
    assert r.returncode == 0, r.stdout + r.stderr
    assert tenv["FAMILY_WEIGHTS"] == "0 1 1 1" and tenv["GILES_HYPER"].endswith(" rule_enrol=0.2:1")
    assert "prior.family_weights=0:1:1:1" in gate and "--absent theory --expect" in gate
    r, _, _, tenv = _launch(checkouts, "new", "myriad2", script="launch_myriad.qsub.sh",
                            LAUNCHER="g4c_launch.sh", OBJECTIVE="new", FAMILY_W="0:1")
    assert r.returncode == 0 and tenv["FAMILY_WEIGHTS"] == "0 1"
