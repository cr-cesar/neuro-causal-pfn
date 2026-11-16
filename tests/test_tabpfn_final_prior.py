"""Job D (16 Nov): TabPFN continued on the final prior. The fine-tuning script
can take its prior from the launch environment (the cfg the PFN training
builds), the launch reaches it through g4c_launch.sh with the final model's
variables, one model per seed, and the configuration and provenance are
written beside the checkpoint. Synthetic arrays only; no GPU, no tabpfn
training (the loop itself needs tabpfn >= 9.2 and a card)."""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import finetune_tabpfn_prior as ft  # noqa: E402

# the prior variables scripts/audit_20261028/g4c_launch.sh exports under
# final1m_launch.sh's call (OBJECTIVE=new FAMILY_W="0 1"); the launch test
# below checks them against the scripts themselves
FINAL_PRIOR_ENV = {
    "FAMILY": "mixture", "FAMILY_WEIGHTS": "0 1", "CONTEXT_POOL": "susceptible", "CONTEXT_MAX": "3000",
    "GILES_HYPER": "random_partitions=1 threshold=0.03:0.08 p_te=0.9:1 p_re=0:0.05 label_noise=0:0",
    "PRIOR_HYPER": "two_arms=1 effect_scale=0.2:1 d_source=network s_source=genetics_1 s_b_source=genetics_2 "
                   "effect_headroom=1 generic_weight=0.5 binary_outcome=1 generic_binary=1",
}
# every variable that reaches the prior's cfg (launch_cfg.cfg_from_env); the
# ones the final launch leaves unset must stay unset on the TabPFN path too
PRIOR_VARS = ("CACHE", "CACHE_ANCHOR_WEIGHT", "SEED", "VARIANT", "FAMILY", "FAMILY_WEIGHTS", "CONTEXT_POOL",
              "PRIOR_HYPER", "GILES_HYPER", "CONTEXT_MAX", "N_QUERY", "AUGMENT", "CONTEXT_REPLACE", "STANDARDIZE")


def _synthetic_cache(path, m=400, K=8, d=8, seed=0):
    """A pool cache of random numbers with the keys of
    scripts/build_prior_cache.py and its extra anatomy channels."""
    rng = np.random.default_rng(seed)
    cover = lambda *shape: (rng.uniform(0, 0.3, size=shape) * (rng.uniform(size=shape) < 0.5)).astype(np.float32)
    gen = cover(m, K, 2)
    np.savez(path, overlaps=gen, overlaps_genetics=gen, overlaps_receptor=cover(m, K, 2),
             overlaps_network=cover(m, K), centroids=rng.normal(size=(m, 3)).astype(np.float32),
             volumes=rng.lognormal(size=m), Z=rng.normal(size=(m, d)).astype(np.float32))
    return str(path)


def test_the_old_options_and_defaults_are_unchanged():
    ap = ft.build_parser()
    a = ap.parse_args(["--cache", "pool/fold0.npz", "--out", "o"])
    ft.check_args(ap, a)
    assert (a.steps, a.lr, a.processes_per_draw, a.n_context, a.n_query, a.n_estimators) == (2000, 1e-5, 4, 2000, 256, 2)
    assert (a.weight_decay, a.grad_clip, a.query_ratio, a.learner, a.save_every) == (0.01, 1.0, 0.2, "s", 500)
    assert a.prior_from_env is False and a.seed is None and a.family == "theory"
    # the record inside the checkpoint keeps its fields
    rec = ft.checkpoint_fields(7, {"prior": {}}, a)
    assert rec == {"steps": 7, "learner": "s", "n_context": 2000, "prior_hyper": a.prior_hyper, "family": "theory",
                   "family_weights": None, "giles_hyper": "", "cache": "fold0.npz"}
    for bad in (["--out", "o"],                                              # neither a cache nor the environment
                ["--out", "o", "--cache", "c", "--pool-md5", "abc"],         # digests belong to the environment mode
                ["--out", "o", "--prior-from-env", "--cache", "c"],          # both
                ["--out", "o", "--prior-from-env", "--giles-hyper", "p_te=1:1"]):
        with pytest.raises(SystemExit):
            ft.check_args(ap, ap.parse_args(bad))
    ft.check_args(ap, ap.parse_args(["--out", "o", "--prior-from-env", "--pool-md5", "abc"]))


def test_prior_from_env_hands_the_launch_cfg_to_the_pfn_builder(monkeypatch):
    from neurocausalpfn.train.launch_cfg import cfg_from_env
    import neurocausalpfn.train.train_pfn as tp

    seen = {}

    def fake_build(cfg, seed_offset=0):
        seen["cfg"], seen["offset"] = cfg, seed_offset
        return object(), 50, True

    monkeypatch.setattr(tp, "_build_prior", fake_build)
    env = dict(FINAL_PRIOR_ENV, CACHE="pools/kch.npz", SEED="2")
    _, d_x, cfg = ft.prior_from_env(env)
    assert d_x == 50 and seen["offset"] == 0 and cfg is seen["cfg"] and cfg == cfg_from_env(env)
    pr = cfg["prior"]
    assert cfg["seed"] == 2 and pr["family"] == "mixture" and pr["family_weights"] == [0.0, 1.0]
    assert pr["hyper"]["generic_weight"] == 0.5 and pr["hyper"]["generic_binary"] == 1.0
    assert pr["hyper_giles"] == {"random_partitions": 1.0, "threshold": [0.03, 0.08], "p_te": [0.9, 1.0],
                                 "p_re": [0.0, 0.05], "label_noise": [0.0, 0.0]}
    assert pr["cache"] == "pools/kch.npz" and pr["context_pool"] == "susceptible"
    assert cfg["pfn"]["n_query"] == 64 and cfg["pfn"]["context_max"] == 3000
    with pytest.raises(SystemExit):
        ft.prior_from_env(FINAL_PRIOR_ENV)                                   # no CACHE


def test_prior_from_env_draws_what_the_pfn_training_draws(tmp_path, monkeypatch):
    """Same environment, same seed: the script's prior and the PFN trainer's
    give the same simulated trials, and they are the final prior's (threshold
    coverage rule and latent-coordinate rule, binary outcomes)."""
    from neurocausalpfn.train.launch_cfg import cfg_from_env
    from neurocausalpfn.train.train_pfn import _build_prior

    monkeypatch.chdir(tmp_path)                                              # no data/atlases: the synthetic atlas
    _synthetic_cache(tmp_path / "pool.npz")
    env = dict(FINAL_PRIOR_ENV, CACHE="pool.npz", SEED="1")                  # CACHE is split on spaces: a relative path
    prior, d_x, cfg = ft.prior_from_env(env)
    ref, _, _ = _build_prior(cfg_from_env(env))
    assert d_x == 8 and prior.min_slice == 72
    families = set()
    for n_ctx in (96, 200):
        a, b = prior.sample_batch(4, n_context=n_ctx), ref.sample_batch(4, n_context=n_ctx)
        for k in ("Xc", "Tc", "Yc", "Xq", "Tq", "mu_q", "mu0", "mu1"):
            np.testing.assert_array_equal(a[k], b[k])
        assert a["Xq"].shape == (4, 64, 8) and set(np.unique(a["Yc"])) <= {0.0, 1.0}
        families |= {p["family"] for p in a["processes"]}
    assert "theory" not in families and all(f == "giles" or f.startswith("generic-") for f in families)
    # the batches become TabPFN datasets as before
    sets = ft.datasets_from_batch(a, np.random.default_rng(0), learner="s")
    assert sets and all(X.shape == (200 + 64, 9) for X, _ in sets)


def test_provenance_names_the_code_the_pool_and_the_starting_weights(tmp_path):
    pool = tmp_path / "pool.npz"; pool.write_bytes(b"synthetic pool bytes")
    weights = tmp_path / "base.ckpt"; weights.write_bytes(b"synthetic weights")
    import hashlib

    md5 = hashlib.md5(b"synthetic pool bytes").hexdigest()
    assert ft.file_md5(str(pool)) == md5
    cfg = {"seed": 1, "pfn": {"n_query": 64, "context_max": 3000},
           "prior": {"family": "mixture", "family_weights": [0.0, 1.0], "cache": str(pool)}}
    pools = ft.pool_record(cfg, md5[:12])
    assert pools == [{"path": str(pool), "bytes": 20, "md5": md5}]
    with pytest.raises(SystemExit):
        ft.pool_record(cfg, "0" * 12)                                        # another pool
    with pytest.raises(SystemExit):
        ft.pool_record({"prior": {"cache": [str(pool), str(pool)]}}, md5[:12])   # the digest names one pool
    base = ft.base_checkpoint_record(str(weights))
    assert base["file"] == "base.ckpt" and base["md5"] == hashlib.md5(b"synthetic weights").hexdigest()
    assert "tabpfn" in base                                                  # the package version, None if absent
    assert ft.base_checkpoint_record(None)["model_path_arg"] is None        # the default: never raises
    ap = ft.build_parser()
    args = ap.parse_args(["--prior-from-env", "--out", str(tmp_path), "--pool-md5", md5[:12], "--atlas-md5", "abc",
                          "--steps", "30", "--lr", "2e-5", "--model-path", str(weights)])
    args.seed, args.n_query = 1, 64
    prov = ft.provenance_record(args, cfg, pools, base, step=30, finished=True)
    assert set(prov) >= {"git_commit", "git_dirty", "launch_env", "torch", "numpy", "python", "pool", "tabpfn",
                         "finetune", "cfg", "steps_done", "finished"}
    assert prov["pool"][0]["md5"] == md5 and prov["pool_md5_expected"] == md5[:12]
    assert prov["atlas_md5_reported_by_launch"] == "abc" and prov["finished"] is True and prov["steps_done"] == 30
    assert prov["finetune"]["steps"] == 30 and prov["finetune"]["lr"] == 2e-5 and prov["finetune"]["seed"] == 1
    assert prov["cfg"]["prior"]["family_weights"] == [0.0, 1.0]
    json.dumps(prov, default=str)
    # the record inside the checkpoint, and the files beside it
    rec = ft.checkpoint_fields(30, cfg, args)
    assert rec["prior"] == cfg["prior"] and rec["seed"] == 1 and rec["n_query"] == 64 and "git_commit" in rec

    calls = {}
    args.provenance = lambda step, finished: ft.provenance_record(args, cfg, pools, base, step, finished)
    save = lambda model, save_path, additional_fields: calls.update(path=save_path, fields=additional_fields)
    ft._save("estimator", str(tmp_path), 10, [{"step": 10, "loss": 0.5, "lr": 1e-5}], cfg, args, save)
    on_disk = json.load(open(tmp_path / "provenance.json"))
    assert on_disk["finished"] is False and on_disk["steps_done"] == 10      # an interrupted run is not finished
    ft._save("estimator", str(tmp_path), 30, [], cfg, args, save, finished=True)
    assert json.load(open(tmp_path / "provenance.json"))["finished"] is True
    assert calls["fields"]["neuro_prior_finetune"]["steps"] == 30
    assert "provenance" not in json.load(open(tmp_path / "history.json"))["config"]
    # without the environment mode nothing new is written
    old = ap.parse_args(["--cache", str(pool), "--out", str(tmp_path / "old")])
    os.makedirs(tmp_path / "old"); old.provenance = None
    ft._save("estimator", str(tmp_path / "old"), 5, [], cfg, old, save)
    assert sorted(os.listdir(tmp_path / "old")) == ["history.json"]


# ----------------------------------------------------------------- the launch
BASH = shutil.which("bash")
NEEDS = [shutil.which(x) for x in ("bash", "git", "md5sum", "find", "xargs", "awk", "diff", "readlink")]
RECORD = '''{ echo "ARGS $*"; for v in %s OUT JOBNAME ITERS BOTH_ARMS ARCH; do
  if [ -n "${!v+x}" ]; then echo "$v=${!v}"; else echo "$v=<unset>"; fi; done; } > "$OUT/call_$1.txt"
''' % " ".join(PRIOR_VARS)


@pytest.fixture
def launch_dir(tmp_path):
    """A folder without spaces in its path: the launch scripts split CACHE on
    spaces (several pools), so a pool under such a path cannot be passed."""
    if " " not in str(tmp_path):
        yield tmp_path
        return
    import pathlib
    import tempfile

    d = pathlib.Path(tempfile.mkdtemp(prefix="ncp_launch_"))
    if " " in str(d):
        pytest.skip("no temporary folder without spaces in its path")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _launch_repo(tmp_path):
    """A clean git checkout holding the three launch scripts and stand-ins for
    everything they start: a ``python`` that records its arguments and the
    launch variables, and a PFN training script that does the same."""
    repo = tmp_path / "repo"
    (repo / "scripts" / "audit_20261028").mkdir(parents=True)
    for name in ("g4c_launch.sh", "final1m_launch.sh", "tabpfn_final_launch.sh"):
        shutil.copy(os.path.join(ROOT, "scripts", "audit_20261028", name), repo / "scripts" / "audit_20261028" / name)
    (repo / "scripts" / "train_pfn_myriad.qsub.sh").write_text(
        "#!/bin/bash\nset -- pfn\n" + RECORD + 'echo weights > "$OUT/pfn.pt"\n', newline="\n")
    venv = tmp_path / "venv" / "bin"; venv.mkdir(parents=True)
    (venv / "activate").write_text('export PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd):$PATH"\n',
                                   newline="\n")
    # the stand-in for python: the gate only records its call; the fine-tuning
    # script records it and leaves a finished checkpoint
    (venv / "python").write_text(
        "#!/bin/bash\nname=$(basename \"$1\" .py); args=\"$*\"; set -- \"$name\" \"$args\"\n"
        + RECORD.replace('echo "ARGS $*"', 'echo "ARGS $2"')
        + 'if [ "$1" = finetune_tabpfn_prior ]; then echo weights > "$OUT/tabpfn_ft.pth"; '
          'echo \'{"finished": true}\' > "$OUT/provenance.json"; fi\n', newline="\n")
    os.chmod(venv / "python", 0o755)
    atlas = tmp_path / "atlas"; (atlas / "2mm_parcellations" / "genetics").mkdir(parents=True)
    (atlas / "functional_parcellation_2mm.nii.gz").write_bytes(b"synthetic atlas")
    (atlas / "2mm_parcellations" / "genetics" / "1_net.nii.gz").write_bytes(b"synthetic halves")
    pool = tmp_path / "pool.npz"; pool.write_bytes(b"synthetic pool")
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.org", "-c", "core.autocrlf=false"]
    for cmd in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "launch scripts"]):
        subprocess.run(git + cmd, cwd=repo, check=True, capture_output=True)
    return repo, venv.parent, atlas, pool


def _run(repo, script, **env):
    full = {**os.environ, **{k: str(v) for k, v in env.items()}}
    for k in ("FAMILY_W", "OBJECTIVE", "GENERIC_W", "RUN_TRAINER", "TABPFN_FT_ARGS", "SEEDS", "STEPS", "LR",
              "FT_BATCH", "N_CONTEXT", "N_EST", "MODEL_PATH", "PARALLEL", "POOL_MD5", "ATLAS_MD5", *PRIOR_VARS):
        if k not in env:
            full.pop(k, None)
    return subprocess.run([BASH, f"scripts/audit_20261028/{script}"], cwd=repo, env=full, capture_output=True,
                          text=True)


def _call(path):
    lines = open(path).read().splitlines()
    return lines[0][5:], dict(line.split("=", 1) for line in lines[1:])


@pytest.mark.skipif(not all(NEEDS), reason="needs bash, git and the coreutils the launch scripts use")
def test_the_launch_trains_tabpfn_on_the_final_models_prior_one_model_per_seed(launch_dir):
    tmp_path = launch_dir
    import hashlib

    repo, venv, atlas, pool = _launch_repo(tmp_path)
    # the atlas digest as final1m_launch.sh computes it (md5sum's line format differs between platforms)
    atlas_md5 = subprocess.run(
        [BASH, "-c", "find functional_parcellation_2mm.nii.gz 2mm_parcellations -type f | LC_ALL=C sort "
                     "| xargs -d '\\n' md5sum | md5sum | cut -c1-12"],
        cwd=atlas, capture_output=True, text=True, check=True).stdout.strip()
    common = dict(VENV=venv.as_posix(), POOL=pool.as_posix(), ATLAS=atlas.as_posix(),
                  POOL_MD5=hashlib.md5(b"synthetic pool").hexdigest()[:12], ATLAS_MD5=atlas_md5)

    # the final model's launch, with the stand-in PFN trainer
    o_pfn = tmp_path / "out_pfn"
    r = _run(repo, "final1m_launch.sh", O=o_pfn.as_posix(), **common)
    assert r.returncode == 0, r.stdout + r.stderr
    pfn_args, pfn_env = _call(o_pfn / "s1" / "call_pfn.txt")
    for k, v in FINAL_PRIOR_ENV.items():
        assert pfn_env[k] == v, k                                            # the constants of this file are the launch's

    # job D's launch
    o = tmp_path / "out_tabpfn"
    r = _run(repo, "tabpfn_final_launch.sh", O=o.as_posix(), **common)
    assert r.returncode == 0, r.stdout + r.stderr
    assert sorted(os.listdir(o)) == ["logs", "s0", "s1", "s2"]              # one model per seed, no fold anywhere
    for s in (0, 1, 2):
        d = o / f"s{s}"
        assert not (d / "call_pfn.txt").exists() and (d / "tabpfn_ft.pth").exists()
        gate_args, gate_env = _call(d / "call_prior_gate.txt")
        args, env = _call(d / "call_finetune_tabpfn_prior.txt")
        assert "prior.family_weights=0:1" in gate_args and "prior.hyper.generic_weight=0.5" in gate_args
        assert args == (f"scripts/finetune_tabpfn_prior.py --prior-from-env --out {d.as_posix()} --steps 2000 "
                        f"--lr 1e-5 --processes-per-draw 4 --n-context 1500 --n-estimators 1 "
                        f"--pool-md5 {common['POOL_MD5']} --atlas-md5 {common['ATLAS_MD5']}")
        assert env == gate_env and env["SEED"] == str(s) and env["CACHE"] == pool.as_posix()
    # the prior variables are the final model's, one by one (same seed: s1)
    _, env1 = _call(o / "s1" / "call_finetune_tabpfn_prior.txt")
    for k in PRIOR_VARS:
        assert env1[k] == pfn_env[k], k
    assert env1["N_QUERY"] == "<unset>" and env1["STANDARDIZE"] == "<unset>" and env1["VARIANT"] == "<unset>"

    # options from the environment; finished seeds are skipped, a new one trains
    r = _run(repo, "tabpfn_final_launch.sh", O=o.as_posix(), SEEDS="1 5", STEPS=300, LR="3e-6", FT_BATCH=2,
             N_CONTEXT=800, N_EST=2, MODEL_PATH=pool.as_posix(), **common)
    assert r.returncode == 0, r.stdout + r.stderr
    assert _call(o / "s1" / "call_finetune_tabpfn_prior.txt")[0].endswith(f"--atlas-md5 {common['ATLAS_MD5']}")
    assert _call(o / "s5" / "call_finetune_tabpfn_prior.txt")[0].endswith(
        f"--steps 300 --lr 3e-6 --processes-per-draw 2 --n-context 800 --n-estimators 2 "
        f"--pool-md5 {common['POOL_MD5']} --atlas-md5 {common['ATLAS_MD5']} --model-path {pool.as_posix()}")

    # refusals: another pool, another atlas, a bad option, a tree with local changes
    for bad in (dict(POOL_MD5="0" * 12), dict(ATLAS_MD5="0" * 12), dict(STEPS="2k"), dict(LR="fast")):
        o_bad = tmp_path / ("bad_" + next(iter(bad)))
        r = _run(repo, "tabpfn_final_launch.sh", O=o_bad.as_posix(), **{**common, **bad})
        assert r.returncode != 0 and not (o_bad / "s0").exists(), bad
    with open(repo / "scripts" / "audit_20261028" / "g4c_launch.sh", "a") as f:
        f.write("# edited\n")
    r = _run(repo, "tabpfn_final_launch.sh", O=(tmp_path / "dirty").as_posix(), **common)
    assert r.returncode != 0 and "local changes" in r.stderr and not (tmp_path / "dirty" / "s0").exists()


@pytest.mark.skipif(not all(NEEDS), reason="needs bash, git and the coreutils the launch scripts use")
def test_without_the_new_variable_the_launch_trains_the_pfn_as_before(launch_dir):
    tmp_path = launch_dir
    repo, venv, atlas, pool = _launch_repo(tmp_path)
    (repo / "data").mkdir(); shutil.copytree(atlas, repo / "data" / "atlases")
    out = tmp_path / "run"
    r = _run(repo, "g4c_launch.sh", VENV=venv.as_posix(), CACHE=pool.as_posix(), OUT=out.as_posix())
    assert r.returncode == 0, r.stdout + r.stderr
    assert (out / "pfn.pt").exists() and not (out / "call_finetune_tabpfn_prior.txt").exists()
    assert "launch tag=G4c " in r.stdout                                     # the run name is G4c's, untagged
    r = _run(repo, "g4c_launch.sh", VENV=venv.as_posix(), CACHE=pool.as_posix(), OUT=out.as_posix(),
             RUN_TRAINER="xgboost")
    assert r.returncode != 0 and "RUN_TRAINER" in r.stderr
