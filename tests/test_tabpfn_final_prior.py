"""Job D (16 Nov): TabPFN continued on the final prior. The fine-tuning script
can take its prior from the launch environment (the cfg the PFN training
builds), the launch reaches it through g4c_launch.sh with the final model's
variables, one model per seed, and the configuration and provenance are
written beside the checkpoint. Synthetic arrays only; no GPU, no tabpfn
training (the loop itself needs tabpfn >= 9.2 and a card)."""
import contextlib
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import types

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_COMMIT = "8fa4b74ba1f0d17d7d5fa0d980ce969be6d8ac34"      # the code before job D
BEFORE_GUARD = "6a7c6c34b1198e1d9b775b177bcae58fae217ecd"     # job D before L2-SP and the drawn context (25 Nov)
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


def _from_base(path: str, commit: str = BASE_COMMIT) -> bytes:
    """A file as it was before job D (or at ``commit``), from git history. A
    clone without that commit cannot make the comparison, and that is a
    failure, not a skip."""
    r = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=ROOT, capture_output=True)
    if r.returncode != 0:
        pytest.fail(f"commit {commit[:7]} is not in this clone's history (fetch it): the comparison with the "
                    f"code before the change cannot run")
    return r.stdout


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
                ["--out", "o", "--cache", "c", "--model-md5", "abc"],
                ["--out", "o", "--prior-from-env", "--cache", "c"],          # both
                ["--out", "o", "--prior-from-env", "--giles-hyper", "p_te=1:1"],
                ["--out", "o", "--prior-from-env", "--pool-md5", "abc"],     # the atlas digest is missing
                ["--out", "o", "--prior-from-env", "--atlas-md5", "abc"]):   # the pool digest is missing
        with pytest.raises(SystemExit):
            ft.check_args(ap, ap.parse_args(bad))
    ft.check_args(ap, ap.parse_args(["--out", "o", "--prior-from-env", "--pool-md5", "abc", "--atlas-md5", "abc"]))


ENV_OK = ["--out", "o", "--prior-from-env", "--pool-md5", "abc", "--atlas-md5", "abc"]


@pytest.mark.parametrize("option", [
    ["--family", "mixture"], ["--family-weights", "0", "1"], ["--prior-hyper", "generic_weight=0"],
    ["--giles-hyper", "p_te=1:1"], ["--n-query", "32"], ["--context-pool", "all"], ["--atlas-dir", "elsewhere"],
    ["--cache", "pool.npz"],
    # written with the very value the parser would give them: still written, still refused
    ["--family", "theory"], ["--n-query", "256"], ["--context-pool", "susceptible"], ["--atlas-dir", "data/atlases"],
    ["--n-query=256"], ["--giles-hyper", ""],
])
def test_prior_from_env_refuses_every_prior_option_instead_of_ignoring_it(option, capsys):
    ap = ft.build_parser()
    argv = ENV_OK + option
    with pytest.raises(SystemExit) as e:
        ft.check_args(ap, ap.parse_args(argv), argv)
    assert e.value.code == 2 and option[0].split("=")[0] in capsys.readouterr().err
    with pytest.raises(SystemExit):                                         # and through the entry point
        ft.main(argv)
    # the options that are not the prior's stay free
    free = ENV_OK + ["--steps", "10", "--lr", "2e-5", "--n-context", "1500", "--n-estimators", "1", "--seed", "1",
                     "--processes-per-draw", "4", "--model-version", "v3.5"]
    ft.check_args(ap, ap.parse_args(free), free)
    assert ft.explicit_options(ap, free) == {"out", "prior_from_env", "pool_md5", "atlas_md5", "steps", "lr",
                                             "n_context", "n_estimators", "seed", "processes_per_draw",
                                             "model_version"}


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
    assert calls["path"].endswith("tabpfn_ft.partial.pth")                   # and is not named as a finished one
    ft._save("estimator", str(tmp_path), 30, [], cfg, args, save, finished=True)
    assert json.load(open(tmp_path / "provenance.json"))["finished"] is True
    assert calls["fields"]["neuro_prior_finetune"]["steps"] == 30 and calls["path"].endswith("tabpfn_ft.pth")
    assert on_disk["torch_seed"] == 1
    assert "provenance" not in json.load(open(tmp_path / "history.json"))["config"]
    # without the environment mode nothing new is written
    old = ap.parse_args(["--cache", str(pool), "--out", str(tmp_path / "old")])
    os.makedirs(tmp_path / "old"); old.provenance = None
    ft._save("estimator", str(tmp_path / "old"), 5, [], cfg, old, save)
    assert sorted(os.listdir(tmp_path / "old")) == ["history.json"]
    assert calls["path"].endswith("tabpfn_ft.pth")                           # one name at every save, as before


# ------------------------------------------- finetune() with stand-ins for tabpfn
# The loop needs tabpfn >= 9.2 and a card. Here every tabpfn call the script
# makes is replaced by a stand-in (a two-layer torch model, a splitter, a
# saver), so these tests run the script's own lines: what it builds before the
# loop, what it refuses, what it seeds and what it writes. They say nothing
# about tabpfn's API.
class _FakeClassifier:
    made = []

    def __init__(self, **kw):
        self.kw = kw
        _FakeClassifier.made.append(self)

    def _initialize_model_variables(self):
        import torch

        self.model_ = torch.nn.Sequential(torch.nn.Dropout(0.5), torch.nn.Linear(9, 2))   # 8 latents + treatment

    def fit_from_preprocessed(self, X_context, y_context, cat_indices, configs, performance_options=None):
        self.context = X_context

    def forward(self, X_query, return_raw_logits=True):
        x = X_query[0]
        return self.model_(x).reshape(len(x), 1, 1, 2)                      # queries, batch, estimators, classes


@pytest.fixture
def fake_tabpfn(monkeypatch, tmp_path):
    """Stand-ins for the tabpfn modules scripts/finetune_tabpfn_prior.py
    imports; ``state`` holds what the package is set to and what was saved."""
    import torch

    cache = tmp_path / "tabpfn_cache"; cache.mkdir()
    (cache / "tabpfn-v3.5-classifier.ckpt").write_bytes(b"synthetic v3.5 weights")
    state = {"saved": [], "resolved": [], "cache": cache,
             "settings": types.SimpleNamespace(tabpfn=types.SimpleNamespace(model_version="v3.5"))}
    _FakeClassifier.made = []

    def module(name, **attrs):
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        monkeypatch.setitem(sys.modules, name, m)

    def chunks(calling_instance, X_raw, y_raw, split_fn, **kw):
        state.setdefault("tables", []).append([len(X) for X in X_raw])      # rows of each table of this draw
        return list(zip(X_raw, y_raw))

    def collator(items):
        (X, y), = items
        X, y, n = torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y), int(0.8 * len(y))
        return types.SimpleNamespace(X_context=[X[:n]], y_context=[y[:n]], X_query=[X[n:]], y_query=y[n:].numpy(),
                                     cat_indices=None, configs=None)

    def save(model, save_path, additional_fields):
        state["saved"].append((os.path.basename(save_path), additional_fields))
        with open(save_path, "w") as f:
            f.write("stand-in weights")

    def resolve(model_path, which, version):
        state["resolved"].append((model_path, which, version))
        path = cache / f"tabpfn-{version}-classifier.ckpt"
        return [path], [cache], [path.name], which

    module("tabpfn", TabPFNClassifier=_FakeClassifier)
    module("tabpfn.architectures")
    module("tabpfn.architectures.interface", PerformanceOptions=lambda **kw: kw)
    module("tabpfn.finetuning")
    module("tabpfn.finetuning._torch_compat", GradScaler=object, autocast=lambda enabled: contextlib.nullcontext(),
           sdpa_kernel_context=contextlib.nullcontext)
    module("tabpfn.finetuning.data_util", get_preprocessed_dataset_chunks=chunks, meta_dataset_collator=collator)
    module("tabpfn.finetuning.finetuned_base", _parameters_unused_by_task=lambda model, task: [],
           _ratio_ctx_query_split=lambda *a, **kw: None)
    module("tabpfn.finetuning.train_util",
           get_and_init_optimizer=lambda params, lr, wd, _, device: torch.optim.AdamW(params, lr=lr, weight_decay=wd),
           get_cosine_schedule_with_warmup=lambda total_steps, warmup_steps, warmup_only: (lambda step: 1.0))
    module("tabpfn.model_loading", save_tabpfn_model=save, resolve_model_path=resolve)
    module("tabpfn.settings", settings=state["settings"])
    return state


@pytest.fixture
def final_prior_launch(monkeypatch, tmp_path):
    """The final prior's launch variables in the environment, on a synthetic
    pool in the working folder; returns the options every run needs."""
    from neurocausalpfn.train.launch_cfg import LAUNCH_VARS

    monkeypatch.chdir(tmp_path)
    for k in LAUNCH_VARS:
        monkeypatch.delenv(k, raising=False)
    for k, v in dict(FINAL_PRIOR_ENV, CACHE="pool.npz", SEED="0").items():   # CACHE is split on spaces: relative
        monkeypatch.setenv(k, v)
    _synthetic_cache(tmp_path / "pool.npz")
    md5 = hashlib.md5((tmp_path / "pool.npz").read_bytes()).hexdigest()
    return ["--prior-from-env", "--pool-md5", md5[:12], "--atlas-md5", "abcdef012345", "--device", "cpu",
            "--steps", "6", "--save-every", "4", "--log-every", "2", "--n-context", "96", "--processes-per-draw", "2",
            "--n-estimators", "1"]


def _losses(out):
    return [h["loss"] for h in json.load(open(os.path.join(out, "history.json")))["history"]]


def test_the_launch_environment_branch_of_finetune_trains_and_records(fake_tabpfn, final_prior_launch, capsys):
    ft.main(final_prior_launch + ["--out", "run0"])
    # six updates on the launch's prior; the unfinished save had its own name and is gone
    assert sorted(os.listdir("run0")) == ["history.json", "provenance.json", "tabpfn_ft.pth"]
    assert [name for name, _ in fake_tabpfn["saved"]] == ["tabpfn_ft.partial.pth", "tabpfn_ft.pth"]
    assert len(_losses("run0")) == 6 and all(np.isfinite(_losses("run0")))
    prov = json.load(open("run0/provenance.json"))
    assert prov["finished"] is True and prov["steps_done"] == 6 and prov["torch_seed"] == 0
    assert prov["cfg"]["prior"]["family_weights"] == [0.0, 1.0] and prov["cfg"]["pfn"]["n_query"] == 64
    assert prov["finetune"]["seed"] == 0 and prov["finetune"]["n_query"] == 64       # the launch's, not this script's
    fields = fake_tabpfn["saved"][-1][1]["neuro_prior_finetune"]
    assert fields["steps"] == 6 and fields["prior"]["hyper_giles"]["p_te"] == [0.9, 1.0]
    # the starting weights: the package's file for v3.5, asked for by version, hashed, handed over by path
    start = fake_tabpfn["cache"] / "tabpfn-v3.5-classifier.ckpt"
    assert fake_tabpfn["resolved"] == [(None, "classifier", "v3.5")]
    assert _FakeClassifier.made[0].kw["model_path"] == str(start)
    base = prov["tabpfn"]
    assert base["md5"] == hashlib.md5(b"synthetic v3.5 weights").hexdigest() and base["model_version"] == "v3.5"
    assert base["model_version_expected"] == "v3.5" and base["model_path_arg"] is None
    assert "model version v3.5 (asked v3.5)" in capsys.readouterr().out

    # the same seed again: the same updates (the global torch generator is seeded; the stand-in model's dropout
    # and initial weights come from it). Another seed: other updates
    ft.main(final_prior_launch + ["--out", "run0_again"])
    assert _losses("run0_again") == _losses("run0")
    os.environ["SEED"] = "1"                                                 # restored by the fixture
    ft.main(final_prior_launch + ["--out", "run1"])
    assert _losses("run1") != _losses("run0")
    assert json.load(open("run1/provenance.json"))["torch_seed"] == 1
    with pytest.raises(SystemExit, match="differs from the launch's SEED"):
        ft.main(final_prior_launch + ["--out", "run1_other", "--seed", "0"])


def test_a_finished_model_is_left_alone_or_stops_the_run(fake_tabpfn, final_prior_launch, capsys):
    ft.main(final_prior_launch + ["--out", "run"])
    made, written = len(_FakeClassifier.made), os.path.getmtime("run/tabpfn_ft.pth")
    # asked again with the same configuration: nothing is trained, nothing is written
    ft.main(final_prior_launch + ["--out", "run"])
    assert len(_FakeClassifier.made) == made and os.path.getmtime("run/tabpfn_ft.pth") == written
    assert "already finished with this configuration" in capsys.readouterr().out
    ft.main(final_prior_launch + ["--out", "run", "--save-every", "2"])     # does not change the model
    assert len(_FakeClassifier.made) == made
    # asked with another configuration: the run stops and says what differs; the model stays
    for change, said in ((["--steps", "8"], "finetune.steps: 6 -> 8"), (["--lr", "3e-6"], "finetune.lr"),
                         (["--n-context", "128"], "finetune.n_context"), (["--n-estimators", "2"], "n_estimators"),
                         (["--processes-per-draw", "4"], "processes_per_draw")):
        with pytest.raises(SystemExit, match=said):
            ft.main(final_prior_launch + ["--out", "run"] + change)
    for edit, said in ((lambda p: p.update(git_commit="0" * 40), "git_commit"),
                       (lambda p: p["pool"][0].update(md5="0" * 32), "pool_md5"),
                       (lambda p: p.update(atlas_md5_reported_by_launch="000000000000"), "atlas_md5"),
                       (lambda p: p["tabpfn"].update(md5="0" * 32), "start_md5"),
                       (lambda p: p["cfg"]["prior"].update(family_weights=[0.5, 0.5]), "prior")):
        saved = json.load(open("run/provenance.json"))
        edit(saved)
        os.makedirs("other", exist_ok=True)
        shutil.copy("run/tabpfn_ft.pth", "other/tabpfn_ft.pth")
        json.dump(saved, open("other/provenance.json", "w"))
        with pytest.raises(SystemExit, match=said):
            ft.main(final_prior_launch + ["--out", "other"])
    assert len(_FakeClassifier.made) == made and os.path.getmtime("run/tabpfn_ft.pth") == written
    # another launch environment is another configuration too
    os.environ["GILES_HYPER"] = FINAL_PRIOR_ENV["GILES_HYPER"].replace("p_te=0.9:1", "p_te=1:1")
    with pytest.raises(SystemExit, match="another configuration"):
        ft.main(final_prior_launch + ["--out", "run"])
    os.environ["GILES_HYPER"] = FINAL_PRIOR_ENV["GILES_HYPER"]
    # an unfinished folder is trained again from the start
    saved = json.load(open("run/provenance.json")); saved["finished"] = False
    json.dump(saved, open("run/provenance.json", "w"))
    ft.main(final_prior_launch + ["--out", "run"])
    assert len(_FakeClassifier.made) == made + 1 and json.load(open("run/provenance.json"))["finished"] is True


def test_the_starting_weights_must_be_named(fake_tabpfn, final_prior_launch, tmp_path):
    opts = final_prior_launch
    # the package is set to another model version than the one asked for
    fake_tabpfn["settings"].tabpfn.model_version = "v2.5"
    with pytest.raises(SystemExit, match="set TABPFN_MODEL_VERSION=v3.5"):
        ft.main(opts + ["--out", "a"])
    (fake_tabpfn["cache"] / "tabpfn-v2.5-classifier.ckpt").write_bytes(b"synthetic v2.5 weights")
    ft.main(opts + ["--out", "a25", "--model-version", "v2.5"])              # asked for by name: allowed, recorded
    assert json.load(open("a25/provenance.json"))["tabpfn"]["model_version_expected"] == "v2.5"
    fake_tabpfn["settings"].tabpfn.model_version = "v3.5"
    # the package's file for the version is not on disk, or its resolver cannot be called
    os.remove(fake_tabpfn["cache"] / "tabpfn-v3.5-classifier.ckpt")
    with pytest.raises(SystemExit, match="pass --model-path"):
        ft.main(opts + ["--out", "b"])
    sys.modules["tabpfn.model_loading"].resolve_model_path = lambda *a: (_ for _ in ()).throw(TypeError("signature"))
    with pytest.raises(SystemExit, match="resolver failed"):
        ft.main(opts + ["--out", "b"])
    del fake_tabpfn["settings"].tabpfn.model_version                         # a release without that setting
    with pytest.raises(SystemExit):
        ft.main(opts + ["--out", "b"])
    assert not _FakeClassifier.made[1:] and not os.path.exists("b/provenance.json")   # nothing trained since a25
    # a file given by path: hashed, handed over, its version marked as not checked; a wrong digest stops
    weights = tmp_path / "given.ckpt"; weights.write_bytes(b"synthetic given weights")
    md5 = hashlib.md5(b"synthetic given weights").hexdigest()
    with pytest.raises(SystemExit, match="starting checkpoint md5"):
        ft.main(opts + ["--out", "c", "--model-path", str(weights), "--model-md5", "0" * 12])
    with pytest.raises(SystemExit, match="pass --model-path"):
        ft.main(opts + ["--out", "c", "--model-path", str(tmp_path / "missing.ckpt")])
    ft.main(opts + ["--out", "c", "--model-path", str(weights), "--model-md5", md5[:12]])
    base = json.load(open("c/provenance.json"))["tabpfn"]
    assert _FakeClassifier.made[-1].kw["model_path"] == str(weights) and base["md5"] == md5
    assert base["version_of_file"] == "not checked: a file given by path"


def test_without_the_new_option_finetune_writes_what_it_wrote_before(fake_tabpfn, final_prior_launch):
    """The other mode through the same stand-ins: the same files, the record
    in history.json with the keys of the options the script had at the base
    commit, no version pinned, one checkpoint name at every save."""
    ft.main(["--cache", "pool.npz", "--out", "old", "--family", "mixture", "--family-weights", "0", "1",
             "--n-query", "64", "--n-context", "96", "--giles-hyper", "random_partitions=1 threshold=0.03:0.08", "--device", "cpu", "--steps", "5",
             "--save-every", "2", "--processes-per-draw", "2"])
    assert sorted(os.listdir("old")) == ["history.json", "tabpfn_ft.pth"]
    assert [name for name, _ in fake_tabpfn["saved"]] == ["tabpfn_ft.pth"] * 3          # steps 2, 4 and the end
    assert "model_path" not in _FakeClassifier.made[0].kw and not fake_tabpfn["resolved"]
    config = json.load(open("old/history.json"))["config"]
    before = _from_base("scripts/finetune_tabpfn_prior.py").decode()
    options = {o.replace("-", "_") for o in re.findall(r'add_argument\("--([a-z-]+)"', before)}
    assert len(options) == 23 and "prior_from_env" not in options and set(config) == options
    assert config["seed"] == 0 and config["n_query"] == 64 and config["cache"] == "pool.npz"
    assert set(fake_tabpfn["saved"][-1][1]["neuro_prior_finetune"]) == {
        "steps", "learner", "n_context", "prior_hyper", "family", "family_weights", "giles_hyper", "cache"}


# ----------------------------------------------------------------- the launch
BASH = shutil.which("bash")
NEEDS = [shutil.which(x) for x in ("bash", "git", "md5sum", "find", "xargs", "awk", "diff", "readlink")]
SEEN = PRIOR_VARS + ("OUT", "JOBNAME", "ITERS", "BOTH_ARMS", "ARCH", "TABPFN_MODEL_VERSION")
RECORD = '''{ echo "ARGS $2"; for v in %s; do
  if [ -n "${!v+x}" ]; then echo "$v=${!v}"; else echo "$v=<unset>"; fi; done; } > "$OUT/call_$1.txt"
export -p > "$OUT/env_$1.txt"
''' % " ".join(SEEN)
INPUTS = ("FAMILY_W", "OBJECTIVE", "GENERIC_W", "RUN_TRAINER", "TABPFN_FT_ARGS", "SEEDS", "STEPS", "LR", "FT_BATCH",
          "N_CONTEXT_RANGE", "L2SP_ALPHA",
          "TRIALS_PER_DRAW", "N_CONTEXT", "N_EST", "MODEL_PATH", "MODEL_MD5", "TABPFN_VERSION", "PARALLEL",
          "POOL_MD5", "ATLAS_MD5", "FAIL_SEED", "RUN_ITERS", "ATLAS_SRC", "EXCLUDE_NETWORKS", "PRIOR_REPAIR",
          "ALLOC", "S_SOURCE", "STD", "DISTINCT", "HEAD_RANGE", "CATE_W", "PRIOR_CHECK_ONLY", *SEEN)
GIT = ["git", "-c", "user.name=t", "-c", "user.email=t@example.org", "-c", "core.autocrlf=false"]


@pytest.fixture
def launch_dir(tmp_path):
    """A folder without spaces in its path: the launch scripts split CACHE on
    spaces (several pools), so a pool under such a path cannot be passed."""
    if " " not in str(tmp_path):
        yield tmp_path
        return
    d = pathlib.Path(tempfile.mkdtemp(prefix="ncp_launch_"))
    if " " in str(d):
        pytest.skip("no temporary folder without spaces in its path")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _launch_repo(tmp_path):
    """A clean git checkout holding the three launch scripts and stand-ins for
    everything they start: a ``python`` that records its arguments and the
    launch variables, and a PFN training script that does the same. The
    stand-in for the fine-tuning script leaves a finished checkpoint, or fails
    for the seed named in FAIL_SEED."""
    repo = tmp_path / "repo"
    (repo / "scripts" / "audit_20261028").mkdir(parents=True)
    for name in ("g4c_launch.sh", "final1m_launch.sh", "tabpfn_final_launch.sh"):
        shutil.copy(os.path.join(ROOT, "scripts", "audit_20261028", name), repo / "scripts" / "audit_20261028" / name)
    (repo / "scripts" / "train_pfn_myriad.qsub.sh").write_text(
        '#!/bin/bash\nset -- pfn pfn\n' + RECORD + 'echo weights > "$OUT/pfn.pt"\n', newline="\n")
    venv = tmp_path / "venv" / "bin"; venv.mkdir(parents=True)
    (venv / "activate").write_text('export PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd):$PATH"\n',
                                   newline="\n")
    (venv / "python").write_text(
        '#!/bin/bash\nname=$(basename "$1" .py); args="$*"; set -- "$name" "$args"\n' + RECORD
        + 'if [ "$1" = finetune_tabpfn_prior ]; then\n'
          '  [ "${FAIL_SEED:-}" != "$SEED" ] || { echo "stand-in: seed $SEED fails"; exit 3; }\n'
          '  echo weights > "$OUT/tabpfn_ft.pth"; echo \'{"finished": true}\' > "$OUT/provenance.json"\nfi\n',
        newline="\n")
    os.chmod(venv / "python", 0o755)
    atlas = tmp_path / "atlas"; (atlas / "2mm_parcellations" / "genetics").mkdir(parents=True)
    (atlas / "functional_parcellation_2mm.nii.gz").write_bytes(b"synthetic atlas")
    (atlas / "2mm_parcellations" / "genetics" / "1_net.nii.gz").write_bytes(b"synthetic halves")
    pool = tmp_path / "pool.npz"; pool.write_bytes(b"synthetic pool")
    for cmd in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "launch scripts"]):
        subprocess.run(GIT + cmd, cwd=repo, check=True, capture_output=True)
    return repo, venv.parent, atlas, pool


def _digests(atlas):
    """The two digests as final1m_launch.sh computes them (md5sum's line
    format differs between platforms, so the atlas's is taken from the shell)."""
    atlas_md5 = subprocess.run(
        [BASH, "-c", "find functional_parcellation_2mm.nii.gz 2mm_parcellations -type f | LC_ALL=C sort "
                     "| xargs -d '\\n' md5sum | md5sum | cut -c1-12"],
        cwd=atlas, capture_output=True, text=True, check=True).stdout.strip()
    return dict(POOL_MD5=hashlib.md5(b"synthetic pool").hexdigest()[:12], ATLAS_MD5=atlas_md5)


def _run(repo, script, **env):
    full = {k: v for k, v in os.environ.items() if k not in INPUTS}
    full.update({k: str(v) for k, v in env.items()})
    return subprocess.run([BASH, f"scripts/audit_20261028/{script}"], cwd=repo, env=full, capture_output=True,
                          text=True)


def _call(path):
    lines = open(path).read().splitlines()
    return lines[0][5:], dict(line.split("=", 1) for line in lines[1:])


@pytest.mark.skipif(not all(NEEDS), reason="needs bash, git and the coreutils the launch scripts use")
def test_the_launch_trains_tabpfn_on_the_final_models_prior_one_model_per_seed(launch_dir):
    tmp_path = launch_dir
    repo, venv, atlas, pool = _launch_repo(tmp_path)
    common = dict(VENV=venv.as_posix(), POOL=pool.as_posix(), ATLAS=atlas.as_posix(), **_digests(atlas))

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
        # the digests come from the shared launch, which checked them; the version is given by name
        assert args == (f"scripts/finetune_tabpfn_prior.py --prior-from-env --out {d.as_posix()} "
                        f"--pool-md5 {common['POOL_MD5']} --atlas-md5 {common['ATLAS_MD5']} --steps 2000 "
                        f"--lr 1e-5 --processes-per-draw 4 --n-context 1500 --n-estimators 1 --model-version v3.5")
        assert env == gate_env and env["SEED"] == str(s) and env["CACHE"] == pool.as_posix()
        assert env["TABPFN_MODEL_VERSION"] == "v3.5"
    # the prior variables are the final model's, one by one (same seed: s1)
    _, env1 = _call(o / "s1" / "call_finetune_tabpfn_prior.txt")
    for k in PRIOR_VARS:
        assert env1[k] == pfn_env[k], k
    assert env1["N_QUERY"] == "<unset>" and env1["STANDARDIZE"] == "<unset>" and env1["VARIANT"] == "<unset>"

    # options from the environment. Every seed is handed to the script again (it is the script that leaves a
    # finished model alone or stops on another configuration); the log is appended to, not replaced
    r = _run(repo, "tabpfn_final_launch.sh", O=o.as_posix(), SEEDS="1 5", STEPS=300, LR="3e-6", TRIALS_PER_DRAW=2,
             N_CONTEXT=800, N_EST=2, TABPFN_VERSION="v2.5", MODEL_PATH=pool.as_posix(),
             MODEL_MD5=common["POOL_MD5"], **common)
    assert r.returncode == 0, r.stdout + r.stderr
    tail = (f"--steps 300 --lr 3e-6 --processes-per-draw 2 --n-context 800 --n-estimators 2 --model-version v2.5 "
            f"--model-path {pool.as_posix()} --model-md5 {common['POOL_MD5']}")
    for s in (1, 5):
        args, env = _call(o / f"s{s}" / "call_finetune_tabpfn_prior.txt")
        assert args.endswith(tail) and env["TABPFN_MODEL_VERSION"] == "v2.5"
    assert (o / "logs" / "train_s1.log").read_text().count(" launch, code ") == 2

    # refusals before anything starts: another pool, another atlas, bad options, a tree with local changes
    for bad in (dict(POOL_MD5="0" * 12), dict(ATLAS_MD5="0" * 12), dict(STEPS="2k"), dict(LR="fast"), dict(LR="0"),
                dict(LR="0e-5"), dict(FT_BATCH="4"), dict(TRIALS_PER_DRAW="0"), dict(TABPFN_VERSION="3.5"),
                dict(MODEL_MD5="abcdef012345"), dict(MODEL_PATH=(tmp_path / "missing.ckpt").as_posix())):
        o_bad = tmp_path / "bad"
        r = _run(repo, "tabpfn_final_launch.sh", O=o_bad.as_posix(), **{**common, **bad})
        assert r.returncode != 0 and not o_bad.exists(), bad
    assert "TRIALS_PER_DRAW" in _run(repo, "tabpfn_final_launch.sh", O=o_bad.as_posix(), FT_BATCH=4, **common).stderr
    with open(repo / "scripts" / "audit_20261028" / "g4c_launch.sh", "a") as f:
        f.write("# edited\n")
    r = _run(repo, "tabpfn_final_launch.sh", O=(tmp_path / "dirty").as_posix(), **common)
    assert r.returncode != 0 and "local changes" in r.stderr and not (tmp_path / "dirty").exists()


@pytest.mark.skipif(not all(NEEDS), reason="needs bash, git and the coreutils the launch scripts use")
def test_the_launch_fails_when_a_seed_fails(launch_dir):
    tmp_path = launch_dir
    repo, venv, atlas, pool = _launch_repo(tmp_path)
    common = dict(VENV=venv.as_posix(), POOL=pool.as_posix(), ATLAS=atlas.as_posix(), **_digests(atlas))
    # one after another: the launch stops at the seed that fails and exits with an error
    o = tmp_path / "seq"
    r = _run(repo, "tabpfn_final_launch.sh", O=o.as_posix(), FAIL_SEED=1, **common)
    assert r.returncode != 0, r.stdout + r.stderr
    assert (o / "s0" / "tabpfn_ft.pth").exists() and not (o / "s1" / "tabpfn_ft.pth").exists()
    assert not (o / "s2").exists()                                           # not started
    assert "s1: FAILED: stand-in: seed 1 fails" in r.stdout and "s2: no finished checkpoint" in r.stdout
    assert "s0: checkpoint " in r.stdout and "FAILED: not every seed has a finished model" in r.stdout
    # every seed failing is an error too, and so is one failing among seeds run at once
    r = _run(repo, "tabpfn_final_launch.sh", O=(tmp_path / "none").as_posix(), SEEDS="1", FAIL_SEED=1, **common)
    assert r.returncode != 0
    o = tmp_path / "par"
    r = _run(repo, "tabpfn_final_launch.sh", O=o.as_posix(), FAIL_SEED=1, PARALLEL=1, **common)
    assert r.returncode != 0, r.stdout + r.stderr
    assert (o / "s0" / "tabpfn_ft.pth").exists() and (o / "s2" / "tabpfn_ft.pth").exists()
    assert "s1: FAILED" in r.stdout and "s2: checkpoint " in r.stdout
    # a seed whose folder holds a finished model but whose run failed (the script refused it) is a failure
    r = _run(repo, "tabpfn_final_launch.sh", O=o.as_posix(), FAIL_SEED=0, **common)
    assert r.returncode != 0 and "s0: FAILED" in r.stdout
    r = _run(repo, "tabpfn_final_launch.sh", O=o.as_posix(), PARALLEL=1, **common)
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.skipif(not all(NEEDS), reason="needs bash, git and the coreutils the launch scripts use")
def test_the_shared_launch_gives_tabpfn_no_pool_without_its_digest(launch_dir):
    """g4c_launch.sh called directly with RUN_TRAINER=tabpfn: no default pool,
    one pool, both digests, checked before the gate."""
    tmp_path = launch_dir
    repo, venv, atlas, pool = _launch_repo(tmp_path)
    (repo / "data").mkdir(); shutil.copytree(atlas, repo / "data" / "atlases")
    out = tmp_path / "run"
    md5 = _digests(atlas)
    base = dict(VENV=venv.as_posix(), OUT=out.as_posix(), RUN_TRAINER="tabpfn")
    for env, said in ((dict(**md5), "needs CACHE"),                                     # the default pool is per fold
                      (dict(CACHE=pool.as_posix()), "needs POOL_MD5 and ATLAS_MD5"),
                      (dict(CACHE=pool.as_posix(), POOL_MD5=md5["POOL_MD5"]), "needs POOL_MD5 and ATLAS_MD5"),
                      (dict(CACHE=pool.as_posix(), POOL_MD5="0" * 12, ATLAS_MD5=md5["ATLAS_MD5"]), "pool md5"),
                      (dict(CACHE=pool.as_posix(), POOL_MD5=md5["POOL_MD5"], ATLAS_MD5="0" * 12), "atlas md5"),
                      (dict(CACHE=f"{pool.as_posix()} {pool.as_posix()}", **md5), "takes one pool")):
        r = _run(repo, "g4c_launch.sh", **base, **env)
        assert r.returncode != 0 and said in r.stderr, (env, r.stderr)
        assert not out.exists()                                              # stopped before the gate
    r = _run(repo, "g4c_launch.sh", CACHE=pool.as_posix(), TABPFN_FT_ARGS="--steps 5", **base, **md5)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (out / "call_prior_gate.txt").exists() and not (out / "call_pfn.txt").exists()
    assert _call(out / "call_finetune_tabpfn_prior.txt")[0] == (
        f"scripts/finetune_tabpfn_prior.py --prior-from-env --out {out.as_posix()} "
        f"--pool-md5 {md5['POOL_MD5']} --atlas-md5 {md5['ATLAS_MD5']} --steps 5")
    assert "launch tag=G4c_tabpfn " in r.stdout
    r = _run(repo, "g4c_launch.sh", VENV=venv.as_posix(), CACHE=pool.as_posix(), OUT=out.as_posix(),
             RUN_TRAINER="xgboost")
    assert r.returncode != 0 and "RUN_TRAINER" in r.stderr


def _exported(path):
    """The exported variables a stand-in saw, without the shell's own."""
    drop = ("declare -x _=", "declare -x SHLVL=", "declare -x OLDPWD", "declare -x PWD=")
    return [line for line in open(path).read().splitlines() if not line.startswith(drop)]


@pytest.mark.skipif(not all(NEEDS), reason="needs bash, git and the coreutils the launch scripts use")
@pytest.mark.parametrize("name,env", [
    ("default", dict()),                                                             # G4c
    ("final", dict(OBJECTIVE="new", FAMILY_W="0 1", SEED=1, RUN_ITERS=250000, ATLAS_SRC="data/atlases")),
    ("nogeneric", dict(GENERIC_W=0, OBJECTIVE="new", FAMILY_W="0.25 0.75")),
    # the variables of the TabPFN path, left in the shell: without RUN_TRAINER they change nothing
    ("leftovers", dict(OBJECTIVE="new", POOL_MD5="0" * 12, ATLAS_MD5="0" * 12, TABPFN_FT_ARGS="--steps 5",
                       TABPFN_MODEL_VERSION="v3.5")),
])
def test_without_the_new_variable_the_shared_launch_does_what_it_did_before(launch_dir, name, env):
    """g4c_launch.sh of the base commit and this tree's, run in the same
    checkout (one commit each) with the same stand-ins: the gate's command
    line, every exported variable at the gate and at the trainer, and what the
    launch prints must be equal."""
    tmp_path = launch_dir
    repo, venv, atlas, pool = _launch_repo(tmp_path)
    (repo / "data").mkdir(); shutil.copytree(atlas, repo / "data" / "atlases")
    out = tmp_path / "run"
    target = repo / "scripts" / "audit_20261028" / "g4c_launch.sh"
    new_script = target.read_bytes()
    seen = {}
    for which, script in (("old", _from_base("scripts/audit_20261028/g4c_launch.sh")), ("new", new_script)):
        target.write_bytes(script)
        subprocess.run(GIT + ["commit", "-q", "-a", "--allow-empty", "-m", which], cwd=repo, check=True,
                       capture_output=True)
        r = _run(repo, "g4c_launch.sh", VENV=venv.as_posix(), CACHE=pool.as_posix(), OUT=out.as_posix(), **env)
        assert r.returncode == 0, r.stdout + r.stderr
        assert (out / "pfn.pt").exists() and not (out / "call_finetune_tabpfn_prior.txt").exists()
        seen[which] = {"gate": _call(out / "call_prior_gate.txt"), "trainer": _call(out / "call_pfn.txt"),
                       "gate_env": _exported(out / "env_prior_gate.txt"), "trainer_env": _exported(out / "env_pfn.txt"),
                       "said": [line for line in r.stdout.splitlines() if not line.startswith("code_commit=")],
                       "stderr": r.stderr}
        shutil.rmtree(out)
    assert new_script != _from_base("scripts/audit_20261028/g4c_launch.sh")  # two versions were run
    for k in seen["old"]:
        assert seen["new"][k] == seen["old"][k], k
    assert any(line.startswith("declare -x GILES_HYPER=") for line in seen["new"]["trainer_env"])
    assert any(line.startswith("launch tag=G4c") for line in seen["new"]["said"])


# ----------------------------------- L2-SP and the drawn context length (25 Nov)
def _toy(seed=0):
    import torch

    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(5, 4), torch.nn.Tanh(), torch.nn.Linear(4, 2))
    model[2].bias.requires_grad = False                                      # a weight the task does not train
    x, y = torch.randn(30, 5), torch.randint(0, 2, (30,))
    return model, x, y


def test_the_l2sp_term_and_its_gradient_are_the_formula_on_a_toy_model():
    import torch

    model, x, y = _toy()
    alpha = 0.3
    guard = ft.L2SP(model.parameters(), alpha)
    start = [p.detach().clone() for p in model.parameters()]
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert len(guard.params) == 3 and all(a is b for a, b in zip(guard.params, trainable))    # the frozen bias is out
    w0_norm = np.sqrt(sum(float((p.detach().double() ** 2).sum()) for p in trainable))
    assert guard.w0_norm == pytest.approx(w0_norm, rel=1e-12)
    assert float(guard.term()) == 0.0                                        # at the starting weights
    with torch.no_grad():                                                    # move away from them
        for p in trainable:
            p.add_(0.1 * torch.randn_like(p))
    dist2 = sum(float(((p.detach() - w) ** 2).sum()) for p, w in zip(trainable, guard.w0))
    assert float(guard.term()) == pytest.approx(0.5 * alpha * dist2, rel=1e-6)
    # its gradient is alpha (w - w0), tensor by tensor
    grads = torch.autograd.grad(guard.term(), trainable)
    for g, p, w in zip(grads, trainable, guard.w0):
        torch.testing.assert_close(g, alpha * (p.detach() - w), rtol=1e-6, atol=1e-9)
    # the loss's own gradient, taken alone, and the logged quantities after the backward pass of loss + term
    loss = torch.nn.functional.cross_entropy(model(x), y)
    own = torch.autograd.grad(loss, trainable, retain_graph=True)
    own_norm = np.sqrt(sum(float((g.double() ** 2).sum()) for g in own))
    (loss + guard.term()).backward()
    s = guard.stats()
    assert set(s) == {"l2sp_term", "l2sp_grad_norm", "loss_grad_norm", "rel_dist"}
    assert s["l2sp_term"] == pytest.approx(0.5 * alpha * dist2, rel=1e-6)
    assert s["l2sp_grad_norm"] == pytest.approx(alpha * np.sqrt(dist2), rel=1e-6)
    assert s["loss_grad_norm"] == pytest.approx(own_norm, rel=1e-4)          # the whole gradient minus alpha (w - w0)
    assert s["rel_dist"] == pytest.approx(np.sqrt(dist2) / w0_norm, rel=1e-6)
    whole = np.sqrt(sum(float((p.grad.double() ** 2).sum()) for p in trainable))
    assert abs(whole - own_norm) > 1e-6                                      # so the two norms are different things
    # training moves the weights and leaves w0 where it was: a copy, without gradient, not the optimizer's
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    for _ in range(5):
        opt.zero_grad()
        (torch.nn.functional.cross_entropy(model(x), y) + guard.term()).backward()
        opt.step()
    for w, p, first in zip(guard.w0, trainable, [s_ for s_, q in zip(start, model.parameters()) if q.requires_grad]):
        assert torch.equal(w, first) and not torch.equal(p.detach(), first)
        assert w is not p and not w.requires_grad and w.grad is None
    frozen = torch.nn.Linear(2, 2)
    for p in frozen.parameters():
        p.requires_grad = False
    with pytest.raises(ValueError, match="trainable"):
        ft.L2SP(frozen.parameters(), 0.1)


def _script_before(tmp_path, monkeypatch):
    """scripts/finetune_tabpfn_prior.py of the commit before these two
    settings, as a module."""
    import importlib.util

    d = tmp_path / "before"; d.mkdir()
    (d / "ft_before.py").write_bytes(_from_base("scripts/finetune_tabpfn_prior.py", BEFORE_GUARD))
    monkeypatch.setattr(sys, "path", list(sys.path))                         # the script adds a folder to it
    spec = importlib.util.spec_from_file_location("ft_before", d / "ft_before.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert not hasattr(mod, "L2SP")
    return mod


def test_without_the_guard_and_the_range_the_updates_are_those_of_the_commit_before(
        fake_tabpfn, final_prior_launch, monkeypatch, tmp_path):
    """alpha unset or 0, no range: every update's loss and rate equal, bit
    for bit, those of the script at the commit before, on the same stand-ins
    and seed. With the guard the first update is the same (w = w0) and the
    later ones are not."""
    before = _script_before(tmp_path, monkeypatch)
    before.main(final_prior_launch + ["--out", "before_run"])
    ft.main(final_prior_launch + ["--out", "unset"])
    ft.main(final_prior_launch + ["--out", "zero", "--l2sp-alpha", "0"])
    old = json.load(open("before_run/history.json"))
    for run in ("unset", "zero"):
        new = json.load(open(f"{run}/history.json"))
        assert new["history"] == old["history"] and len(old["history"]) == 6     # steps, losses, rates: equal floats
        assert set(new) == set(old) == {"history", "config", "prior"} and new["prior"] == old["prior"]
        # the files are the old ones plus the two settings, recorded as off; nothing else differs
        assert set(new["config"]) - set(old["config"]) == {"l2sp_alpha", "n_context_range"}
        assert {k: v for k, v in new["config"].items() if k in old["config"] and k != "out"} == \
               {k: v for k, v in old["config"].items() if k != "out"}
        assert new["config"]["l2sp_alpha"] == 0.0 and new["config"]["n_context_range"] is None
        p_new, p_old = (json.load(open(f"{d}/provenance.json")) for d in (run, "before_run"))
        assert set(p_new) == set(p_old)
        assert set(p_new["finetune"]) - set(p_old["finetune"]) == {"l2sp_alpha", "n_context_range"}
        assert all(set(h) == {"step", "loss", "lr"} for h in new["history"])
        prov = json.load(open(f"{run}/provenance.json"))["finetune"]
        assert prov["l2sp_alpha"] == 0.0 and prov["n_context_range"] is None and prov["n_context"] == 96
    assert json.load(open("unset/provenance.json"))["cfg"] == json.load(open("before_run/provenance.json"))["cfg"]
    ft.main(final_prior_launch + ["--out", "guarded", "--l2sp-alpha", "5", "--lr", "0.05"])
    ft.main(final_prior_launch + ["--out", "free", "--lr", "0.05"])
    a, b = _losses("guarded"), _losses("free")
    assert a[0] == b[0] and a[1:] != b[1:]


def test_the_guard_in_finetune_records_its_quantities_and_holds_the_weights(
        fake_tabpfn, final_prior_launch, monkeypatch, capsys):
    import torch

    made = []

    class Recording(ft.L2SP):
        def __init__(self, parameters, alpha):
            super().__init__(parameters, alpha)
            self.first = [w.clone() for w in self.w0]
            made.append(self)

    monkeypatch.setattr(ft, "L2SP", Recording)
    opts = final_prior_launch + ["--lr", "0.05", "--steps", "12", "--save-every", "0"]
    ft.main(opts + ["--out", "tight", "--l2sp-alpha", "20"])
    out = capsys.readouterr().out
    guard, = made
    hist = json.load(open("tight/history.json"))["history"]
    keys = {"step", "loss", "lr", "l2sp_term", "l2sp_grad_norm", "loss_grad_norm", "rel_dist"}
    assert len(hist) == 12 and all(set(h) == keys and all(np.isfinite(list(h.values()))) for h in hist)
    # before the first update the weights are the starting ones: no distance, no term, and a gradient of the loss
    assert hist[0]["rel_dist"] == 0.0 and hist[0]["l2sp_term"] == 0.0 and hist[0]["l2sp_grad_norm"] == 0.0
    assert hist[0]["loss_grad_norm"] > 0 and all(h["rel_dist"] > 0 for h in hist[1:])
    for h in hist[1:]:                                                       # the three quantities are one distance
        assert h["l2sp_grad_norm"] == pytest.approx(20 * h["rel_dist"] * guard.w0_norm, rel=1e-9)
        assert h["l2sp_term"] == pytest.approx(h["l2sp_grad_norm"] ** 2 / (2 * 20), rel=1e-9)
    # the starting weights were copied once and not moved; the model's weights moved
    assert all(torch.equal(w, f) for w, f in zip(guard.w0, guard.first))
    assert any(not torch.equal(p.detach(), w) for p, w in zip(guard.params, guard.w0))
    live = [p for p in _FakeClassifier.made[-1].model_.parameters() if p.requires_grad]
    assert len(guard.params) == len(live) and all(a is b for a, b in zip(guard.params, live))
    # the log: the head line, and the four quantities at each log line (every 2 updates here)
    assert "L2-SP: alpha 20 towards the starting weights, 2 tensors" in out
    lines = [line for line in out.splitlines() if line.startswith("step ")]
    assert len(lines) == 6
    for line in lines:
        assert all(word in line for word in ("loss ", "l2sp ", "|grad l2sp| ", "|grad loss| ", "|w-w0|/|w0| "))
    assert f"|w-w0|/|w0| {hist[-1]['rel_dist']:.3e}" in lines[-1] and f"l2sp {hist[-1]['l2sp_term']:.3e}" in lines[-1]
    # recorded beside the checkpoint and inside it
    assert json.load(open("tight/provenance.json"))["finetune"]["l2sp_alpha"] == 20.0
    assert fake_tabpfn["saved"][-1][1]["neuro_prior_finetune"]["l2sp_alpha"] == 20.0
    # a weaker guard lets the weights go further in the same updates
    ft.main(opts + ["--out", "loose", "--l2sp-alpha", "1e-9"])
    loose = json.load(open("loose/history.json"))["history"]
    assert loose[-1]["rel_dist"] > hist[-1]["rel_dist"] and loose[0]["loss"] == hist[0]["loss"]


def _without(opts, name):
    i = opts.index(name)
    return opts[:i] + opts[i + 2:]


def test_context_lengths_are_drawn_in_the_range_and_repeat_with_the_seed(fake_tabpfn, final_prior_launch, capsys):
    opts = _without(final_prior_launch, "--n-context") + ["--steps", "14"]
    ft.main(opts + ["--out", "r0", "--n-context-range", "80:140"])
    out = capsys.readouterr().out
    h = json.load(open("r0/history.json"))
    drawn = [d["n_context"] for d in h["context_draws"]]
    assert len(drawn) >= 7 and all(80 <= n <= 140 for n in drawn)            # 14 updates, at most 2 per draw
    assert len(set(drawn)) > 3                                               # they differ between draws
    assert [d["after_step"] for d in h["context_draws"]] == sorted(d["after_step"] for d in h["context_draws"])
    # every draw says how many updates used its length; together they are the 14 updates, in order
    assert sum(d["updates"] for d in h["context_draws"]) == 14 and all(0 <= d["updates"] <= 2 for d in h["context_draws"])
    assert [u["n_context"] for u in h["history"]] == [d["n_context"] for d in h["context_draws"]
                                                      for _ in range(d["updates"])]
    done = 0
    for d in h["context_draws"]:
        assert d["after_step"] == done
        done += d["updates"]
    # each table of a draw has that many context rows plus the launch's 64 queries
    assert len(fake_tabpfn["tables"]) == len(drawn)
    for rows, n in zip(fake_tabpfn["tables"], drawn):
        assert rows and all(r == n + 64 for r in rows)
    # every length drawn is in the log, in order
    printed = [int(n) for line in out.splitlines() if "contexts drawn" in line
               for n in line.split("contexts drawn")[1].split("(")[0].split()]
    assert printed == drawn and "context 80:140 (drawn per draw of the prior)+64" in out
    # recorded: the range, and no fixed length
    prov = json.load(open("r0/provenance.json"))["finetune"]
    assert prov["n_context_range"] == [80, 140] and prov["n_context"] is None
    fields = fake_tabpfn["saved"][-1][1]["neuro_prior_finetune"]
    assert fields["n_context_range"] == [80, 140] and fields["n_context"] is None
    # the same seed draws the same lengths; another seed, others
    ft.main(opts + ["--out", "r0_again", "--n-context-range", "80:140"])
    again = json.load(open("r0_again/history.json"))
    assert again["context_draws"] == h["context_draws"] and again["history"] == h["history"]
    os.environ["SEED"] = "1"                                                 # restored by the fixture
    ft.main(opts + ["--out", "r1", "--n-context-range", "80:140"])
    assert [d["n_context"] for d in json.load(open("r1/history.json"))["context_draws"]][:7] != drawn[:7]
    os.environ["SEED"] = "0"
    # the lengths come from a generator of their own: a range of one length is the fixed length, update for update
    ft.main(opts + ["--out", "one", "--n-context-range", "96:96"])
    ft.main(opts + ["--out", "fixed", "--n-context", "96"])
    assert _losses("one") == _losses("fixed")
    assert ft.context_length_rng(3).integers(0, 10 ** 9) == ft.context_length_rng(3).integers(0, 10 ** 9)
    assert ft.context_length_rng(3).integers(0, 10 ** 9) != np.random.default_rng(3).integers(0, 10 ** 9)


def test_the_range_and_the_guard_are_refused_where_they_do_not_apply(capsys):
    ap = ft.build_parser()
    ok = ENV_OK + ["--n-context-range", "256:3000", "--l2sp-alpha", "0.003"]
    args = ap.parse_args(ok)
    ft.check_args(ap, args, ok)
    assert args.n_context_range == (256, 3000) and args.l2sp_alpha == 0.003 and args.n_context is None
    plain = ap.parse_args(ENV_OK)
    ft.check_args(ap, plain, ENV_OK)
    assert plain.n_context_range is None and plain.l2sp_alpha == 0.0 and plain.n_context == 2000
    # the range together with a fixed length: refused, also when the fixed length is the default value
    for n in ("1500", "2000"):
        argv = ok + ["--n-context", n]
        with pytest.raises(SystemExit) as e:
            ft.check_args(ap, ap.parse_args(argv), argv)
        assert e.value.code == 2 and "give one of the two" in capsys.readouterr().err
        with pytest.raises(SystemExit):
            ft.main(argv)
    # values that are not a range or not a weight
    for bad in (["--n-context-range", "3000:256"], ["--n-context-range", "0:10"], ["--n-context-range", "256"],
                ["--n-context-range", "256-3000"], ["--n-context-range", "2.5:10"], ["--l2sp-alpha", "-0.1"],
                ["--l2sp-alpha", "strong"], ["--l2sp-alpha", "inf"], ["--l2sp-alpha", "nan"],
                ["--l2sp-alpha", "1e999"]):
        with pytest.raises(SystemExit) as e:
            ap.parse_args(ENV_OK + bad)
        assert e.value.code == 2
    # the same weight written in other ways is the same weight; 0 in any form is 0, not -0
    assert ft.parse_alpha("3e-3") == ft.parse_alpha("0.003") == 0.003
    for zero in ("0", "0.000", "-0", "1e-999"):
        assert ft.parse_alpha(zero) == 0.0 and str(ft.parse_alpha(zero)) == "0.0"
    # they belong to the launch-environment mode
    for extra in (["--l2sp-alpha", "0.003"], ["--n-context-range", "256:3000"]):
        argv = ["--cache", "pool.npz", "--out", "o"] + extra
        with pytest.raises(SystemExit) as e:
            ft.check_args(ap, ap.parse_args(argv), argv)
        assert e.value.code == 2 and "need --prior-from-env" in capsys.readouterr().err


def test_a_finished_model_is_compared_on_the_guard_and_the_range(fake_tabpfn, final_prior_launch, capsys):
    ranged = _without(final_prior_launch, "--n-context")
    asked = ranged + ["--out", "run", "--l2sp-alpha", "0.5", "--n-context-range", "80:140"]
    ft.main(asked)
    made = len(_FakeClassifier.made)
    prov = json.load(open("run/provenance.json"))
    assert prov["finetune"]["l2sp_alpha"] == 0.5 and prov["finetune"]["n_context_range"] == [80, 140]
    ident = ft.run_identity(prov)
    assert ident["finetune.l2sp_alpha"] == 0.5 and ident["finetune.n_context_range"] == [80, 140]
    ft.main(asked)                                                           # the same settings: left as it is
    assert len(_FakeClassifier.made) == made and "already finished" in capsys.readouterr().out
    for change, said in (
            (ranged + ["--l2sp-alpha", "0.1", "--n-context-range", "80:140"], "finetune.l2sp_alpha: 0.5 -> 0.1"),
            (ranged + ["--n-context-range", "80:140"], "finetune.l2sp_alpha: 0.5 -> 0.0"),
            (ranged + ["--l2sp-alpha", "0.5", "--n-context-range", "80:141"], "finetune.n_context_range"),
            (final_prior_launch + ["--l2sp-alpha", "0.5"], "finetune.n_context_range")):
        with pytest.raises(SystemExit, match=said):
            ft.main(change + ["--out", "run"])
    # and the other way: a model finished without the settings is not the one asked for with them
    ft.main(final_prior_launch + ["--out", "plain"])
    for change in (final_prior_launch + ["--l2sp-alpha", "0.003"], ranged + ["--n-context-range", "96:96"]):
        with pytest.raises(SystemExit, match="another configuration"):
            ft.main(change + ["--out", "plain"])
    assert len(_FakeClassifier.made) == made + 1


@pytest.mark.skipif(not all(NEEDS), reason="needs bash, git and the coreutils the launch scripts use")
def test_the_launch_passes_the_guard_and_the_context_range(launch_dir):
    tmp_path = launch_dir
    repo, venv, atlas, pool = _launch_repo(tmp_path)
    common = dict(VENV=venv.as_posix(), POOL=pool.as_posix(), ATLAS=atlas.as_posix(), **_digests(atlas))
    o = tmp_path / "out"
    r = _run(repo, "tabpfn_final_launch.sh", O=o.as_posix(), SEEDS="0", STEPS=20000, LR="3e-7",
             N_CONTEXT_RANGE="256:3000", L2SP_ALPHA="0.003", **common)
    assert r.returncode == 0, r.stdout + r.stderr
    args, _ = _call(o / "s0" / "call_finetune_tabpfn_prior.txt")
    assert args.endswith("--steps 20000 --lr 3e-7 --processes-per-draw 4 --n-context-range 256:3000 "
                         "--n-estimators 1 --model-version v3.5 --l2sp-alpha 0.003")
    assert "--n-context " not in args                                        # no fixed length beside the range
    assert "context 256:3000, L2-SP alpha 0.003, estimators 1" in r.stdout
    # unset: neither option is passed (the exact command line is in the first launch test)
    o2 = tmp_path / "out_unset"
    r = _run(repo, "tabpfn_final_launch.sh", O=o2.as_posix(), SEEDS="0", **common)
    args, _ = _call(o2 / "s0" / "call_finetune_tabpfn_prior.txt")
    assert "l2sp" not in args and "range" not in args and "--n-context 1500 " in args
    assert "context 1500, L2-SP alpha none" in r.stdout
    # a fixed length with the guard, and a guard of 0 written out
    o3 = tmp_path / "out_fixed"
    r = _run(repo, "tabpfn_final_launch.sh", O=o3.as_posix(), SEEDS="0", N_CONTEXT=3000, L2SP_ALPHA="0", **common)
    args, _ = _call(o3 / "s0" / "call_finetune_tabpfn_prior.txt")
    assert "--n-context 3000 " in args and args.endswith("--l2sp-alpha 0")
    assert "L2-SP alpha 0 (zero: no term)" in r.stdout
    r = _run(repo, "tabpfn_final_launch.sh", O=(tmp_path / "out_tiny").as_posix(), SEEDS="0", L2SP_ALPHA="1e-999",
             **common)
    assert r.returncode == 0 and "L2-SP alpha 1e-999 (zero: no term)" in r.stdout    # said, not hidden
    # refused before anything starts: both lengths, a range that is not one, a weight that is not one
    for bad in (dict(N_CONTEXT_RANGE="256:3000", N_CONTEXT=1500), dict(N_CONTEXT_RANGE="3000:256"),
                dict(N_CONTEXT_RANGE="256-3000"), dict(N_CONTEXT_RANGE="0:10"), dict(N_CONTEXT_RANGE="256"),
                dict(L2SP_ALPHA="-0.003"), dict(L2SP_ALPHA="strong"), dict(L2SP_ALPHA="1e999"),
                dict(L2SP_ALPHA="-0")):
        o_bad = tmp_path / "bad"
        r = _run(repo, "tabpfn_final_launch.sh", O=o_bad.as_posix(), **{**common, **bad})
        assert r.returncode != 0 and not o_bad.exists(), bad
    r = _run(repo, "tabpfn_final_launch.sh", O=o_bad.as_posix(), N_CONTEXT_RANGE="256:3000", N_CONTEXT=1500, **common)
    assert "set one of the two" in r.stderr


def test_a_draw_that_gives_no_table_is_recorded_with_no_update(fake_tabpfn, final_prior_launch, monkeypatch, capsys):
    """The prior can return a draw from which no table can be made. Its
    length was drawn, so it is recorded and printed, with 0 updates."""
    real, calls = ft.datasets_from_batch, []

    def sometimes_empty(b, rng, learner="s"):
        calls.append(len(b["Xc"][0]))
        sets = real(b, rng, learner=learner)                                 # the outcome draws are made either way
        return [] if len(calls) in (2, 3) else sets

    monkeypatch.setattr(ft, "datasets_from_batch", sometimes_empty)
    ft.main(_without(final_prior_launch, "--n-context") + ["--out", "gaps", "--n-context-range", "80:140"])
    out = capsys.readouterr().out
    h = json.load(open("gaps/history.json"))
    draws = h["context_draws"]
    assert [d["n_context"] for d in draws] == calls                          # the length the prior was asked for
    assert [d["updates"] for d in draws] == [2, 0, 0, 2, 2]
    assert [d["after_step"] for d in draws] == [0, 2, 2, 2, 4]               # three draws at the same update count
    assert sum(d["updates"] for d in draws) == 6 == len(h["history"])
    assert [u["n_context"] for u in h["history"]] == [d["n_context"] for d in draws for _ in range(d["updates"])]
    printed = [int(n) for line in out.splitlines() if "contexts drawn" in line
               for n in line.split("contexts drawn")[1].split("(")[0].split()]
    assert printed == calls


def test_under_mixed_precision_the_guard_is_measured_on_unscaled_gradients_before_clipping(
        fake_tabpfn, final_prior_launch, monkeypatch):
    """On a card the loop scales the loss (GradScaler) and unscales the
    gradients before clipping. With a stand-in scaler that multiplies by
    1,024, the recorded quantities must be those of the plain path; and a
    clip far below the gradient's norm must not show in them."""
    import torch

    events = []

    class Scaler:
        factor = 1024.0

        def scale(self, x):
            return x * self.factor

        def unscale_(self, optimizer):
            events.append("unscale")
            for group in optimizer.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        p.grad.div_(self.factor)

        def step(self, optimizer):
            events.append("step")
            optimizer.step()

        def update(self):
            pass

    class Stays(torch.nn.Sequential):                                        # "cuda" without a card: stay where it is
        def to(self, *a, **kw):
            return self

    def init(self):
        self.model_ = Stays(torch.nn.Dropout(0.5), torch.nn.Linear(9, 2))

    real_stats, real_clip, real_loss = ft.L2SP.stats, torch.nn.utils.clip_grad_norm_, ft.ce_loss_from_raw_logits

    def stats(self):
        events.append("stats")
        return real_stats(self)

    def clip(*a, **kw):
        events.append("clip")
        return real_clip(*a, **kw)

    monkeypatch.setattr(_FakeClassifier, "_initialize_model_variables", init)
    monkeypatch.setattr(ft.L2SP, "stats", stats)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    opts = final_prior_launch + ["--lr", "0.05", "--l2sp-alpha", "2", "--save-every", "0"]
    ft.main(opts + ["--out", "plain"])                                       # the path without a scaler
    assert events[:3] == ["stats", "clip", "stats"] and "unscale" not in events
    del events[:]
    monkeypatch.setattr(sys.modules["tabpfn.finetuning._torch_compat"], "GradScaler", Scaler)
    monkeypatch.setattr(ft, "ce_loss_from_raw_logits", lambda logits, y, device: real_loss(logits, y, "cpu"))
    ft.main(opts + ["--out", "amp", "--device", "cuda"])
    assert events[:4] == ["unscale", "stats", "clip", "step"] and events.count("stats") == 6
    plain, amp = (json.load(open(f"{d}/history.json"))["history"] for d in ("plain", "amp"))
    for a, b in zip(plain, amp):
        for k in ("loss", "l2sp_term", "l2sp_grad_norm", "loss_grad_norm", "rel_dist"):
            assert b[k] == pytest.approx(a[k], rel=1e-4, abs=1e-12), k       # not 1,024 times larger
    # before clipping: a clip of 1e-6 changes the updates, not the first update's recorded gradient
    ft.main(opts + ["--out", "amp_clipped", "--device", "cuda", "--grad-clip", "1e-6"])
    clipped = json.load(open("amp_clipped/history.json"))["history"]
    assert amp[0]["loss_grad_norm"] > 1e-3                                   # a thousand times the clip
    assert clipped[0]["loss_grad_norm"] == pytest.approx(amp[0]["loss_grad_norm"], rel=1e-4)
    assert clipped[-1]["rel_dist"] != pytest.approx(amp[-1]["rel_dist"], rel=1e-3)
