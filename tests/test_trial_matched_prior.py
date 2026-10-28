"""Trial-matched prior, both-arm training targets and launch hygiene (28 Oct).

The trial kinds give effects of exactly -1, 0 and +1 with the requested null
share when TE = 1 and RE = 0; the earlier generic kinds draw the same
processes as before unless the no-clip option is set; the both-arm loss and
the widened head train; the launch cfg is the one the training script built
before; the checkpoint records its code; run_jobs.sh never reuses a
checkpoint of another job line; the fold builder counts groups that
straddle the split; the prior gate reads its metrics from a batch.
"""
import copy
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from neurocausalpfn.prior.generic_mix import (TRIAL_KINDS, GenericEffectMixture, generic_process,
                                              trial_process, wrap_generic)

torch = pytest.importorskip("torch")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.parametrize("kind", TRIAL_KINDS)
def test_trial_kinds_give_unit_effects_with_the_requested_null_share(kind):
    rng = np.random.default_rng(0)
    Xc, Xq = rng.normal(size=(1500, 10)), rng.normal(size=(500, 10))
    pr = trial_process(rng, Xc, Xq, kind, null_frac=0.5, n_active=4)
    p0 = np.concatenate([pr["p0c"], pr["p0q"]]); p1 = np.concatenate([pr["p1c"], pr["p1q"]])
    tau = p1 - p0
    assert set(np.unique(p0)) <= {0.0, 1.0} and set(np.unique(p1)) <= {0.0, 1.0}   # deterministic outcomes
    assert set(np.unique(tau)) == {-1.0, 0.0, 1.0}
    assert abs((tau == 0).mean() - 0.5) < 0.03
    assert np.all(p0[tau == 0] == 1.0)              # the null group responds to both arms, as in the trial
    noisy = trial_process(rng, Xc, Xq, kind, null_frac=0.5, n_active=4, te=0.8, re=0.1)
    t2 = noisy["p1q"] - noisy["p0q"]
    assert np.isclose(np.abs(t2[t2 != 0]).max(), 1 - (1 - 0.8) * 0.9 - 0.1)
    with pytest.raises(ValueError):
        trial_process(rng, Xc, Xq, "step", 0.5, 4)


def test_noclip_keeps_the_nominal_effect_and_the_default_is_unchanged():
    rng = np.random.default_rng(1)
    Xc, Xq = rng.normal(size=(800, 8)), rng.normal(size=(200, 8))
    kw = dict(scale=0.9, null_frac=0.4, n_active=3)
    old = generic_process(np.random.default_rng(2), Xc, Xq, "step", **kw)
    again = generic_process(np.random.default_rng(2), Xc, Xq, "step", **kw, noclip=False)
    np.testing.assert_array_equal(old["p1q"], again["p1q"])                     # same draws as before
    tau_old = np.concatenate([old["p1c"] - old["p0c"], old["p1q"] - old["p0q"]])
    assert np.abs(tau_old[tau_old != 0]).mean() < 0.85                          # clipping shrinks 0.9
    new = generic_process(np.random.default_rng(2), Xc, Xq, "step", **kw, noclip=True)
    tau = np.concatenate([new["p1c"] - new["p0c"], new["p1q"] - new["p0q"]])
    np.testing.assert_allclose(np.abs(tau[np.abs(tau) > 1e-9]), 0.9)
    p = np.concatenate([new["p0c"], new["p1c"]])
    assert p.min() >= 0.0 and p.max() <= 1.0


class _Cohort:
    d_x = 6

    def __init__(self):
        self.rng = np.random.default_rng(0)

    def sample_batch(self, batch_size, n_context=None):
        n, B = n_context or 64, batch_size
        Tq = self.rng.integers(0, 2, size=(B, 16)).astype(float)
        mu0 = np.full((B, 16), 0.5); mu1 = mu0 + 0.05
        return {"Xc": self.rng.normal(size=(B, n, 6)), "Tc": self.rng.integers(0, 2, size=(B, n)).astype(float),
                "Yc": np.full((B, n), 0.5), "Xq": self.rng.normal(size=(B, 16, 6)), "Tq": Tq,
                "mu_q": np.where(Tq == 1, mu1, mu0), "mu0": mu0, "mu1": mu1,
                "processes": [{"family": "theory", "n_slice": 300} for _ in range(B)]}


def test_mixture_draws_trial_kinds_with_binary_context_and_kind_weights():
    mix = wrap_generic(_Cohort(), {"generic_weight": 1.0, "generic_kinds": "trial+trial2+step",
                                   "generic_kind_weights": [1.0, 1.0, 0.0], "generic_null": [0.5, 0.5],
                                   "generic_clean": 1.0, "generic_te": [0.7, 0.9], "generic_re": [0.1, 0.2]},
                       seed=3)
    assert isinstance(mix, GenericEffectMixture) and "kind weights" in mix.describe()
    out = mix.sample_batch(20, n_context=200)
    fams = {p["family"] for p in out["processes"]}
    assert fams <= {"generic-trial", "generic-trial2"}                         # weight 0 never drawn
    tau = out["mu1"] - out["mu0"]
    assert set(np.unique(tau)) <= {-1.0, 0.0, 1.0}                             # clean = 1 overrides TE / RE
    assert set(np.unique(out["Yc"])) <= {0.0, 1.0}
    with pytest.raises(ValueError):
        GenericEffectMixture(_Cohort(), 1.0, kinds=("trial", "spiral"))
    with pytest.raises(ValueError):
        GenericEffectMixture(_Cohort(), 1.0, kinds=("trial",), kind_weights=[1.0, 2.0])


def test_uniform_kind_draws_are_unchanged_by_the_new_options():
    a = GenericEffectMixture(_Cohort(), 1.0, kinds=("linear", "step"), seed=5).sample_batch(6, n_context=50)
    b = GenericEffectMixture(_Cohort(), 1.0, kinds=("linear", "step"), seed=5, kind_weights=None,
                             noclip=False, te=None, re=None).sample_batch(6, n_context=50)
    np.testing.assert_array_equal(a["mu1"], b["mu1"])


def test_both_arm_loss_and_wide_head_train_and_reach_unit_outcomes():
    from neurocausalpfn.pfn.model import NeuroCausalPFN
    from neurocausalpfn.train.train_pfn import pfn_loss

    torch.manual_seed(0)
    B, n, nq, d = 2, 20, 5, 4
    batch = {"Xc": torch.randn(B, n, d), "Tc": torch.randint(0, 2, (B, n)).float(), "Yc": torch.rand(B, n),
             "Xq": torch.randn(B, nq, d), "Tq": torch.randint(0, 2, (B, nq)).float(),
             "mu0": torch.zeros(B, nq), "mu1": torch.ones(B, nq)}
    batch["mu_q"] = torch.where(batch["Tq"] == 1, batch["mu1"], batch["mu0"])
    model = NeuroCausalPFN(d_x=d, d_model=16, n_layers=1, n_heads=2, n_bins=64, lo=-0.05, hi=1.05)
    base = pfn_loss(model, batch, {})
    both = pfn_loss(model, batch, {"both_arms": True})
    with_cate = pfn_loss(model, batch, {"both_arms": True, "cate_weight": 1.0})
    assert torch.isfinite(base) and torch.isfinite(both) and with_cate > both
    with_cate.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
    # the target histogram of an outcome of exactly 1 has its mean at 1 with the wide range, not with [0, 1]
    wide, narrow = model.head, NeuroCausalPFN(d_x=d, d_model=16, n_layers=1, n_heads=2, n_bins=64).head
    for head, ok in ((wide, True), (narrow, False)):
        m = float((head.target_hist(torch.tensor([1.0])) * head.centers).sum())
        assert (abs(m - 1.0) < 0.003) == ok


G5_ENV = {"FAMILY_WEIGHTS": "0.5 0.5", "SEED": "0", "ARCH": "linear", "FAMILY": "mixture",
          "GILES_HYPER": "random_partitions=1 threshold=0.03:0.08 p_te=0.9:1 p_re=0:0.05 label_noise=0:0 "
                         "bias=0:0 bias_weights=1:0:0",
          "PRIOR_HYPER": "two_arms=1 effect_scale=0.2:1 d_source=network s_source=genetics_1 "
                         "s_b_source=genetics_2 effect_headroom=1 generic_weight=0.5 binary_outcome=1 "
                         "generic_binary=1 gamma=0:0",
          "LR_SCHEDULE": "constant", "CONTEXT_POOL": "susceptible", "CONTEXT_MAX": "3000",
          "CACHE": "pool/fold0.npz", "OUT": "out/G5/fold0", "ITERS": "20000"}


def _old_qsub_cfg(env):
    """The cfg the training script built before 28 Oct (its inline code, verbatim in logic)."""
    from neurocausalpfn.train.train_pfn import e12_config, parse_hyper_opts, reduced_config, with_family

    g = lambda k, d="": env.get(k) or d
    fam = g("FAMILY") or "theory"
    w = [float(x) for x in g("FAMILY_WEIGHTS").split()] or None
    cp = g("CONTEXT_POOL", "susceptible")
    cfg = e12_config(g("VARIANT"), fam, w, cp) if g("VARIANT") else with_family(reduced_config(), fam, w, cp)
    cfg["out_dir"] = g("OUT", "outputs/pfn_reduced"); cfg["seed"] = int(g("SEED", "0"))
    cfg["pfn"]["iters"] = int(g("ITERS", "20000")); cfg["pfn"]["arch"] = g("ARCH", "tabicl")
    cfg["pfn"]["lr_schedule"] = g("LR_SCHEDULE", "constant")
    if g("CONTEXT_MAX"): cfg["pfn"]["context_max"] = int(g("CONTEXT_MAX"))
    caches = g("CACHE").split()
    cfg["prior"]["cache"] = caches[0] if len(caches) == 1 else caches
    cfg["prior"]["atlas_dir"] = "data/atlases"
    cfg["prior"]["hyper"] = {**cfg["prior"].get("hyper", {}), **parse_hyper_opts(g("PRIOR_HYPER"))}
    if g("GILES_HYPER"):
        cfg["prior"]["hyper_giles"] = {**cfg["prior"].get("hyper_giles", {}), **parse_hyper_opts(g("GILES_HYPER"))}
    return cfg


def test_launch_cfg_reproduces_the_g5_cfg_and_reads_the_new_switches():
    from neurocausalpfn.train.launch_cfg import cfg_from_env

    assert cfg_from_env(G5_ENV) == _old_qsub_cfg(G5_ENV)
    env = dict(G5_ENV, BOTH_ARMS="1", CATE_WEIGHT="0.5", HEAD_LO="-0.05", HEAD_HI="1.05", N_QUERY="256")
    p = cfg_from_env(env)["pfn"]
    assert p["both_arms"] is True and p["cate_weight"] == 0.5 and (p["head_lo"], p["head_hi"]) == (-0.05, 1.05)
    assert p["n_query"] == 256


def test_training_records_provenance_and_the_estimator_reads_it(tmp_path):
    from test_prior_families import _cache, _tiny_cfg

    from neurocausalpfn.pfn.estimator import PFNEstimator
    from neurocausalpfn.train.train_pfn import _build_prior, run_pfn

    cfg = _tiny_cfg(_cache(tmp_path), tmp_path / "run", "mixture")
    cfg["prior"]["hyper"] = {"generic_weight": 0.8, "generic_kinds": "trial+trial2_mlp+step",
                             "generic_null": [0.3, 0.7], "generic_clean": 0.7, "generic_te": [0.75, 1.0],
                             "generic_re": [0.0, 0.25], "generic_noclip": 1.0}
    cfg["pfn"].update({"both_arms": True, "cate_weight": 0.5, "head_lo": -0.05, "head_hi": 1.05})
    prior, _, _ = _build_prior(cfg)
    assert "trial TE" in prior.describe() and "no clipping" in prior.describe()
    model, history = run_pfn(cfg)
    assert all(math.isfinite(h["loss"]) for h in history)
    ck = torch.load(os.path.join(cfg["out_dir"], "pfn.pt"), map_location="cpu", weights_only=False)
    assert set(ck["provenance"]) >= {"git_commit", "git_dirty", "launch_env", "torch"}
    # inside a git checkout the commit is recorded and dirtiness is a boolean, never None
    if ck["provenance"]["git_commit"] is not None:
        assert isinstance(ck["provenance"]["git_dirty"], bool)
    assert os.path.isfile(os.path.join(cfg["out_dir"], "provenance.json"))
    est = PFNEstimator(os.path.join(cfg["out_dir"], "pfn.pt"))
    assert est.provenance == ck["provenance"]
    assert float(est.model.head.edges[0]) == pytest.approx(-0.05)


def test_prior_gate_metrics_and_checks():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import prior_gate as pg

    batch = {"mu0": np.array([[0, 1, 1, 0.5]], float), "mu1": np.array([[1, 1, 0, 0.52]], float),
             "Yc": np.array([[0, 1, 1]], float), "Tc": np.array([[1, 0, 1]], float),
             "processes": [{"family": "generic-trial", "n_slice": 300}]}
    (row,) = pg.process_stats(batch)
    assert row["share_zero"] == 0.25 and row["share_unit"] == 0.5 and row["share_small"] == 0.25
    assert row["pos_share"] == pytest.approx(2 / 3) and row["ctx_binary"] == 1.0
    s = pg.summarise([row])
    assert s["all"]["processes"] == 1 and s["generic-trial"]["median_slice"] == 300
    errs = pg.check(s, pg.DEFAULT_REQUIRE)
    assert any(e.startswith("share_zero") for e in errs) and any(e.startswith("share_small") for e in errs)
    cfg = {"prior": {"hyper": {"generic_weight": 0.8, "generic_null": [0.3, 0.7]}, "family": "mixture"}}
    assert pg.check_expect(cfg, ["prior.hyper.generic_weight=0.8", "prior.hyper.generic_null=0.3:0.7",
                                 "prior.family=mixture"]) == []
    assert len(pg.check_expect(cfg, ["prior.hyper.generic_weight=0.5", "prior.hyper.missing=1"])) == 2


def test_straddling_groups_counts_repeats_that_train_while_their_group_is_tested():
    from neurocausalpfn.prior.giles_replica import group_folds, straddling_groups

    files = [f"img{i}.nii.gz" for i in range(12)]
    groups = {f: (f"g{i}", 0) for i, f in enumerate(files[:10])}
    groups[files[10]] = ("g0", 1); groups[files[11]] = ("g1", 1)     # repeats of two tested groups
    giles = group_folds(files, groups, n_folds=5, mode="giles")
    strict = group_folds(files, groups, n_folds=5, mode="strict")
    assert straddling_groups(files, groups, giles) == 2
    assert straddling_groups(files, groups, strict) == 0


@pytest.mark.skipif(sys.platform.startswith("win"), reason="needs a POSIX shell with readlink -f")
def test_run_jobs_never_reuses_a_checkpoint_of_another_line(tmp_path):
    out = tmp_path / "job"
    jobs = tmp_path / "jobs.txt"
    venv = tmp_path / "venv"; (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").write_text("#!/bin/sh\n"); (venv / "bin" / "python").chmod(0o755)
    line = f"OUT={out} bash -c 'mkdir -p {out} && touch {out}/pfn.pt && echo x > {out}/pfn.pt'"
    jobs.write_text(line + "\n")
    run = lambda: subprocess.run(["bash", os.path.join(ROOT, "scripts", "run_jobs.sh"), str(jobs)],
                                 capture_output=True, text=True, env={**os.environ, "VENV": str(venv),
                                                                      "WORKDIR": str(tmp_path)})
    first = run()
    assert first.returncode == 0 and (out / ".jobline").read_text() == line
    assert "skip, done already" in run().stdout
    jobs.write_text(line.replace("echo x", "echo y") + "\n")
    third = run()
    assert third.returncode != 0 and "different job line" in third.stdout
