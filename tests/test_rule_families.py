"""Job G (16 Nov): the volume and centroid rule families of the prior
generator, mixture weights over all families, and the gate's checks of them.
Synthetic arrays only.

The first tests are the hard requirement: with the new families at weight 0
the prior draws exactly what the code before the change drew (commit 8fa4b74),
compared array by array for fixed seeds."""
import io
import os
import shutil
import subprocess
import sys
import tarfile

import numpy as np
import pytest

from neurocausalpfn.prior.atlas import FunctionalAtlas
from neurocausalpfn.prior.cohort import FAMILIES, MIXTURE_FAMILIES, NeuroPriorCohort
from neurocausalpfn.prior.feature_rules import (RULE_FAMILIES, FeatureRuleDGP, average_ranks, make_rule_dataset,
                                                principal_coordinate, rule_effect)
from neurocausalpfn.prior.neuro_prior import HyperPrior

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
BASE_COMMIT = "8fa4b74ba1f0d17d7d5fa0d980ce969be6d8ac34"      # scale-budget-ckpt, the code before this change
FIRST_ROUND = "78713055fe78174e072758ac619f7899cb125cec"      # this branch before the options of 17 Nov
KEYS = ("Xc", "Tc", "Yc", "Xq", "Tq", "mu_q", "mu0", "mu1")
# every option of the volume and centroid families away from its default, in two combinations
ALL_RULE_OPTIONS = ("rule_graded=0.9 rule_enrol_by=network rule_axis=pool rule_axis_sign=random",
                    "rule_graded=0.1 rule_enrol=0.3:0.9 rule_enrol_by=trial rule_axis_sign=random")

# the prior variables scripts/audit_20261028/g4c_launch.sh exports under
# final1m_launch.sh's call (OBJECTIVE=new FAMILY_W="0 1")
FINAL_PRIOR_ENV = {
    "FAMILY": "mixture", "FAMILY_WEIGHTS": "0 1", "CONTEXT_POOL": "susceptible", "CONTEXT_MAX": "3000",
    "GILES_HYPER": "random_partitions=1 threshold=0.03:0.08 p_te=0.9:1 p_re=0:0.05 label_noise=0:0",
    "PRIOR_HYPER": "two_arms=1 effect_scale=0.2:1 d_source=network s_source=genetics_1 s_b_source=genetics_2 "
                   "effect_headroom=1 generic_weight=0.5 binary_outcome=1 generic_binary=1",
}

# draws batches of the prior of the launch environment and saves them; run
# once on the code of BASE_COMMIT and once on this checkout
DRAW = '''
import sys
import numpy as np
from neurocausalpfn.train.launch_cfg import cfg_from_env
from neurocausalpfn.train.train_pfn import _build_prior
prior, _, _ = _build_prior(cfg_from_env())
out, fams = {}, []
for j, n_ctx in enumerate((64, 200, 96, 512, 128)):
    b = prior.sample_batch(4, n_context=n_ctx)
    for k in ("Xc", "Tc", "Yc", "Xq", "Tq", "mu_q", "mu0", "mu1"):
        out[f"{k}_{j}"] = np.asarray(b[k])
    fams += [p["family"] for p in b["processes"]]
np.savez(sys.argv[1], families=np.array(fams), **out)
'''


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


def _src_at(commit, d):
    """src/ of an earlier commit, taken from git history. Without git, or in
    a clone that does not hold the commit, the comparison cannot be made: that
    is a failure, not a skip, so "all tests pass" always includes it."""
    if shutil.which("git") is None:
        pytest.fail("git is not available: the comparison with the code before the change cannot run")
    r = subprocess.run(["git", "archive", "--format=tar", commit, "src/neurocausalpfn"], cwd=ROOT,
                       capture_output=True)
    if r.returncode != 0:
        pytest.fail(f"commit {commit[:7]} is not in this clone's history (fetch it): the comparison with the code "
                    "before the change cannot run")
    with tarfile.open(fileobj=io.BytesIO(r.stdout)) as tar:
        try:
            tar.extractall(d, filter="data")
        except TypeError:                                    # a Python without the filter argument
            tar.extractall(d)
    return str(d / "src")


@pytest.fixture(scope="module")
def base_src(tmp_path_factory):
    """src/ of the code before the change."""
    src = _src_at(BASE_COMMIT, tmp_path_factory.mktemp("base_code"))
    assert not os.path.exists(os.path.join(src, "neurocausalpfn", "prior", "feature_rules.py"))
    return src


def test_a_clone_without_the_base_commit_fails_the_comparison(tmp_path):
    with pytest.raises(pytest.fail.Exception, match="not in this clone's history"):
        _src_at("0" * 40, tmp_path)


def _draw(src, workdir, name, **env):
    """Batches drawn by the code under ``src`` for a launch environment."""
    script = workdir / "draw.py"
    script.write_text(DRAW)
    out = workdir / f"{name}.npz"
    full = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "GILES_HYPER", "PRIOR_HYPER")}
    full.update(PYTHONPATH=src, **env)
    r = subprocess.run([sys.executable, str(script), str(out)], cwd=workdir, env=full, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    with np.load(out) as z:
        return {k: z[k] for k in z.files}


def _same(a, b):
    assert sorted(a) == sorted(b)
    for k in a:
        assert a[k].shape == b[k].shape and np.array_equal(a[k], b[k]), k


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_final_prior_is_unchanged_draw_for_draw(base_src, tmp_path, seed):
    """The final model's prior (family weights 0 : 1, generic share 0.5): the
    code before the change and this code give the same batches, with the
    weights as two numbers (the final launch) and as four ending in zeros."""
    _synthetic_cache(tmp_path / "pool.npz")
    env = dict(FINAL_PRIOR_ENV, CACHE="pool.npz", SEED=str(seed))
    before = _draw(base_src, tmp_path, "before", **env)
    after = _draw(os.path.join(ROOT, "src"), tmp_path, "after", **env)
    four = _draw(os.path.join(ROOT, "src"), tmp_path, "four", **dict(env, FAMILY_WEIGHTS="0 1 0 0"))
    _same(before, after)
    _same(before, four)
    # the options of the new families, all away from their defaults, with those families at weight 0:
    # nothing is drawn for them, so the batches are still the ones of the code before the change
    for i, opts in enumerate(ALL_RULE_OPTIONS):
        set_ = _draw(os.path.join(ROOT, "src"), tmp_path, f"opts{i}",
                     **dict(env, FAMILY_WEIGHTS="0 1 0 0", GILES_HYPER=env["GILES_HYPER"] + " " + opts))
        _same(before, set_)
    fams = set(before["families"].tolist())
    assert "giles" in fams and any(f.startswith("generic-") for f in fams)   # both components were drawn
    assert not fams & {"theory", "volume", "centroid"}
    assert before["Xc_3"].shape == (4, 512, 8) and before["mu0_0"].shape == (4, 64)


def test_the_earlier_priors_are_unchanged_too(base_src, tmp_path):
    """Family weights 0.5 : 0.5 (the 500,000-trial models) with and without
    the generic share, and the single families."""
    _synthetic_cache(tmp_path / "pool.npz")
    src = os.path.join(ROOT, "src")
    half = dict(FINAL_PRIOR_ENV, CACHE="pool.npz", SEED="0", FAMILY_WEIGHTS="0.5 0.5")
    before = _draw(base_src, tmp_path, "before", **half)
    _same(before, _draw(src, tmp_path, "after", **half))
    _same(before, _draw(src, tmp_path, "four", **dict(half, FAMILY_WEIGHTS="0.5 0.5 0 0")))
    assert {"theory", "giles"} <= set(before["families"].tolist())
    plain = dict(half, PRIOR_HYPER=half["PRIOR_HYPER"].replace("generic_weight=0.5", "generic_weight=0"))
    _same(_draw(base_src, tmp_path, "before0", **plain), _draw(src, tmp_path, "after0", **plain))
    giles = dict(CACHE="pool.npz", SEED="3", FAMILY="giles", CONTEXT_MAX="512",
                 PRIOR_HYPER="random_partitions=1 threshold=0.03:0.08 p_te=0.9:1")
    _same(_draw(base_src, tmp_path, "before_g", **giles), _draw(src, tmp_path, "after_g", **giles))


# ------------------------------------------------------------ the two rules
def test_average_ranks_share_ties():
    np.testing.assert_array_equal(average_ranks([10.0, 30.0, 20.0]), [1, 3, 2])
    np.testing.assert_array_equal(average_ranks([5, 1, 5, 5, 2]), [4, 1, 4, 4, 2])
    np.testing.assert_array_equal(average_ranks([7, 7]), [1.5, 1.5])
    assert average_ranks([]).shape == (0,)


def test_threshold_rule_gives_thirds():
    v = np.arange(9.0)
    np.testing.assert_array_equal(rule_effect(v, "threshold"), [-1, -1, -1, 0, 0, 0, 1, 1, 1])
    rng = np.random.default_rng(0)
    for n in (3, 30, 300):                                   # a multiple of 3, no ties: n / 3 each, by value
        v = rng.normal(size=n)
        e = rule_effect(v, "threshold")
        assert (e == 1).sum() == (e == -1).sum() == (e == 0).sum() == n // 3
        assert v[e == 1].min() > v[e == 0].max() and v[e == 0].min() > v[e == -1].max()
    for n in (10, 11, 100):                                  # otherwise the two ends have the same size
        e = rule_effect(rng.normal(size=n), "threshold")
        assert (e == 1).sum() == (e == -1).sum() and abs((e == 1).sum() - n / 3) < 1
    assert rule_effect([4.0], "threshold").tolist() == [0.0]
    with pytest.raises(ValueError):
        rule_effect(v, "cubic")


def test_threshold_rule_with_ties_is_a_function_of_the_value():
    # six lesions, the cuts after the 2nd and the 4th
    np.testing.assert_array_equal(rule_effect([1, 2, 3, 4, 5, 6], "threshold"), [-1, -1, 0, 0, 1, 1])
    # a tie inside a third stays there
    np.testing.assert_array_equal(rule_effect([1, 1, 3, 4, 6, 6], "threshold"), [-1, -1, 0, 0, 1, 1])
    # a tie across a cut goes where its mean rank falls: (3 + 4 + 5) / 3 = 4, the middle
    np.testing.assert_array_equal(rule_effect([1, 2, 5, 5, 5, 6], "threshold"), [-1, -1, 0, 0, 0, 1])
    # ... (4 + 5 + 6) / 3 = 5, the top
    np.testing.assert_array_equal(rule_effect([1, 2, 3, 6, 6, 6], "threshold"), [-1, -1, 0, 1, 1, 1])
    # a pair exactly on a cut (ranks 2 and 3, mean 2.5) is not in the bottom third
    np.testing.assert_array_equal(rule_effect([1, 2, 2, 4, 5, 6], "threshold"), [-1, 0, 0, 0, 1, 1])
    # all equal: no effect for anyone, either shape
    assert not rule_effect(np.full(7, 2.0), "threshold").any() and not rule_effect(np.full(7, 2.0), "graded").any()
    rng = np.random.default_rng(1)
    v = rng.integers(0, 6, size=200).astype(float)            # many ties
    for shape in ("threshold", "graded"):
        e = rule_effect(v, shape)
        for x in np.unique(v):
            assert len(np.unique(e[v == x])) == 1
        assert (np.diff(e[np.argsort(v, kind="stable")]) >= 0).all()      # never decreasing in the value
        np.testing.assert_array_equal(rule_effect(-v, shape), -e)         # antisymmetric


def test_graded_rule_is_linear_in_the_rank():
    np.testing.assert_allclose(rule_effect([3.0, 1.0, 2.0, 5.0, 4.0], "graded"), [0, -1, -0.5, 1, 0.5])
    np.testing.assert_allclose(rule_effect([1, 2, 2, 3], "graded"), [-1, 0, 0, 1])
    rng = np.random.default_rng(2)
    v = rng.lognormal(size=101)
    e = rule_effect(v, "graded")
    assert e.min() == -1 and e.max() == 1 and abs(e.mean()) < 1e-12
    np.testing.assert_allclose(np.sort(e), np.linspace(-1, 1, 101))
    np.testing.assert_array_equal(np.argsort(e), np.argsort(v))
    assert rule_effect([4.0], "graded").tolist() == [0.0]


def test_principal_coordinate_is_the_first_axis_of_the_centroids():
    rng = np.random.default_rng(3)
    t = rng.normal(size=500)
    direction = np.array([2.0, -1.0, 0.5]) / np.linalg.norm([2.0, -1.0, 0.5])
    c = 10.0 + t[:, None] * 5.0 * direction + 0.1 * rng.normal(size=(500, 3))
    coord = principal_coordinate(c)
    assert abs(np.corrcoef(coord, t)[0, 1]) > 0.999 and abs(coord.mean()) < 1e-9
    assert np.corrcoef(coord, t)[0, 1] > 0                    # the axis's largest component is made positive
    np.testing.assert_allclose(principal_coordinate(c[::-1]), coord[::-1])       # no dependence on the order
    np.testing.assert_allclose(principal_coordinate(c + 100.0), coord, atol=1e-8)
    assert principal_coordinate(c[:1]).tolist() == [0.0]


def _process(feature, shape, sign=1, m=120, seed=0, **kw):
    rng = np.random.default_rng(seed)
    volumes, centroids = rng.lognormal(size=m), rng.normal(size=(m, 3)) * [8.0, 2.0, 1.0]
    dgp = FeatureRuleDGP(feature, rng, HyperPrior(), shape=shape, sign=sign, **kw)
    return dgp.fit(volumes, centroids, np.arange(m)), volumes, centroids, rng


@pytest.mark.parametrize("feature", RULE_FAMILIES)
def test_expected_outcomes_are_the_definition_at_full_efficacy(feature):
    """Efficacy 1, recovery 0: 0.5 + effect / 2 under arm 1 and 0.5 - effect / 2
    under arm 0, with the effect given by the family's own scalar."""
    for shape in ("threshold", "graded"):
        dgp, volumes, centroids, rng = _process(feature, shape, p_te=1.0, p_re=0.0, bias=0.0, label_noise=0.0)
        scalar = volumes if feature == "volume" else principal_coordinate(centroids)
        e = rule_effect(scalar, shape)
        np.testing.assert_array_equal(dgp.effect, e)
        np.testing.assert_allclose(dgp.mu(e, 1), 0.5 + e / 2)
        np.testing.assert_allclose(dgp.mu(e, 0), 0.5 - e / 2)
        if shape == "threshold":
            top, bottom = np.argsort(scalar)[-40:], np.argsort(scalar)[:40]
            assert (dgp.mu(e, 1)[top] == 1).all() and (dgp.mu(e, 0)[top] == 0).all()
            assert (dgp.mu(e, 1)[bottom] == 0).all() and (dgp.mu(e, 0)[bottom] == 1).all()
            middle = np.argsort(scalar)[40:80]
            assert (dgp.mu(e, 1)[middle] == 0.5).all() and (dgp.mu(e, 0)[middle] == 0.5).all()
        # the other family's scalar does not enter
        other = FeatureRuleDGP(feature, rng, shape=shape, sign=1, p_te=1.0, p_re=0.0)
        if feature == "volume":
            other.fit(volumes, rng.normal(size=centroids.shape), np.arange(len(volumes)))
        else:
            other.fit(rng.lognormal(size=len(volumes)), centroids, np.arange(len(volumes)))
        np.testing.assert_array_equal(other.effect, e)
    # the coin: the high end responds to the other arm
    flipped, *_ = _process(feature, "graded", sign=-1, p_te=1.0, p_re=0.0)
    np.testing.assert_array_equal(flipped.effect, -rule_effect(scalar, "graded"))


def test_efficacy_and_recovery_follow_the_virtual_trial():
    dgp, *_ = _process("volume", "threshold", p_te=0.8, p_re=0.1)
    e = np.array([1.0, 0.0, -1.0, 0.5])
    # mu_t = 1 - (1 - TE r_t)(1 - RE), r_1 = 0.5 + e / 2
    np.testing.assert_allclose(dgp.mu(e, 1), 1 - (1 - 0.8 * np.array([1, 0.5, 0, 0.75])) * 0.9)
    np.testing.assert_allclose(dgp.mu(e, 1) - dgp.mu(e, 0), 0.8 * 0.9 * e)
    # ranges come from the hyper-prior the virtual-trial family uses
    h = HyperPrior(p_te=(0.9, 1.0), p_re=(0.0, 0.05), label_noise=(0.0, 0.0), bias=(0.0, 0.0), rule_graded=1.0)
    rng = np.random.default_rng(4)
    for _ in range(20):
        d = FeatureRuleDGP("centroid", rng, h)
        assert 0.9 <= d.p_te <= 1 and 0 <= d.p_re <= 0.05 and d.label_noise == 0 and d.bias == 0
        assert d.shape == "graded" and d.sign in (-1, 1) and d.enrol == 1.0
    assert {FeatureRuleDGP("volume", rng, HyperPrior(rule_graded=0.0)).shape for _ in range(10)} == {"threshold"}
    assert {FeatureRuleDGP("volume", rng).shape for _ in range(40)} == {"threshold", "graded"}
    for bad in (dict(rule_graded=1.5), dict(rule_enrol=(0.0, 1.0)), dict(rule_enrol=(0.8, 0.5))):
        with pytest.raises(ValueError):
            FeatureRuleDGP("volume", rng, HyperPrior(**bad))
    with pytest.raises(ValueError):
        FeatureRuleDGP("area", rng)


def test_rule_dataset_draws_outcomes_and_allocation_like_the_virtual_trial():
    m = 3000
    dgp, volumes, centroids, rng = _process("volume", "threshold", m=m, p_te=1.0, p_re=0.0, bias=0.0,
                                            bias_type="axis", label_noise=0.0)
    idx = np.arange(m)
    X = rng.normal(size=(m, 4))
    d = make_rule_dataset(dgp, idx[:2400], idx[2400:], volumes, X[:2400], X[2400:], rng)
    e_c, e_q = dgp.effect[:2400], dgp.effect[2400:]
    assert set(np.unique(d["Yc"])) == {0.0, 1.0} and abs(d["Tc"].mean() - 0.5) < 0.04     # randomised at strength 0
    np.testing.assert_allclose(d["mu1"] - d["mu0"], e_q)
    np.testing.assert_array_equal(d["mu_q"], np.where(d["Tq"] == 1, d["mu1"], d["mu0"]))
    # all-or-none at the ends: the outcome is the arm's fit; the middle is a coin
    np.testing.assert_array_equal(d["Yc"][e_c == 1], d["Tc"][e_c == 1])
    np.testing.assert_array_equal(d["Yc"][e_c == -1], 1 - d["Tc"][e_c == -1])
    assert abs(d["Yc"][e_c == 0].mean() - 0.5) < 0.06
    np.testing.assert_array_equal(d["u_ctx"], np.sign(e_c))
    # allocation: a ramp along the volume rank (severity) or the centroid coordinate (axis), or towards the fit
    sev, *_ = _process("volume", "graded", m=m, bias=0.4, bias_type="severity")
    p = sev.propensity(sev.effect, sev.coordinate, volumes)
    assert abs(np.corrcoef(p, average_ranks(volumes))[0, 1]) > 0.999 and p.min() >= 0.1 - 1e-9 and p.max() <= 0.9 + 1e-9
    axis, *_ = _process("volume", "graded", m=m, bias=0.4, bias_type="axis")
    assert abs(np.corrcoef(axis.propensity(axis.effect, axis.coordinate, volumes), average_ranks(axis.coordinate))[0, 1]) > 0.999
    agn, *_ = _process("centroid", "threshold", m=m, bias=0.9, bias_type="agnostic")
    p = agn.propensity(agn.effect, agn.coordinate, volumes)
    assert set(np.round(np.unique(p), 6)) == {0.1, 0.5, 0.9} and (p[agn.effect > 0] == 0.9).all()
    # the probability itself for a continuous-outcome process
    cont = FeatureRuleDGP("volume", np.random.default_rng(0), HyperPrior(continuous_outcome=1.0), shape="graded")
    cont.fit(volumes, centroids, idx)
    dc = make_rule_dataset(cont, idx[:50], idx[50:60], volumes, X[:50], X[50:60], rng)
    np.testing.assert_allclose(dc["Yc"], np.where(dc["Tc"] == 1, cont.mu(cont.effect[:50], 1), cont.mu(cont.effect[:50], 0)))
    # a lesion that was not enrolled cannot be drawn
    part = FeatureRuleDGP("volume", rng, shape="graded").fit(volumes, centroids, idx[:100])
    with pytest.raises(ValueError):
        make_rule_dataset(part, idx[90:110], idx[:10], volumes, X[90:110], X[:10], rng)


# ----------------------------------------------------- the cohort's mixture
@pytest.fixture(scope="module")
def cohort_kw(tmp_path_factory):
    path = _synthetic_cache(tmp_path_factory.mktemp("pool") / "pool.npz", m=300)
    with np.load(path) as z:
        cache = {k: z[k] for k in z.files}
    atlas = FunctionalAtlas.synthetic(n_networks=8, seed=0)
    h = HyperPrior(random_partitions=1.0, threshold=(0.03, 0.08), p_te=(1.0, 1.0), p_re=(0.0, 0.0),
                   label_noise=(0.0, 0.0), bias=(0.0, 0.0))
    return dict(atlas=atlas, lesion_pool=None, cache=cache, n_context=96, n_query=16, hyper_giles=h, augment=False)


def test_single_rule_families_draw_the_definition(cohort_kw):
    assert FAMILIES == ("theory", "giles", "mixture", "volume", "centroid")
    assert MIXTURE_FAMILIES == ("theory", "giles", "volume", "centroid")
    kw = {k: v for k, v in cohort_kw.items() if k != "hyper_giles"}
    volumes = np.asarray(cohort_kw["cache"]["volumes"], float)
    coord = principal_coordinate(cohort_kw["cache"]["centroids"])
    for fam, scalar in (("volume", volumes), ("centroid", coord)):
        coh = NeuroPriorCohort(seed=5, family=fam, hyper=cohort_kw["hyper_giles"], **kw)
        assert isinstance(coh.hyper, HyperPrior)
        b = coh.sample_batch(12)
        shapes = set()
        for i, p in enumerate(b["processes"]):
            assert p["family"] == fam and p["feature"] == fam and p["n_slice"] == 300
            shapes.add(p["shape"])
            # the queries are pool lesions: find them by their latents and compare with the rule on the whole pool
            rows = [int(np.flatnonzero((coh.X == x).all(axis=1))[0]) for x in b["Xq"][i]]
            e = p["sign"] * rule_effect(scalar, p["shape"])[rows]
            np.testing.assert_allclose(b["mu1"][i], 0.5 + e / 2)
            np.testing.assert_allclose(b["mu0"][i], 0.5 - e / 2)
            assert set(np.unique(b["Yc"][i])) <= {0.0, 1.0}
        assert shapes == {"threshold", "graded"}
    # a single family whose ranges are not given takes the virtual trial's defaults
    assert isinstance(NeuroPriorCohort(seed=0, family="volume", **kw).hyper, HyperPrior)


def test_mixture_weights_over_four_families_and_weight_zero_excludes(cohort_kw):
    def drawn(weights, n=40, **extra):
        coh = NeuroPriorCohort(seed=7, family="mixture", family_weights=weights, **{**cohort_kw, **extra})
        return [p["family"] for _ in range(n) for p in coh.sample_batch(4)["processes"]]

    fams = drawn((0, 1, 1, 2))
    share = {f: fams.count(f) / len(fams) for f in MIXTURE_FAMILIES}
    assert share["theory"] == 0 and abs(share["giles"] - 0.25) < 0.1 and abs(share["volume"] - 0.25) < 0.1
    assert abs(share["centroid"] - 0.5) < 0.12
    for left_out, weights in (("volume", (0, 1, 0, 1)), ("centroid", (0, 1, 1, 0)), ("giles", (0, 0, 1, 1))):
        fams = drawn(weights, n=25)
        assert left_out not in fams and "theory" not in fams and len(set(fams)) == 2
    assert set(drawn((0, 0, 1, 0), n=5)) == {"volume"}
    assert set(drawn((0, 1, 0, 0), n=5)) == {"giles"} == set(drawn((0, 1), n=5))
    for bad in ((1.0,), (1, 1, 1), (0, 0, 0, 0), (0, 1, -1, 1), (0, 1, 1, 1, 1)):
        with pytest.raises(ValueError):
            NeuroPriorCohort(seed=0, family="mixture", family_weights=bad, **cohort_kw)
    # a random part of the pool: the rule is fixed on the enrolled lesions only
    part = HyperPrior(p_te=(1.0, 1.0), p_re=(0.0, 0.0), label_noise=(0.0, 0.0), bias=(0.0, 0.0),
                      rule_enrol=(0.3, 0.6), rule_graded=0.0)
    coh = NeuroPriorCohort(seed=9, family="mixture", family_weights=(0, 0, 1, 1), **{**cohort_kw, "hyper_giles": part})
    b = coh.sample_batch(10)
    sizes = [p["n_slice"] for p in b["processes"]]
    assert all(90 <= n <= 180 for n in sizes) and len(set(sizes)) > 3
    tau = b["mu1"] - b["mu0"]
    assert set(np.unique(tau)) <= {-1.0, 0.0, 1.0} and 0.2 < (tau == 0).mean() < 0.47
    # the whole pool when contexts are not restricted to a slice
    allpool = NeuroPriorCohort(seed=9, family="volume", context_pool="all",
                               **{k: v for k, v in cohort_kw.items() if k != "hyper_giles"}, hyper=part)
    assert all(p["n_slice"] == 300 for p in allpool.sample_batch(3)["processes"])


def test_the_launch_environment_reaches_the_new_families(tmp_path, monkeypatch):
    """FAMILY_WEIGHTS of four numbers and the rule options in GILES_HYPER, through
    the cfg the training builds; the generic share applies to every family."""
    from neurocausalpfn.train.launch_cfg import cfg_from_env
    from neurocausalpfn.train.train_pfn import _build_prior, e12_config, with_family, reduced_config

    monkeypatch.chdir(tmp_path)
    _synthetic_cache(tmp_path / "pool.npz")
    env = dict(FINAL_PRIOR_ENV, CACHE="pool.npz", SEED="0", FAMILY_WEIGHTS="0 1 1 1",
               GILES_HYPER=FINAL_PRIOR_ENV["GILES_HYPER"] + " rule_graded=0.25 rule_enrol=0.5:1")
    cfg = cfg_from_env(env)
    assert cfg["prior"]["family_weights"] == [0.0, 1.0, 1.0, 1.0]
    assert cfg["prior"]["hyper_giles"]["rule_graded"] == 0.25 and cfg["prior"]["hyper_giles"]["rule_enrol"] == [0.5, 1.0]
    prior, _, _ = _build_prior(cfg)
    fams = [p["family"] for _ in range(30) for p in prior.sample_batch(4, n_context=128)["processes"]]
    assert {"giles", "volume", "centroid"} <= set(fams) and "theory" not in fams
    assert any(f.startswith("generic-") for f in fams)
    # a family alone, and the staged curriculum's table, know the new names
    assert with_family(reduced_config(), "centroid")["prior"]["family"] == "centroid"
    assert e12_config("ctx+stages", "volume")["prior"]["stages"][0]["hyper"] == {"bias": [0.0, 0.3]}
    with pytest.raises(ValueError):
        with_family(reduced_config(), "area")


# ------------------------------------------------------------------ the gate
def test_gate_checks_the_new_keys_and_the_absent_families():
    import prior_gate as pg

    from neurocausalpfn.train.launch_cfg import cfg_from_env

    env = dict(FINAL_PRIOR_ENV, CACHE="pool.npz", FAMILY_WEIGHTS="0 1 1 0",
               GILES_HYPER=FINAL_PRIOR_ENV["GILES_HYPER"] + " rule_graded=0.5 rule_enrol=0.2:1")
    cfg = cfg_from_env(env)
    good = ["prior.family=mixture", "prior.family_weights=0:1:1:0", "prior.hyper.generic_weight=0.5",
            "prior.hyper_giles.rule_graded=0.5", "prior.hyper_giles.rule_enrol=0.2:1",
            "prior.hyper_giles.random_partitions=1"]
    assert pg.check_expect(cfg, good) == []
    for wrong in ("prior.family_weights=0:1:0:1",            # the other family left out
                  "prior.family_weights=0:1",                # two weights where four were launched
                  "prior.hyper_giles.rule_graded=1", "prior.hyper_giles.rule_enrol=0.5:1",
                  "prior.hyper_giles.rule_enrol=UNSET"):
        errs = pg.check_expect(cfg, [wrong])
        assert len(errs) == 1 and wrong.split("=")[0] in errs[0]
    # a launch without the options: the keys are unset, and expected so
    plain = cfg_from_env(dict(FINAL_PRIOR_ENV, CACHE="pool.npz"))
    assert pg.check_expect(plain, ["prior.family_weights=0:1", "prior.hyper_giles.rule_graded=UNSET",
                                   "prior.hyper_giles.rule_enrol=UNSET"]) == []
    assert len(pg.check_expect(plain, ["prior.hyper_giles.rule_graded=0.5"])) == 1
    rows = [{"family": f} for f in ("giles", "volume", "generic-step", "giles", "generic-linear")]
    assert pg.check_absent(rows, []) == [] and pg.check_absent(rows, ["theory", "centroid"]) == []
    assert "volume" in pg.check_absent(rows, ["volume"])[0] and "1 of 5" in pg.check_absent(rows, ["volume"])[0]
    assert "2 of 5" in pg.check_absent(rows, ["generic"])[0]                 # generic-<kind> labels
    assert "unknown family" in pg.check_absent(rows, ["area"])[0]


def test_gate_runs_on_a_prior_with_the_new_families(tmp_path, monkeypatch, capsys):
    import prior_gate as pg

    monkeypatch.chdir(tmp_path)
    _synthetic_cache(tmp_path / "pool.npz")
    env = dict(FINAL_PRIOR_ENV, CACHE="pool.npz", SEED="0", FAMILY_WEIGHTS="0 1 1 0",
               PRIOR_HYPER=FINAL_PRIOR_ENV["PRIOR_HYPER"].replace("generic_weight=0.5", "generic_weight=0"))
    for k in ("GILES_HYPER", "PRIOR_HYPER", "FAMILY", "FAMILY_WEIGHTS", "CACHE", "SEED", "CONTEXT_POOL",
              "CONTEXT_MAX", "VARIANT", "N_QUERY", "STANDARDIZE", "AUGMENT", "CONTEXT_REPLACE", "BATCH"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    opened = "share_zero=0:1 share_unit=0:1 share_small=0:1 pos_share=0:1 ctx_binary=0:1 ctx_treated=0:1".split()
    base = ["prior_gate.py", "--batches", "6", "--contexts", "128", "--require", *opened]
    ok = base + ["--absent", "theory", "centroid", "generic", "--expect", "prior.family_weights=0:1:1:0",
                 "prior.hyper_giles.rule_graded=UNSET"]
    monkeypatch.setattr(sys, "argv", ok)
    pg.main()
    out = capsys.readouterr().out
    assert "prior gate ok" in out and "volume:" in out and "giles:" in out and "centroid:" not in out
    for wrong in (base + ["--absent", "volume"],                              # a family that is in the prior
                  base + ["--expect", "prior.family_weights=0:1:0:1"]):       # the wrong family left out
        monkeypatch.setattr(sys, "argv", wrong)
        with pytest.raises(SystemExit) as stop:
            pg.main()
        assert stop.value.code == 1 and "PRIOR GATE FAILED" in capsys.readouterr().out


# ------------------------------------------ second round (17 Nov, after the audit)
def test_a_lesion_without_a_finite_value_is_an_error_not_the_top_rank():
    """NaN sorts last, so it used to take the top effect in silence."""
    for bad in (np.nan, np.inf, -np.inf):
        for shape in ("threshold", "graded"):
            with pytest.raises(ValueError, match="not finite"):
                rule_effect([1.0, bad, 3.0, 2.0], shape)
    with pytest.raises(ValueError, match="not finite"):
        principal_coordinate(np.array([[0.0, 0.0, 0.0], [1.0, np.nan, 0.0], [2.0, 1.0, 0.0]]))
    rng = np.random.default_rng(0)
    volumes, centroids = rng.lognormal(size=60), rng.normal(size=(60, 3))
    vol_nan, cen_nan = volumes.copy(), centroids.copy()
    vol_nan[7], cen_nan[9, 1] = np.nan, np.nan
    everyone, without = np.arange(60), np.setdiff1d(np.arange(60), [7, 9])
    for feature in RULE_FAMILIES:
        dgp = FeatureRuleDGP(feature, np.random.default_rng(1))
        # either feature: the volume also orders the severity ramp, the centroid the axis ramp
        with pytest.raises(ValueError, match="1 of 60 enrolled lesions"):
            dgp.fit(vol_nan, centroids, everyone)
        with pytest.raises(ValueError, match="1 of 60 enrolled lesions"):
            dgp.fit(volumes, cen_nan, everyone)
        # a lesion that is not enrolled is not ranked, so it does not stop the process
        dgp.fit(vol_nan, cen_nan, without)
        assert np.isfinite(dgp.effect[without]).all() and np.isnan(dgp.effect[[7, 9]]).all()
    # with the whole pool's axis every centroid of the pool is read
    with pytest.raises(ValueError, match="not finite"):
        FeatureRuleDGP("centroid", np.random.default_rng(1), axis_from="pool").fit(volumes, cen_nan, without)


def test_a_pool_with_a_nan_volume_stops_a_rule_process_and_no_other(cohort_kw):
    cache = dict(cohort_kw["cache"])
    cache["volumes"] = np.asarray(cache["volumes"], float).copy()
    cache["volumes"][3] = np.nan
    kw = dict(cohort_kw, cache=cache)
    with pytest.raises(ValueError, match="no finite volume and centroid"):
        NeuroPriorCohort(seed=0, family="mixture", family_weights=[0, 0, 1, 0], **kw).sample_batch(2)
    # the families of the final prior do not rank volumes: as before the change
    b = NeuroPriorCohort(seed=0, family="mixture", family_weights=[0, 1, 0, 0], **kw).sample_batch(2)
    assert {p["family"] for p in b["processes"]} == {"giles"}


def test_the_new_options_default_to_what_the_branch_did(tmp_path):
    """Volume and centroid processes drawn with the options of 17 Nov unset
    are, draw for draw, those of this branch before the options existed."""
    (tmp_path / "first_round").mkdir()
    before, now = _src_at(FIRST_ROUND, tmp_path / "first_round"), os.path.join(ROOT, "src")
    _synthetic_cache(tmp_path / "pool.npz")
    final = dict(FINAL_PRIOR_ENV, CACHE="pool.npz", SEED="0")
    mix = dict(final, FAMILY_WEIGHTS="0 1 1 1")
    enrol = dict(final, FAMILY_WEIGHTS="0 0 1 1",
                 GILES_HYPER=final["GILES_HYPER"] + " rule_graded=0.25 rule_enrol=0.4:1")
    drawn = _draw(now, tmp_path, "mix_now", **mix)
    _same(_draw(before, tmp_path, "mix_before", **mix), drawn)
    _same(_draw(before, tmp_path, "enrol_before", **enrol), _draw(now, tmp_path, "enrol_now", **enrol))
    assert {"volume", "centroid"} <= set(drawn["families"].tolist())         # the new families were drawn
    # and the defaults, written out, are the defaults
    explicit = dict(mix, GILES_HYPER=final["GILES_HYPER"] + " rule_enrol_by=share rule_axis=enrolled")
    _same(_draw(now, tmp_path, "explicit", **explicit), drawn)


def _rows(coh, X):
    return np.array([int(np.flatnonzero((coh.X == x).all(axis=1))[0]) for x in X])


def test_enrolment_by_network_cuts_the_rule_within_each_trials_slice(cohort_kw):
    """rule_enrol_by=network: a process enrols the slice of a virtual-trial
    process (the lesions its networks reach), so thirds and ranks are those of
    a different set in each trial. The default enrols the whole pool: every
    process of a family then ranks the same lesions."""
    kw = {k: v for k, v in cohort_kw.items() if k != "hyper_giles"}
    volumes = np.asarray(cohort_kw["cache"]["volumes"], float)
    h = cohort_kw["hyper_giles"]
    net = HyperPrior(**{**h.__dict__, "rule_enrol_by": "network", "rule_graded": 0.0})
    coh = NeuroPriorCohort(seed=3, family="volume", hyper=net, **kw)
    sizes, cuts = set(), set()
    for _ in range(12):
        fam, dgp, sl = coh._draw_process()
        assert dgp.enrol_by == "network" and coh.min_slice <= len(sl) < coh.m
        # the slice is one a virtual-trial process could enrol: every lesion in it is reached by some network
        assert (np.asarray(cohort_kw["cache"]["overlaps"]).reshape(coh.m, -1)[sl] > 0).any(axis=1).all()
        # the rule is the definition applied within the slice, and is not defined outside it
        np.testing.assert_array_equal(dgp.effect[sl], dgp.sign * rule_effect(volumes[sl], "threshold"))
        assert np.isnan(np.delete(dgp.effect, sl)).all()
        assert abs(int((dgp.effect[sl] == 1).sum()) - int((dgp.effect[sl] == -1).sum())) == 0   # thirds of the slice
        sizes.add(len(sl))
        cuts.add(float(volumes[sl][dgp.sign * dgp.effect[sl] == 1].min()))                     # where the top third starts
    assert len(sizes) > 3 and len(cuts) > 3                                  # the cut moves from trial to trial
    b = coh.sample_batch(6)
    for i, p in enumerate(b["processes"]):
        assert p["enrol_by"] == "network" and p["n_slice"] < coh.m
        assert set(np.unique(b["mu1"][i] - b["mu0"][i])) <= {-1.0, 0.0, 1.0}
    # the default: one set of lesions, one cut
    whole = NeuroPriorCohort(seed=3, family="volume", hyper=HyperPrior(**{**h.__dict__, "rule_graded": 0.0}), **kw)
    cuts = set()
    for _ in range(12):
        fam, dgp, sl = whole._draw_process()
        assert len(sl) == whole.m and dgp.enrol_by == "share"
        cuts.add(float(volumes[dgp.sign * dgp.effect == 1].min()))
    assert len(cuts) == 1
    # with every lesion a candidate (context_pool "all") no family restricts its rows, and neither does this one
    everyone = NeuroPriorCohort(seed=3, family="volume", hyper=net, context_pool="all", **kw)
    fam, dgp, sl = everyone._draw_process()
    assert sl is None and np.isfinite(dgp.effect).all()
    # the two ways of enrolling do not combine
    both = HyperPrior(**{**h.__dict__, "rule_enrol_by": "network", "rule_enrol": (0.5, 1.0)})
    with pytest.raises(ValueError, match="rule_enrol"):
        NeuroPriorCohort(seed=3, family="volume", hyper=both, **kw).sample_batch(1)
    with pytest.raises(ValueError, match="rule_enrol_by"):
        FeatureRuleDGP("volume", np.random.default_rng(0), enrol_by="cohort")


def test_the_principal_axis_is_the_enrolled_lesions_or_the_pools():
    rng = np.random.default_rng(4)
    m = 200
    # a pool that is long in x; the enrolled part (a thin slab of it) is long in y
    centroids = np.column_stack([rng.normal(scale=10, size=m), rng.normal(scale=3, size=m), rng.normal(size=m)])
    volumes = rng.lognormal(size=m)
    part = np.flatnonzero(np.abs(centroids[:, 0]) < 2.0)
    assert 20 < len(part) < 100
    kw = dict(shape="graded", sign=1, p_te=1.0, p_re=0.0)
    own = FeatureRuleDGP("centroid", np.random.default_rng(0), **kw).fit(volumes, centroids, part)
    pool = FeatureRuleDGP("centroid", np.random.default_rng(0), axis_from="pool", **kw).fit(volumes, centroids, part)
    assert own.axis_from == "enrolled" and own.describe()["axis"] == "enrolled" and pool.describe()["axis"] == "pool"
    # default: the first principal axis of the enrolled centroids, which here is y
    np.testing.assert_allclose(own.coordinate[part], principal_coordinate(centroids[part]))
    assert abs(np.corrcoef(own.coordinate[part], centroids[part, 1])[0, 1]) > 0.95
    # pool: the pool's axis (x), read at the enrolled lesions; ranks and thirds are still the enrolled lesions'
    np.testing.assert_allclose(pool.coordinate[part], principal_coordinate(centroids)[part])
    assert abs(np.corrcoef(pool.coordinate[part], centroids[part, 0])[0, 1]) > 0.95
    np.testing.assert_array_equal(pool.effect[part], rule_effect(principal_coordinate(centroids)[part], "graded"))
    assert pool.effect[part].min() == -1.0 and pool.effect[part].max() == 1.0
    assert not np.allclose(own.effect[part], pool.effect[part])
    # the whole pool enrolled: the two options are the same process
    everyone = np.arange(m)
    a = FeatureRuleDGP("centroid", np.random.default_rng(0), **kw).fit(volumes, centroids, everyone)
    b = FeatureRuleDGP("centroid", np.random.default_rng(0), axis_from="pool", **kw).fit(volumes, centroids, everyone)
    np.testing.assert_array_equal(a.effect, b.effect)
    # the volume family does not read the axis for its effect
    v1 = FeatureRuleDGP("volume", np.random.default_rng(0), **kw).fit(volumes, centroids, part)
    v2 = FeatureRuleDGP("volume", np.random.default_rng(0), axis_from="pool", **kw).fit(volumes, centroids, part)
    np.testing.assert_array_equal(v1.effect[part], v2.effect[part])
    with pytest.raises(ValueError, match="rule_axis"):
        FeatureRuleDGP("centroid", np.random.default_rng(0), axis_from="trial")


def test_the_direction_of_the_axis_is_fixed_by_a_rule_and_does_not_change_the_family():
    rng = np.random.default_rng(5)
    c = rng.normal(size=(150, 3)) * np.array([1.0, 6.0, 2.0])
    coord = principal_coordinate(c)
    # deterministic: the same call, the rows in another order, the points mirrored
    np.testing.assert_array_equal(principal_coordinate(c), coord)
    perm = rng.permutation(len(c))
    np.testing.assert_allclose(principal_coordinate(c[perm]), coord[perm], atol=1e-9)
    np.testing.assert_allclose(principal_coordinate(-c), -coord, atol=1e-9)            # the axis keeps its direction
    # the rule: the component of the axis that is largest in size is positive (here y)
    assert np.corrcoef(coord, c[:, 1])[0, 1] > 0.9
    # the opposite convention would only rename the processes: the high end with sign +1 is the low end with -1,
    # and the sign is a fair coin per process
    for shape in ("threshold", "graded"):
        np.testing.assert_array_equal(rule_effect(-coord, shape), -rule_effect(coord, shape))
    signs = [FeatureRuleDGP("centroid", np.random.default_rng(s)).sign for s in range(400)]
    assert set(signs) == {-1, 1} and abs(np.mean(signs)) < 0.15


def test_the_axis_ramp_has_its_own_direction_unlike_the_virtual_trial():
    """In the virtual trial the axis ramp points to the half that responds to
    arm 1. Here its direction is a coin of its own: over many centroid
    processes allocation leans with the effect about as often as against."""
    rng = np.random.default_rng(6)
    centroids, volumes = rng.normal(size=(300, 3)) * np.array([5.0, 1.0, 1.0]), rng.lognormal(size=300)
    idx, leans = np.arange(300), []
    for s in range(200):
        dgp = FeatureRuleDGP("centroid", np.random.default_rng(s), shape="graded", bias=0.9, bias_type="axis",
                             p_te=1.0, p_re=0.0).fit(volumes, centroids, idx)
        p = dgp.propensity(dgp.effect, dgp.coordinate, volumes)
        leans.append(np.sign(np.corrcoef(p, dgp.effect)[0, 1]))               # +1: arm 1 goes to those it suits
        assert dgp.ramp_sign * dgp.sign == leans[-1]
    assert set(leans) == {-1.0, 1.0} and abs(np.mean(leans)) < 0.2


def test_the_new_options_travel_from_the_launch_environment(tmp_path, monkeypatch):
    from neurocausalpfn.train.launch_cfg import cfg_from_env
    from neurocausalpfn.train.train_pfn import _build_prior

    monkeypatch.chdir(tmp_path)
    _synthetic_cache(tmp_path / "pool.npz")
    env = dict(FINAL_PRIOR_ENV, CACHE="pool.npz", SEED="0", FAMILY_WEIGHTS="0 0 1 1",
               GILES_HYPER=FINAL_PRIOR_ENV["GILES_HYPER"] + " rule_enrol_by=network rule_axis=pool")
    cfg = cfg_from_env(env)
    assert cfg["prior"]["hyper_giles"]["rule_enrol_by"] == "network" and cfg["prior"]["hyper_giles"]["rule_axis"] == "pool"
    prior, _, _ = _build_prior(cfg)
    procs = [p for _ in range(6) for p in prior.sample_batch(4, n_context=128)["processes"]]
    rule = [p for p in procs if p["family"] in RULE_FAMILIES]
    assert rule and all(p["enrol_by"] == "network" and p["axis"] == "pool" for p in rule)
    assert len({p["n_slice"] for p in rule}) > 1
    import prior_gate as pg

    assert pg.check_expect(cfg, ["prior.hyper_giles.rule_enrol_by=network", "prior.hyper_giles.rule_axis=pool"]) == []
    assert len(pg.check_expect(cfg, ["prior.hyper_giles.rule_enrol_by=UNSET"])) == 1
    assert len(pg.check_expect(cfg, ["prior.hyper_giles.rule_axis=enrolled"])) == 1


def test_the_gates_default_requirements_do_not_fit_these_families(tmp_path, monkeypatch, capsys):
    """PRIOR_REPAIR=1 passes no requirement, so the gate applies its defaults:
    the class shares of an all-or-none trial. A threshold rule gives thirds
    and a graded rule no exact zero, so a prior of these families fails
    share_zero there. The defaults stay; the launch must name the ranges
    (g4c_launch.sh: GATE_REQUIRE)."""
    import prior_gate as pg

    monkeypatch.chdir(tmp_path)
    _synthetic_cache(tmp_path / "pool.npz")
    repaired = FINAL_PRIOR_ENV["GILES_HYPER"].replace("p_te=0.9:1 p_re=0:0.05", "p_te=1:1 p_re=0:0")
    for k in ("GILES_HYPER", "PRIOR_HYPER", "FAMILY", "FAMILY_WEIGHTS", "CACHE", "SEED", "CONTEXT_POOL",
              "CONTEXT_MAX", "VARIANT", "N_QUERY", "STANDARDIZE", "AUGMENT", "CONTEXT_REPLACE", "BATCH"):
        monkeypatch.delenv(k, raising=False)
    env = dict(FINAL_PRIOR_ENV, CACHE="pool.npz", SEED="0", FAMILY_WEIGHTS="0 0 1 1", GILES_HYPER=repaired,
               PRIOR_HYPER=FINAL_PRIOR_ENV["PRIOR_HYPER"].replace("generic_weight=0.5", "generic_weight=0"))
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert pg.DEFAULT_REQUIRE["share_zero"] == (0.35, 0.65) and pg.DEFAULT_REQUIRE["share_unit"] == (0.30, 0.65)
    base = ["prior_gate.py", "--batches", "6", "--contexts", "128", "--expect", "prior.family_weights=0:0:1:1"]
    monkeypatch.setattr(sys, "argv", base)                                   # the defaults, as PRIOR_REPAIR=1 leaves them
    with pytest.raises(SystemExit) as stop:
        pg.main()
    out = capsys.readouterr().out
    assert stop.value.code == 1 and "PRIOR GATE FAILED" in out and "share_zero" in out.split("PRIOR GATE FAILED")[1]
    # the two class shares named for this launch; the other requirements keep the gate's defaults
    monkeypatch.setattr(sys, "argv", base + ["--require", "share_zero=0:1", "share_unit=0:1"])
    pg.main()
    assert "prior gate ok" in capsys.readouterr().out


# ------------------------------------------ third round (17 Nov, after the recheck)
def test_a_random_axis_direction_draws_both_and_only_for_centroid_processes(tmp_path):
    rng = np.random.default_rng(7)
    centroids, volumes = rng.normal(size=(120, 3)) * np.array([1.0, 5.0, 1.0]), rng.lognormal(size=120)
    idx, coord = np.arange(120), principal_coordinate(centroids)
    fixed, random = HyperPrior(), HyperPrior(rule_axis_sign="random")
    assert fixed.rule_axis_sign == "fixed"
    seen = []
    for s in range(300):
        dgp = FeatureRuleDGP("centroid", np.random.default_rng(s), random, shape="graded").fit(volumes, centroids, idx)
        seen.append(dgp.axis_sign)
        assert dgp.describe()["axis_sign"] == dgp.axis_sign
        # the coordinate is the principal coordinate, or its mirror image; the rule is cut on it
        np.testing.assert_allclose(dgp.coordinate, dgp.axis_sign * coord)
        np.testing.assert_array_equal(dgp.effect, dgp.sign * rule_effect(dgp.axis_sign * coord, "graded"))
    assert set(seen) == {-1, 1} and abs(np.mean(seen)) < 0.15                # both directions, about half each
    # the default: one direction, the rule of principal_coordinate, and no draw for it:
    # a process built with the default leaves the generator where the code before the option left it
    for feature, hyper in (("centroid", fixed), ("volume", fixed), ("volume", random)):
        for s in range(20):
            a, b = np.random.default_rng(s), np.random.default_rng(s)
            dgp = FeatureRuleDGP(feature, a, hyper)
            ref = FeatureRuleDGP(feature, b, fixed, axis_sign=1)             # axis_sign given: never drawn
            assert dgp.axis_sign == 1 and a.bit_generator.state == b.bit_generator.state
            assert (dgp.shape, dgp.sign, dgp.p_te, dgp.bias_type, dgp.ramp_sign) == \
                   (ref.shape, ref.sign, ref.p_te, ref.bias_type, ref.ramp_sign)
    # a centroid process with the random direction makes exactly one more draw, after all the others
    a, b = np.random.default_rng(3), np.random.default_rng(3)
    one, ref = FeatureRuleDGP("centroid", a, random), FeatureRuleDGP("centroid", b, fixed)
    assert (one.shape, one.sign, one.p_te, one.ramp_sign, one.enrol) == (ref.shape, ref.sign, ref.p_te, ref.ramp_sign, ref.enrol)
    assert a.bit_generator.state != b.bit_generator.state
    with pytest.raises(ValueError, match="rule_axis_sign"):
        FeatureRuleDGP("centroid", np.random.default_rng(0), HyperPrior(rule_axis_sign="coin"))


def test_the_axis_direction_option_changes_nothing_without_centroid_processes(tmp_path):
    """Draw for draw: the option at its default written out, and the option
    at ``random`` with the centroid family at weight 0, are the prior without
    the option; with centroid processes drawn it is another prior."""
    now = os.path.join(ROOT, "src")
    _synthetic_cache(tmp_path / "pool.npz")
    final = dict(FINAL_PRIOR_ENV, CACHE="pool.npz", SEED="0")
    for name, weights in (("vol", "0 1 1 0"), ("all", "0 1 1 1")):
        plain = _draw(now, tmp_path, name, **dict(final, FAMILY_WEIGHTS=weights))
        fixed = _draw(now, tmp_path, name + "_fixed", **dict(
            final, FAMILY_WEIGHTS=weights, GILES_HYPER=final["GILES_HYPER"] + " rule_axis_sign=fixed"))
        _same(plain, fixed)
        coin = _draw(now, tmp_path, name + "_random", **dict(
            final, FAMILY_WEIGHTS=weights, GILES_HYPER=final["GILES_HYPER"] + " rule_axis_sign=random"))
        if name == "vol":
            assert "volume" in set(plain["families"].tolist()) and "centroid" not in set(plain["families"].tolist())
            _same(plain, coin)                                               # no centroid process: no draw, no change
        else:
            assert "centroid" in set(plain["families"].tolist())
            assert any(not np.array_equal(plain[k], coin[k]) for k in plain)


def test_enrolment_by_trial_cuts_the_rule_among_the_trials_own_lesions(cohort_kw):
    """rule_enrol_by=trial: rows are drawn as with ``share`` and the ranks,
    thirds and axis are those of the distinct lesions in this trial's context
    and queries, so every trial has thirds of its own rows."""
    kw = {k: v for k, v in cohort_kw.items() if k != "hyper_giles"}
    volumes = np.asarray(cohort_kw["cache"]["volumes"], float)
    h = cohort_kw["hyper_giles"]
    coh = NeuroPriorCohort(seed=4, family="volume", hyper=HyperPrior(**{**h.__dict__, "rule_enrol_by": "trial",
                                                                        "rule_graded": 0.0}), **kw)
    cuts, differ = set(), 0
    for _ in range(10):
        b = coh.sample_batch(1)
        p = b["processes"][0]
        ci, qi = _rows(coh, b["Xc"][0]), _rows(coh, b["Xq"][0])
        own = np.unique(np.concatenate([ci, qi]))
        assert p["enrol_by"] == "trial" and p["n_slice"] == coh.m and p["n_ranked"] == len(own) < coh.m
        assert len(own) < len(ci) + len(qi)                                  # the context repeats lesions: ranked once
        # the effect of each query is the rule applied among this trial's lesions, not among the pool's
        e_own = dict(zip(own.tolist(), (p["sign"] * rule_effect(volumes[own], "threshold")).tolist()))
        np.testing.assert_allclose(b["mu1"][0] - b["mu0"][0], [e_own[i] for i in qi.tolist()])
        e_pool = p["sign"] * rule_effect(volumes, "threshold")
        cuts.add(float(volumes[own][np.array([e_own[i] for i in own.tolist()]) * p["sign"] == 1].min()))
        # thirds of the trial's own lesions: as many at the top as at the bottom
        vals = np.array([e_own[i] for i in own.tolist()])
        assert (vals == 1).sum() == (vals == -1).sum() and abs((vals == 0).sum() - len(own) / 3) <= 2
        differ += int(not np.array_equal([e_own[i] for i in qi.tolist()], e_pool[qi]))
    assert len(cuts) > 3                                                     # the cut moves from trial to trial
    assert differ > 0                                                        # so it is not the pool's rule
    # a share of the pool as candidates, then the same cut among the trial's lesions
    part = NeuroPriorCohort(seed=4, family="centroid", hyper=HyperPrior(**{
        **h.__dict__, "rule_enrol_by": "trial", "rule_enrol": (0.5, 0.6), "rule_graded": 1.0}), **kw)
    b = part.sample_batch(3)
    for i, p in enumerate(b["processes"]):
        assert 0.5 * part.m - 1 <= p["n_slice"] <= 0.6 * part.m + 1 and p["n_ranked"] < p["n_slice"]
        tau = b["mu1"][i] - b["mu0"][i]
        assert np.abs(tau).max() <= 1.0 and len(np.unique(np.round(tau, 9))) > 10      # graded, within the trial
    # with every lesion a candidate the rows come from the whole pool and are ranked the same way
    everyone = NeuroPriorCohort(seed=4, family="volume", context_pool="all", hyper=HyperPrior(**{
        **h.__dict__, "rule_enrol_by": "trial", "rule_graded": 0.0}), **kw)
    b = everyone.sample_batch(2)
    assert all(p["n_ranked"] <= 96 + 16 for p in b["processes"])
    assert set(np.unique(b["mu1"] - b["mu0"])) <= {-1.0, 0.0, 1.0}
