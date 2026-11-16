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
KEYS = ("Xc", "Tc", "Yc", "Xq", "Tq", "mu_q", "mu0", "mu1")

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


@pytest.fixture(scope="module")
def base_src(tmp_path_factory):
    """src/ of the code before the change, taken from git history."""
    if shutil.which("git") is None:
        pytest.skip("needs git")
    r = subprocess.run(["git", "archive", "--format=tar", BASE_COMMIT, "src/neurocausalpfn"], cwd=ROOT,
                       capture_output=True)
    if r.returncode != 0:
        pytest.skip(f"commit {BASE_COMMIT[:7]} is not in this clone's history")
    d = tmp_path_factory.mktemp("base_code")
    with tarfile.open(fileobj=io.BytesIO(r.stdout)) as tar:
        try:
            tar.extractall(d, filter="data")
        except TypeError:                                    # a Python without the filter argument
            tar.extractall(d)
    assert not os.path.exists(d / "src" / "neurocausalpfn" / "prior" / "feature_rules.py")
    return str(d / "src")


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
