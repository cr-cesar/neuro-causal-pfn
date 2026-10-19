# Audit of the train and score path of the Neuro-Causal PFN

Read-only audit. Aggregate findings only: no patient identifiers, no image file names, no latent values.

## Scope and version

- Audited tree: `~/Scratch/neuro-causal-pfn` on Myriad, branch `results-snapshot`, HEAD `a557021`, with uncommitted edits to `pfn/estimator.py` and `prior/giles_replica.py`.
- The `src` of that tree is identical to `origin/claude/phase-1-vae-encoder-f02ugc` at `5a1b278`, so the audit applies to that branch.
- Paths below are under `src/neurocausalpfn`. Aliases: `coh` = `prior/cohort.py`, `ist` = `prior/intersynth_theory.py`, `np` = `prior/neuro_prior.py`, `gr` = `prior/giles_replica.py`, `est` = `pfn/estimator.py`, `tab` = `pfn/tabicl_model.py`, `trn` = `train/train_pfn.py`.
- No job was launched. Tests ran on the login node at the request of the audit prompt, 8.8 s.

## Tests

Command: `python -m pytest -q tests/test_replica_pfn_estimator.py tests/test_giles_replica.py tests/test_e12.py`

Result: 21 passed, 1 failed in 8.8 s.

The failure is `test_off_the_shelf_causalpfn_wrapper_returns_potential_outcomes`. It stops inside the causalpfn package (0.1.4, `causal_estimator.py:397`), where scikit-learn 1.4.2 rejects `min_samples_leaf=0` for a `GradientBoostingRegressor`. The test passes only 60 rows. Real contexts have at least 438 rows. The failure is not on the path of our own PFN.

## What could not be checked

No `replica_results.csv` on disk, for the PFN runs or for CausalPFN, contains the effect decomposition columns (`tau_slope`, `rescale_factor`, `n_pos`, `n_zero`, `n_neg`, `tau_hat_*`). The measured ratio of 0.4 therefore could not be corroborated from files. Any rescoring with the current code records them (`gr:341-384`); a linear PFN scoring takes about 6 minutes.

## Differences between a training batch and a scoring call

| Difference | Train side | Score side | Magnitude | Ranking | Cheap CPU test |
|---|---|---|---|---|---|
| Covariate scale, linear arch | coh:241-243, pool standardised once, slices keep pool statistics | est:100-102, context mean and sd plus 1e-6 | possible | possible, columns rescaled by factors never seen | script S, variant slice_standardised |
| Covariate scale, TabICL | tab:127-137, standardises with context statistics | same function at scoring; the estimator pass is a no-op to 1e-6 | none | none | not needed |
| Random rotation | coh:280-282 and 346-347, one orthogonal matrix per process shared by context and queries; off with AUGMENT=0 | none, axes as the encoder emits them | none directly | yes if the net uses axes or is brittle | script S, variant rotated |
| Source of Z and dtype | float32 cache from one encoder on the pool, build_prior_cache.py:52, coh:211-218, float32 in pfn/tokens.py:33 | per-fold Ztr and Zte, ncpfold/score.py:81-124, float32 | none | yes if the geometry differs | ridge R2 of overlaps on Z, pool vs fold |
| Column order | not reordered | not reordered | not found | not found | not needed |
| Treatment encoding | float 0/1, ist:285; Tq random per query, ist:293 | W as float32, est:104; Tq all zeros then all ones, pfn/inference.py:19-22 | none | none | not needed |
| Outcome type | continuous with noise in theory unless binary_outcome (default 0), ist:85 and 287-290; binary only in giles, np:208 | binary, noise-free at TE 1 and RE 0, gr:71 and 249-259 | possible for theory-only checkpoints | possible | script S, variant binary_Y |
| Outcome scaling | not found, pfn/model.py:38 and tab:138 | not found | none | none | not needed |
| Head support and target | pfn/cepo_ppd.py:21 and 29-34, support 0 to 1, sigma 0.02 | pfn/inference.py:23-25, est:111, raw p1 minus p0 against tau, gr:296 | tau_hat capped at 0.968 for 256 and 1024 bins (computed), at most about 3 percent | none | closed form |
| Posterior-mean target | cepo_ppd.py:39-41 fits a distribution over mu | difference of two posterior means | shrinks whenever the posterior is wide | yes, near-zero posteriors rank by noise | script S, extra_trees line |
| Context composition | trn:119-128 and 234-238, 256 up to a cap of 1024; drawn with replacement from the slice minus queries, coh:342-344, so duplicates with independent T and Y, ist:285-290 | whole train slice of unique patients, subsampled to the checkpoint context_max, est:17-24, 32-48 and 80-81 | unknown | unknown | rescore with np.repeat on the context |
| Treated share | ist:257-263, propensity clipped to 0.01-0.99 | gr:187-190, random half at BIAS 0 | none | none | inside support; BIAS 1 (gr:199-217) is outside the clip |
| Query count | trn:109, 64 | est:107-110, chunks up to 512 | none | none | not found: attention.py:18 and tab:170 isolate queries; test max_queries 1 vs 512 |
| Slice definition | theory: raw D above 0, coh:293, D a weighted overlap over all networks, ist:177, so almost the whole pool; giles: threshold in 1 to 3 networks, coh:289-291 | more than 5 percent overlap of one half of one deficit pair, gr:154 and 163 | yes | yes | prior_diagnostics, below |
| Effect scale | ist:158 and 183-194, coh:325; pool-wide 95th percentile of the absolute CATE equals a log-uniform target | absolute tau is 1 for single-hit patients | yes if the scale is not identified | secondary | below |
| Giles-family ranges | np:39-50 and 93-103, p_te 0.3 to 1, p_re 0 to 0.5 | TE 1, RE 0, gr:71 | the trial sits at the edge | none | retrain with narrow ranges (GPU) |
| Precision | fp16 autocast on CUDA, utils/runtime.py:34-46 | fp32, no autocast | negligible | negligible | score one batch under autocast |
| Determinism | | eval mode, dropout 0, model.py:22; seeded subsample, est:41; the trial reseeds the global numpy generator per fold, gr:181 and 188, so all estimators see the same W and Y | none | none | not needed |

Repeats of a tested image in the context: giles mode adds later acquisitions to training in every fold (gr:515-516), but the headline protocol tests only single-image patients (ncpfold/score.py:161-171). No leakage found for the headline.

## Does the context identify the process scale?

Only through the outcome contrast between arms, mixed with confounding (ist:257-263) and with the unknown map from anatomy to the readout weights. Those weights are Dirichlet(1) over 16 networks (ist:143-145 and 156), and the CATE sign comes from S_A minus S_B (ist:236-238). While the net cannot locate the partition, the Bayes-optimal predictor under squared loss returns the posterior mean, which is below the true effect. The mean of a log-uniform target between 0.2 and 1 is 0.5.

## Three most likely explanations of the 0.36

### 1. Training slice and effect geometry differ from the trial

In the theory family nearly the whole pool counts as susceptible (coh:293) and the scale is set on the whole pool (coh:325). A typical patient of a training slice has an effect near zero, and ranking inside that group is noise. In the trial every slice patient is hit and the absolute tau is 1 for single-hit patients. This fits the observation that the 0.4 does not change with effect scale, pool or context length, because none of them changes how many patients sit near zero.

Probe of 13 October, family giles:

| checkpoint | corr in own family | corr in giles | mean prediction | mean truth |
|---|---|---|---|---|
| theory linear | 0.81 | 0.48 | 0.052 | 0.150 |
| mixture linear | 0.78 | 0.59 | 0.114 | 0.150 |
| giles linear | 0.22 | 0.59 | 0.107 | 0.150 |

Even the giles-trained checkpoint predicts about 70 percent of the magnitude in its own family.

Confirm (use qsub, not the login node):

```bash
python scripts/prior_diagnostics.py --cache outputs/prior_cache/kch_e1_32d.npz --atlas-dir data/atlases \
  --hyper "two_arms=1 effect_scale=0.2:1" --processes 200 --out /tmp/prior_diag.csv
```

The trial side needs the shares of n_pos, n_zero and n_neg per deficit, written by the current `evaluate_representation` after any rescoring.

### 2. Dependence on the covariate basis in the linear architecture

Training standardises with the pool and rotates; scoring re-standardises with the slice (coh:241-243 and 346-347 against est:100-102). TabICL is exempt (tab:127-137). A failure shows as lost correlation inside the prior when the estimator's transformations are applied. Confirm with script S, lines `slice_standardised` and `rotated` against `as_trained`.

### 3. Network far from the Bayes limit

Default batch 4 and 20,000 iterations (`reduced_config`, trn:109) show the net about 80,000 processes, each requiring it to locate a random readout over 16 networks. If a two-model ExtraTrees beats the net inside the prior, the net is undertrained and the 100,000-iteration runs are the right test. If the net wins inside the prior and still loses on the trial, the problem is the prior, as in explanation 1. Confirm with script S, line `extra_trees`.

## Script S

Run from the repository root on Myriad through qsub, less than ten minutes of CPU.

```python
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from neurocausalpfn.train.train_pfn import _build_prior
from neurocausalpfn.pfn.estimator import PFNEstimator

ck = "outputs/pfn_reduced_kch_e1_32d-theory-linear-2arms-scale/pfn.pt"
est = PFNEstimator(ck, device="cpu", standardize=False)
prior, d_x, _ = _build_prior(est.cfg, seed_offset=10_000)
R = np.linalg.qr(np.random.default_rng(0).normal(size=(d_x, d_x)))[0]

def et_two_model(Xc, T, Y, Xq):
    p = [ExtraTreesRegressor(200, min_samples_leaf=3, n_jobs=4, random_state=0)
         .fit(Xc[T == w], Y[T == w]).predict(Xq) for w in (1, 0)]
    return p[0] - p[1]

res = {}
for _ in range(40):
    b = prior.sample_batch(1, n_context=1024)
    Xc, T, Y, Xq = (b[k][0] for k in ("Xc", "Tc", "Yc", "Xq"))
    truth = b["mu1"][0] - b["mu0"][0]
    est.standardize = False
    ybin = (np.random.default_rng(1).uniform(size=len(Y)) < Y).astype(float)
    outs = {"as_trained": np.subtract(*est(Xc, T, Y, Xq)),
            "rotated": np.subtract(*est(Xc @ R, T, Y, Xq @ R)),
            "binary_Y": np.subtract(*est(Xc, T, ybin, Xq)),
            "extra_trees": et_two_model(Xc, T, Y, Xq)}
    est.standardize = True
    outs["slice_standardised"] = np.subtract(*est(Xc, T, Y, Xq))
    for k, v in outs.items():
        res.setdefault(k, []).append((np.corrcoef(v, truth)[0, 1],
                                      np.sqrt(np.mean((v - truth) ** 2)),
                                      (v * truth).sum() / max((v * v).sum(), 1e-12)))
for k, v in res.items():
    c, r, s = np.nanmean(v, axis=0)
    print(f"{k:20s} corr {c:.3f}  rmse {r:.3f}  rescale {s:.2f}")
```

Reading: a `rescale` above 1 on `as_trained` measures the Bayes shrinkage of explanation 1.
