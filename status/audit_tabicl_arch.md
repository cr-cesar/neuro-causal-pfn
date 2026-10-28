# Audit of the `tabicl` architecture of the Neuro-Causal PFN

Read-only. Aggregate numbers only: no patient identifiers, no image file names, no latent values. No code was changed and nothing was committed.

## Answer in one paragraph

The `tabicl` checkpoint does learn. Inside its own prior it reaches a correlation of 0.74 with the true CATE, the same as a two-model ExtraTrees fitted on the same 1,024 rows (0.74), with the oracle rescaling factor at 0.97. It is not stuck near a constant and it is not hurt by the standardisation or the rotation of the scoring path. What fails is transfer to the trial, and the clearest architecture-specific difference found is how it reads the outcome column: when the context outcomes are replaced by Bernoulli draws of themselves, the correlation falls from 0.74 to 0.01, whereas the linear checkpoint of the same seed falls from 0.80 to 0.38. That stress test is harsher than the trial, so this is a strong lead, not yet a proven cause. A second contribution is a weaker and slower-fitting model than the linear one inside the prior (0.74 against 0.80 to 0.82), consistent with an optimisation recipe tuned for the linear model. Nothing in the forward path is a bug.

## Scope and version

- Tree: `~/Scratch/neuro-causal-pfn`, HEAD `bae0c76` on `results-snapshot`. `src/neurocausalpfn/pfn/tabicl_model.py` is unchanged since commit `1aea0aa` (173 lines), so what is audited is the version the checkpoint was trained with.
- Checkpoints: `outputs/pfn_reduced_kch_e1_32d_x-theory-2arms-scale-project-head-s0/pfn.pt` (tabicl, 7,676,385 parameters) and `..._x-theory-linear-2arms-scale-project-head-s0/pfn.pt` (linear, 4,831,489 parameters). Their printed configurations are identical except `arch`: same prior hyper-parameters (two arms, effect scale 0.2 to 1, D from the whole-network overlap, S_A and S_B from the two genetics halves, effect headroom on), same cache, same 20,000 iterations, same seed.
- "project-head" is not a code option. As confirmed by the project owner, it is only the `VARIANT_TAG` used to name the runs of the headroom prior; there is nothing to look for in `src` or `scripts`.
- The probe ran as a CPU job (id 9999529, 4 cores, 5 minutes). Everything else was a seconds-long read.

## 1. Did it train? Loss comparison

Loss from `history.json`: iteration 0 is the single value, the other rows are the mean of the 100 iterations ending at the shown iteration. No NaN or infinite value in either run. Learning rate constant at 2e-4 for all 20,000 iterations in both (no warmup, no decay). Context length grows from 256 to 1,024 over the first half.

| point | iteration | context | tabicl loss | linear loss |
|---|---|---|---|---|
| start | 0 | 256 | 5.679 | 6.032 |
| 25 % | 4,999 | 640 | 4.256 | 4.134 |
| 50 % | 9,999 | 1,024 | 4.058 | 4.019 |
| end | 19,999 | 1,024 | 3.983 | 3.970 |
| median of the last 2,000 | | 1,024 | 3.980 | 3.968 |
| minimum single batch | | | 3.648 | 3.701 |

The 25 % row differs slightly from the 4.31 in the brief only because of the averaging window. The two curves are indistinguishable at the end.

Reference floor: with a Gaussian-smoothed point target of sigma 0.02 on 256 bins of width 1/256, the entropy of the target is about ln(0.02 x 4.13 x 256) = 3.05 nats. Both runs end 0.9 nats above it, so neither is close to the Bayes limit, and equal loss says nothing about which one transfers.

Logs: tabicl 41 min 20 s wall (18:13:23 to 18:54:43), linear 23 min 34 s on a possibly different node, so tabicl is about 1.75 times slower per run. Neither log shows a warning besides the usual nested-tensor notice of the linear log, and neither shows an error.

## 2. Does the checkpoint predict inside its own prior?

Held-out processes of the checkpoint's own prior, context 1,024, 40 processes; each cell is the mean over processes. `scripts/prior_consistency_probe.py` via `prior_consistency_probe_myriad.qsub.sh`.

| variant | tabicl s0: corr | rescale | slope | linear s0: corr | rescale | slope |
|---|---|---|---|---|---|---|
| as trained | 0.736 | 0.97 | 0.57 | 0.804 | 1.00 | 0.67 |
| slice-standardised | 0.736 | 0.97 | 0.57 | 0.80 (brief) | | |
| rotated | 0.727 | 0.95 | 0.58 | 0.80 (brief) | | |
| outcomes as Bernoulli(Y) | 0.011 | 0.17 | 0.004 | 0.375 | 0.26 | 0.58 |
| two-model ExtraTrees, continuous Y | 0.738 | 1.05 | 0.48 | 0.738 | 1.05 | 0.48 |

The ExtraTrees row is the same in both columns because the processes are the same.

Other linear seeds, same prior: as trained 0.820 and 0.811, Bernoulli outcomes 0.484 and 0.460, ExtraTrees 0.773 and 0.765.

Reading. The brief's rule says a value near 0.80 means the problem is transfer, a low value means training. The tabicl value is 0.74: not as high as the linear one, but equal to the ExtraTrees yardstick and with rescale 0.97, so the checkpoint is calibrated and informative inside its prior. The problem is mostly transfer. The slice standardisation changes nothing (identical to six digits) and the extra rotation costs 0.009.

## What the scoring looks like (E1 published, seed 0, aggregated over 16 deficits and 10 folds)

| run | PEHE | bal. acc. | slope | rescale | PEHE after rescale | tau_hat for tau = +1 / 0 / -1 | between / within |
|---|---|---|---|---|---|---|---|
| tabicl | 0.525 | 0.560 | 0.086 | 1.59 | 0.476 | 0.15 / 0.07 / 0.02 | 0.280 / 0.008 |
| linear, no rotation | 0.550 | 0.576 | 0.043 | 2.60 | 0.517 | 0.10 / 0.06 / 0.02 | 0.311 / 0.007 |
| linear, with rotation | 0.384 | 0.837 | 0.41 (brief) | not on disk | not on disk | not on disk | not on disk |

Share of test patients with tau = +1, 0, -1 in this slice: 0.28, 0.63, 0.09. For the tabicl run nearly all of the error is the group means (between), and the spread of tau_hat inside each group is tiny (within 0.008): it returns an almost constant prediction at scoring, positive on average, with the wrong sign for the tau = -1 group (+0.02 against -1). Rescaling does not help (0.525 to 0.476), so this is a loss of information, not a calibration problem. The linear run with rotation has no decomposition columns on disk (it was scored before they were added), so its 0.41 slope could not be re-derived here.

## 3. Forward path read against what training and scoring feed it

Lines refer to `src/neurocausalpfn/pfn/tabicl_model.py` unless stated.

(a) Standardisation. The model standardises the covariate columns with context statistics inside `_embed_cells` (lines 127-137), at training and at scoring. The prior standardises the pool once and rotates (cohort.py, `augment=True`, `standardize="pool"`, line 251 of the current file). Standardising again inside the model is an anisotropic rescale of rotated columns, applied identically on both sides. The estimator standardises first (estimator.py:154-156) and the model again; the second pass is idempotent. Measured: as-trained and slice-standardised agree to six digits. Ruled out. Per-column shape of the two sides is also similar: pool latents have median absolute skew 0.14 and excess kurtosis 0.03, fold-encoder latents 0.12 to 0.19 and 0.15; no near-constant column on either side; spectral participation ratio 26.6 for the prior pool and 27.2 to 29.0 for two fold encoders (out of 50). Marginal-shape mismatch is not what separates the two sides.

(b) Masks and query leakage. The query outcome cell is overwritten with a learned mask token (lines 139 and 143), the target-aware addition runs only on context rows (line 145), the outcome row encoding runs only on context rows (line 169), the inducing points of the column stage read context rows only (lines 88-91) and the in-context stage uses the context-only mask (line 170). Queries neither attend to each other nor see their own outcome. Ruled out.

(c) CLS representation and zero-initialised branches. The row representation is the concatenation of 4 CLS outputs (384 values). All blocks start as the identity, but after training the output projections of the six in-context blocks have mean absolute weight 0.0085 to 0.0168 (all away from zero), and the in-prior correlation is 0.74, so the in-context stage is active, not constant. The mean absolute weight is about half of a default linear initialisation of that width (about 0.026), and smallest in the last block, which fits a model that is still fitting. Ruled out as a cause of a constant output; consistent with under-training.

(d) Head. The same `CEPOHead` as the linear model: 256 bins on [0, 1], sigma 0.02, mean of the softmax over bin centres. Mean absolute weight of the head projection 0.040 against 0.031 for the linear model. Ruled out.

(e) Estimator. `PFNEstimator` recovers `d_x` from `col_embed` of shape (52, 96), so 50, equal to the configuration. The standardise flag resolves to true from `prior.augment`. Confirmed consistent.

One structural observation: the outcome is injected into every column cell of every context row (`y_col_enc`, line 145) and again in row space (`y_row_enc`, line 169), 53 additions per context row, with first-layer weights of magnitude 0.47 (the covariate projection has the same magnitude). The model therefore reads the outcome very strongly at several depths, which is the architectural reason to expect a different sensitivity to the outcome distribution than the linear model, which injects it once.

## 4. Learning rate and optimiser

Shared with the linear model: AdamW, learning rate 2e-4 constant (`lr_schedule: constant`), weight decay 0.01, gradient clip 1.0, batch size 4 contexts, 64 queries per context, context up to 1,024, fp16 autocast with a gradient scaler on CUDA. That is 4 x 1,024 context rows and 256 supervised queries per step for a 7.7M-parameter two-axis model. There is no warmup and no decay. The zero-initialised residual branches make the first steps move only the output projections, and a constant learning rate with such a small batch leaves the loss noisy (single-batch minimum 3.65 against a median of 3.98). From memory, not verified here, the TabICL pre-training recipe uses far larger batches of tables, a warmup and a decay; the scripts already expose `LR_SCHEDULE=cosine` (warmup 2 percent, decay to 0.1 of the peak) and `BATCH`. Plausible contributor, not tested for tabicl.

## 5. Tests

`tests/test_tabicl.py`, four tests:

- forward and five Adam steps on a toy batch: output shape and a finite loss;
- query independence: perturbing the covariates and treatment of one query leaves another unchanged, and changing a context outcome changes the prediction (with zero-initialisation turned off);
- `build_model` chooses the architecture and the v0 skeleton for old configurations;
- a three-iteration `run_pfn` smoke test with a finite loss.

Not covered: fitting a trivial prior to a low loss (nothing shows the model can learn anything); a query's own outcome being invisible (true by reading the code, untested); the effect of the outcome type (binary against continuous); invariance to the context standardisation and to a rotation; equivalence of the loaded checkpoint and the trained model; fp16 against fp32; any run longer than three iterations. `tests/test_disconnectome_fusion.py` mentions TabICL only through the encoder.

## Ranked causes

| rank | cause | status | evidence |
|---|---|---|---|
| 1 | The outcome channel: tabicl reads the outcome far more sharply than the linear model, and the theory-only prior never shows binary outcomes | plausible, strongest lead, not yet shown to cause the scoring collapse | Bernoulli(Y) outcomes: 0.74 to 0.01 (tabicl) against 0.80 to 0.38 (linear). Outcome injected at 53 places per context row. Counter-evidence: the stress is harsher than the trial, whose binary outcome is a noise-free function of susceptibility; and the linear model retrained on binary outcomes scores worse (PEHE 0.48 against 0.38) |
| 2 | Weaker, slower-fitting model under a recipe tuned for the linear one (constant 2e-4, batch 4, no warmup) | plausible, contributes | in-prior correlation 0.74 against 0.80 to 0.82; ICL output weights half of default scale, smallest in the last block; loss 0.9 nats above the floor in both |
| 3 | Transfer to the fold encoders, which the column-wise embedding may handle worse than the linear row embedding | plausible, untested for tabicl | linear without rotation collapses the same way (slope 0.04, PEHE 0.55), showing transfer is fragile; but marginal shape and spectrum of the two latent sides match |
| 4 | Standardisation or rotation interaction | ruled out | as trained, slice-standardised and rotated agree within 0.009 |
| 5 | Query leakage or visible own outcome | ruled out | lines 88-91, 139, 143, 145, 169, 170 |
| 6 | Constant output from zero-initialised blocks | ruled out | output weights away from zero, correlation 0.74 |
| 7 | `d_x` or standardise flag mismatch | ruled out | d_x 50 recovered, flag true, harmless |
| 8 | Head, support, NaN, precision | ruled out | same head as linear, no NaN, no warning |

## Top cause: proposed fix and cheapest experiment

Fix to try if the experiment below confirms it: make the training outcomes cover the trial's. Add an outcome-type option to the theory prior that, with some probability per process, replaces the continuous outcome by its noise-free binarisation (Y = 1 when the expected outcome plus noise exceeds 0.5), next to the existing Bernoulli option (`binary_outcome`, `intersynth_theory.py` lines 85 and 287-290), which is noisy and, for the linear model, did not help. A second, independent measure is `LR_SCHEDULE=cosine` with a larger `BATCH`.

Cheapest experiment, CPU, minutes, no code change in the repository: a standalone script for a compute node that scores both checkpoints on held-out processes with deterministic binary outcomes and adds the missing yardstick, ExtraTrees on the binary outcomes. Run from the repository root through qsub, never on a login node.

```python
import numpy as np, sys
from sklearn.ensemble import ExtraTreesRegressor
from neurocausalpfn.train.train_pfn import _build_prior
from neurocausalpfn.pfn.estimator import PFNEstimator

CKS = {"tabicl": "outputs/pfn_reduced_kch_e1_32d_x-theory-2arms-scale-project-head-s0/pfn.pt",
       "linear": "outputs/pfn_reduced_kch_e1_32d_x-theory-linear-2arms-scale-project-head-s0/pfn.pt"}
ests = {k: PFNEstimator(v, device="cpu", standardize=True) for k, v in CKS.items()}
prior, d_x, _ = _build_prior(ests["tabicl"].cfg, seed_offset=10_000)   # same prior for both

def et(Xc, T, Y, Xq):
    p = [ExtraTreesRegressor(200, min_samples_leaf=3, n_jobs=4, random_state=0)
         .fit(Xc[T == w], Y[T == w]).predict(Xq) for w in (1, 0)]
    return p[0] - p[1]

def corr(a, b):
    return np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan

res, rng = {}, np.random.default_rng(1)
for _ in range(40):
    b = prior.sample_batch(1, n_context=1024)
    Xc, T, Y, Xq = (b[k][0] for k in ("Xc", "Tc", "Yc", "Xq"))
    truth = b["mu1"][0] - b["mu0"][0]
    ctx = {"continuous": Y,
           "bernoulli": (rng.uniform(size=len(Y)) < Y).astype(float),
           "threshold": (Y > 0.5).astype(float)}
    for name, Yc in ctx.items():
        for k, e in ests.items():
            res.setdefault((name, k), []).append(corr(np.subtract(*e(Xc, T, Yc, Xq)), truth))
        res.setdefault((name, "extra_trees"), []).append(corr(et(Xc, T, Yc, Xq), truth))
for (name, k), v in sorted(res.items()):
    print(f"{name:11s} {k:12s} corr {np.nanmean(v):.3f}")
```

How to read it. If tabicl keeps a correlation of about 0.5 or more under the `threshold` outcomes, the outcome channel is not what breaks the scoring and cause 3 moves up. If it falls to about 0 while the linear model and ExtraTrees keep a clearly positive correlation, cause 1 is confirmed, and the fix above is justified.

If it is confirmed, one GPU run, to be launched only with approval. The default launcher passes no hyper-prior in the non-per-fold `phase2`, so the full string is given. The existing cache `kch_e1_32d_x` is reused:

```bash
ARCH=tabicl PRIOR_FAMILY=theory TAG=kch_e1_32d_x VARIANT_TAG=2arms-scale-bin05-s0 SEED=0 \
PRIOR_HYPER="two_arms=1 effect_scale=0.2:1 d_source=network s_source=genetics_1 s_b_source=genetics_2 effect_headroom=1 binary_outcome=0.5" \
bash scripts/myriad_phase1b.sh phase2
```

This yields the run `kch_e1_32d_x-theory-2arms-scale-bin05-s0`. It uses the existing Bernoulli option at half the processes; it tests whether exposure to binary contexts is enough. The deterministic-binarisation option above is a code change and would need its own run.

## Corrections to the earlier audit of the train and score path

- The earlier audit said every patient of a trial slice is hit and the absolute tau is 1 for single-hit patients. The shares measured here are 0.28 with tau = +1, 0.09 with tau = -1 and 0.63 with tau = 0, so most test patients have no effect. That weakens the earlier explanation 1 (training slices have most patients near zero effect), because the trial slice is also mostly zero effect. The argument about which patients carry the effect, and about the scale being fixed on the whole pool, stands, but the contrast with the trial is smaller than stated there.
- The earlier note that no result on disk carries the effect decomposition holds only for old runs; the newer runs (tabicl, no-rotation linear, binary-outcome linear) have it.

## What was not done

- No GPU job, no retraining, no change of code or configuration.
- The linear checkpoint was not re-probed; its values come from the probe summaries already in `status/audit_consistency/`.
- The two linear PEHE values are two scorings of the same linear checkpoint (seed 0), as confirmed by the project owner: 0.384 is the original scoring with the context limited to 1,024 rows (the one on the leaderboard I read); 0.379 is the scoring with the full context (`PFN_CONTEXT_CAP=0`, folder `outputs_perfold_iso/headroom_s0`), the one adopted since stage 0. The 0.41 slope and its decomposition columns come from that full-context scoring. The tabicl scoring read here used the default cap, so the like-for-like linear figure is 0.384.
- No test shows that `tabicl` can fit a trivial prior to a low loss. The project owner agrees it does not exist; if the cause above is confirmed, that test enters together with the fix.
- The cause of the scoring collapse is not proven; the ranking above is by evidence, and the CPU experiment is the step that separates causes 1 and 3.

## Addendum, 20 Oct 20:xx

CPU probe (job 1637, 40 processes of the checkpoint's own prior, context 1,024), mean correlation with the true CATE:

| context outcomes | tabicl | linear | ExtraTrees |
|---|---|---|---|
| continuous | 0.736 | 0.804 | 0.738 |
| Bernoulli(Y) | -0.013 | 0.453 | 0.438 |
| deterministic Y > 0.5 | 0.126 | 0.395 | 0.490 |

Cause 1 (the binary-outcome channel) is confirmed at probe level: with binary context outcomes the linear model and ExtraTrees keep a clearly positive correlation and tabicl collapses, also under the deterministic binarisation closest to the trial. The probe shows tabicl misreads binary outcomes; it does not show this is the only cause of the 0.525.

GPU run queued: `kch_e1_32d_x-theory-2arms-scale-bin05-s0` (binary_outcome=0.5, job 1638, scoring 1639). Probe log: `binY-probe.o1637` (aggregate figures only).

Note: the scheduler started failing after the GPU run was queued (the qmaster refused connections), so the state of jobs 1638 and 1639 was not checked.
