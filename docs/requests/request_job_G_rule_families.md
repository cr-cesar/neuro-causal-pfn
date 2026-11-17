# Request: job G. Priors with volume and centroid rule families, each family left out in turn (from Claude, auditor; 17 Nov 2026)

**For the agent on the machine where the trainings run, and for the agent on siena, who scores.** The trainings and the short tests of step 0 need Gladys's go-ahead in your session, with the GPU hours she grants. Scoring on the validation set only (1,291 images, ten pseudo-folds, clean context); nothing on the UCLH test folds. Aggregates only. **Edit no script of the repository: if a step needs a change of code, stop and report. Code changes are Cesar's.**

**Whose choice is what.** The code is the branch `jobG-rule-families` at commit 0386acf, cut from `scale-budget-ckpt` (8fa4b74). Cesar accepted it on the board on 17 November with three values: `RULE_ENROL_BY=trial`, `RULE_AXIS=pool`, `RULE_AXIS_SIGN=random` (read by the auditor in the board's database; passed on by Gladys). He asked that this request state the prior's rule definitions beside those of run 2; they are in the table below. Every other value here is the auditor's proposal. Cesar may edit it before the commit on the machine; after it, it is fixed.

## The question

Run 2 (13 November) scored the final model on four rules built on quantities its prior has no rule for: lesion volume and the position of the centroid, each as a threshold rule and as a graded rule. Its PEHE was 0.316 to 0.345, against 0.147 to 0.279 for the best estimator fitted to outcomes (verified). The final prior holds one family of rules, the virtual trial, and a share of latent-coordinate processes.

Does a prior with one more rule family do better on a family it never saw? Job G trains priors that hold the new families, together and one at a time.

**"Never saw" has a limit.** Half of the processes of every prior here, the final one included, are latent-coordinate processes, and some of those are three-group rules on a linear score of the latents (`generic_mix.py`, kind `step`). So the final model has seen threshold-type rules on latent scores. What it has not seen is a rule defined on volume or on the centroid.

## The priors

Family weights are `FAMILY_W`, in the order `theory giles volume centroid` (`giles` is the virtual-trial family). The latent-coordinate share stays at one half of the processes in all of them (`GENERIC_W` unset, 0.5). Weights are renormalised, so a family left out raises the shares of the others.

| Name | `FAMILY_W` | Holds | Trained here |
|---|---|---|---|
| h | `0 1` | virtual trial | no: this is the final model, its three seeds exist |
| G-all | `0 1 1 1` | virtual trial, volume, centroid | yes |
| G-volume | `0 1 1 0` | virtual trial, volume | yes |
| G-centroid | `0 1 0 1` | virtual trial, centroid | yes |
| G-no-trial | `0 0 1 1` | volume, centroid | yes |

Four priors are trained, three seeds each: twelve trainings. For the volume rules, G-volume has the family and G-centroid is the prior that never saw it; for the centroid rules, the reverse.

## The rules in the prior, beside those of run 2

| | In the priors of job G (training; read in the code at 0386acf) | In run 2 (scoring; from its page) |
|---|---|---|
| Quantities | lesion volume; the centroid's coordinate on a principal axis | the same two |
| Lesions | the training pool: KCH lesions as E1-bin latents | the validation lesions, on siena |
| The axis | first principal axis of the centroids of the whole pool (`RULE_AXIS=pool`) | first principal axis of "the cohort's centroids". Whole cohort or one replicate: siena states it in step 0 |
| Direction of the axis | a fair coin per centroid process (`RULE_AXIS_SIGN=random`); a volume process draws none | not in the page. Siena states it in step 0 |
| Where the thirds are cut | among the distinct lesions of each simulated trial, context and queries (`RULE_ENROL_BY=trial`) | not in the page: within the replicate or over the whole listing. Siena states it in step 0 |
| Shapes | threshold (top third +1, bottom third −1, middle 0) and graded (linear in rank, −1 to +1); half of the processes each (`RULE_GRADED` unset, 0.5) | the same two shapes, each as its own rule |
| Which arm the high end responds to | a fair coin per process | arm A: the page gives 0.5 + effect/2 under A |
| Expected outcome | 1 − (1 − TE·r)(1 − RE), with r = 0.5 + effect/2 for arm 1 and 0.5 − effect/2 for arm 0; TE from 0.9 to 1 and RE from 0 to 0.05, the final prior's ranges | 0.5 + effect/2 under A, 0.5 − effect/2 under B. The prior gives this at TE = 1, RE = 0 |
| Outcomes | Bernoulli draws in the context | Bernoulli draws |
| Allocation | confounded: the three mechanisms of the virtual trial with their default weights and strength; the ramp runs along the principal coordinate, with a direction coin of its own | randomised |
| Other processes in the prior | half are latent-coordinate processes | none |
| Ties, and n not a multiple of 3 | a tied group goes where its mean rank falls, and to 0 when it sits on a cut; without ties the two end thirds have equal size | as coded in `b13_rules.py`. Siena states it in step 0 |

So for a prior that holds a family, the rules of run 2 are close to rules of its prior, not equal to them. The differences in the table are part of every reading.

## The training

- **Unchanged from the final model:** the network (single-axis, 6 layers), the pool (md5 beginning `341e229d88f4`, or its copy checked by the seven member digests of the job K request), the objective, 250,000 updates of 4 simulated trials, the context schedule, the optimiser and its constant learning rate, the ranges of efficacy and recovery, the generic share.
- **Changed:** `FAMILY_W` as in the table above, and the three rule options.
- **Launch.** As in the request for job K: call `g4c_launch.sh` once per prior and seed with the variables `final1m_launch.sh` passes at 69250c3 (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VENV OBJECTIVE=new ATLAS_SRC SEED RUN_ITERS=250000 JOBNAME CACHE OUT`), with `FAMILY_W` and the three rule options. Make the atlas link and do that script's checks by hand: no local change in tracked files, the pool, the atlas md5 `fd1b531e7cf9`. If the machine needs a new wrapper script, that is new code: stop and report.
- **A clean shell.** The launch does not clear the variables it reads. Before each launch none of these may be set: `RULE_GRADED`, `RULE_ENROL`, `GATE_REQUIRE`, `PRIOR_REPAIR`, `GENERIC_W`, `ALLOC`, `STD`, `DISTINCT`, `S_SOURCE`, `EXCLUDE_NETWORKS`, `CATE_W`, `HEAD_RANGE`, and any `RUN_*` other than `RUN_ITERS`. With `PRIOR_REPAIR` unset the gate's ranges are open, as for the final model. With four weights the gate still fails if it draws a family of weight 0; that is wanted.
- **The tree.** A checkout at 0386acf with no local change; the launch prints the commit and the checkpoint records it. This request is committed on a branch of its own, as for jobs K and F, not in the training tree.
- **Names.** One new output folder and job name per prior and seed, for example `jobG_all_s0`. No folder of an earlier run is reused.
- **Seeds:** 0, 1 and 2.
- **Checkpoint:** the last one. None is chosen by a score on validation.
- **Stop rules.** The code does not stop on a non-finite loss (read at 0386acf): the agent reads each training log at least every six hours and, if the logged loss is nan or inf, stops that training and reports it. It is not restarted with other settings. If a training runs out of memory, stop; do not lower the context. The training cannot resume: ask for the longest wall time the machine allows, and if step 0 forecasts a training above 40 hours, stop.

## Step 0, before any training (send this back first)

On the training machine:
1. **The code.** The branch at 0386acf on the machine, by the way `scale-budget-ckpt` was brought for job F. `tests/test_rule_families.py` and `tests/test_rule_families_launch.py` run there: the counts of passed and failed. They need commits 8fa4b74 and 7871305 in the history.
2. **The pool.** For the members `volumes` and `centroids`: the number of values that are not finite. Counts only. One line of Python that loads those two arrays is allowed for this; it is not a change to the repository. With `RULE_AXIS=pool` a bad centroid stops every run at once; a bad volume stops a run only when a trial draws that lesion, possibly hours in.
3. **Four short runs, with Gladys's go-ahead,** one per prior, on the card type to be used: a small `RUN_ITERS`, its own folder, its checkpoint discarded. The context ramps up over the first half of a run, so the second half is at 3,000 rows. From the head of each log: the commit, the line "rule families", the line "effective composition", what the gate prints, and the configuration the launch built. From the run: seconds per update at 3,000 rows, peak memory, a finite loss, and the hours forecast per training as that figure times 250,000 (an upper bound). If trainings are to share a card, as the final launch did, one test is run that way.
4. **The configurations side by side** with the final model's. Apart from output paths, job names and seeds, only the family weights and the three rule options may differ.

The gate draws 40 processes, so its shares are rough; they are description.

On siena:
5. **The rules of run 2 as coded** (`scripts/audit_20261112/b13_rules.py`): the four cells of the table above that say "siena states", and the commit.
6. **The same draws.** The scoring below must use the replicates, allocations and outcome draws of run 2. The stored rows hold results, not draws. So: score the final model, seed 0, on one rule again and compare with its stored per-replicate rows. They must agree to four decimals. If not, stop and report.
7. **Two aggregates:** the rank correlation, in absolute value, between volume and the centroid coordinate, over the validation lesions with the axis of run 2, and over the training pool with the pool's axis. Two numbers. Siena may write a small script of its own for them, as for its tables.

**Go or stop.** The trainings start only if: the two test files pass; both counts of point 2 are zero; the gate passes for the four priors; the forecast is within the wall time and the hours Gladys grants. Otherwise stop and report; Cesar decides.

Checkpoints move from the training machine to siena. Gladys or Cesar orders the transfer.

## The scoring (siena)

**On the rules of run 2.** The five rules (four and the null rule), 32 replicates, ten pseudo-folds, encoder E1-bin seed 0, as run 2. A replicate's value is the mean over its pseudo-folds; for each prior it is then the mean of the PEHE of its three seeds. The final model h, rung e and the estimators fitted to outcomes are reused from the stored rows. Paired differences over the 32 replicates; every interval is the paired t interval at 95 %, with 31 degrees of freedom.

**On the virtual trials.** As rung f of run 1: the 32 trials, efficacy 1 and recovery 0, full contexts. Per trial, the mean of the three seeds. Paired differences over the 16 deficits, on the mean of the two splits; t intervals at 95 % with 15 degrees of freedom. The final model (0.2788), TabPFN-3.5 as S-learner (0.2693) and the predictor of no effect are reused from the stored rows of run 1.

## Readings, fixed now

Lower is better. Each of the four rules of run 2 gets one state, in this order. For a rule on quantity Q, "G-Q" is the prior that holds Q and "G-other" the prior that holds the other new family and never saw Q.

1. **Not learnt.** G-Q minus h is not negative with an interval excluding zero. Then training on Q did not teach the rule in this budget, and no state on transfer is given.
2. **Transfer by half or more.** Per replicate, u = G-other − (h + G-Q)/2. The upper limit of the interval of u is below zero: the prior that never saw Q reaches half or more of what training on Q gives.
3. **No transfer.** The interval of G-other − h includes zero or lies above it. If above, add "and it is worse".
4. **Some transfer.** G-other − h is negative with an interval excluding zero, and state 2 does not hold. "Less than half" if the lower limit of the interval of u is above zero; otherwise "share not decided".

A statement about a quantity is made only if its two rules, threshold and graded, have the same state. If they differ, the page gives the two states and makes no statement for the quantity. The four rules are read without a correction for multiple comparisons; the page says so.

**Cost of a wider prior.** d = G-all − h on the virtual trials. The yardstick is 0.00475, half of the gap of h to TabPFN-3.5.
- "No cost at this yardstick": the upper limit of the interval of d is below 0.00475.
- "A cost of half the gap or more": d is 0.00475 or more and the interval excludes zero.
- "A cost below the yardstick": the interval lies above zero and d is below 0.00475. This can hold together with the first.
- Otherwise not decided.

**Seeds.** The intervals above hold the noise of the replicates or the deficits, not the spread across training seeds. So for each prior and rule the page gives the PEHE of each seed and their SD (three values, n − 1). If that SD, for either prior of a difference, is larger than the absolute difference, the page adds "within the spread of seeds" beside the state. If a seed was stopped, no state is given for its prior.

**Reported without a reading:** G-no-trial on the virtual trials, beside h, TabPFN-3.5 as S-learner and the predictor of no effect, with its differences to h and to no effect; G-all on the four rules, minus h and minus G-Q; every prior on the null rule (the final model scored 0.3495 there); every prior minus the best estimator fitted to outcomes, per rule; the two aggregates of point 7.

## What the run cannot show

- The 32 replicates of a rule measure the sampling noise of one rule on one cohort. They are not 32 mechanisms. An interval that excludes zero says the difference is not that noise; it does not say it holds for other rules.
- Two new families, one at a time. If volume and centroid are related (point 7), a transfer between them may be that relation and nothing more general.
- The rules of run 2 differ from the prior's in the ways of the table: axis and thirds cut on other lesions, a fixed arm, randomised allocation.
- G-Q and G-other differ in one family; both differ from h by one family and by the share the virtual trial loses to it.
- G-no-trial keeps the latent-coordinate share, so its result on the virtual trials is not that of volume and centroid rules alone.
- One budget and one network. The code of the branch has not been run on the real pool before this job, and its last four commits had a light check only.

## What to send back

Step 0; per prior and seed, the head and the last lines of the training log and the md5 of the checkpoint; one csv per scoring (estimator, rule or trial, seed, PEHE, paired differences with their intervals) and the per-replicate and per-trial rows; one page ending with the states and the cost reading; the commits and the md5 of each file. Each statement labelled verified, reported or conjecture.

## Registration

Checked once by a second instance before registration (`audit_request_job_G.md`); its points are taken in above. Registered by the auditor before any training of these priors exists; the sha256 is given to Gladys with the file. Keep the file as received, commit it unchanged on a branch of its own, and report the commit and the sha256 you compute. If you received it as pasted text, do not commit the paste: ask for the file.
