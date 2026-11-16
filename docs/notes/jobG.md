# Job G: volume and centroid rule families, and leaving a family out (code only)

Branch `jobG-rule-families`, cut from `scale-budget-ckpt` (`8fa4b74`). Written 16 November 2026.
Nothing was trained or scored. No pool, latent, image or participant file was opened; tests use random arrays.
The experiment (which priors, which readings) is not fixed here.

## What changed

1. New `src/neurocausalpfn/prior/feature_rules.py`: families `volume` and `centroid`.
   The scalar is the lesion volume, or the centroid's coordinate on the first principal axis of the enrolled
   lesions' centroids. Threshold shape: top third +1, bottom third -1, middle 0. Graded shape: linear in the rank
   from -1 to +1.
2. `prior/cohort.py`: the two names are families (alone, or in the mixture). `family_weights` takes two numbers
   (theory, giles) as before, or four (theory, giles, volume, centroid). A weight of 0 leaves a family out.
3. `prior/neuro_prior.py`: two fields in `HyperPrior`, `rule_graded` (share of graded processes, default 0.5) and
   `rule_enrol` (range of the share of the pool a process enrols, default the whole pool).
4. `scripts/audit_20261028/g4c_launch.sh`: `FAMILY_W` accepts two or four numbers; `RULE_GRADED` and `RULE_ENROL`
   set the two options. With two numbers the variables and the gate command are the same as before.
   `launch_myriad.qsub.sh` passes the new variables.
5. `scripts/prior_gate.py`: `--absent family ...` fails when a drawn process belongs to a family that must be
   absent. With four weights the launch passes every weight-0 family to it, and expects the two rule options.
6. `train/train_pfn.py`: the family table of the staged curriculum knows the new names; comments.
7. One existing assertion changed: `tests/test_prior_families.py` compared `FAMILIES` with the old three names.

## The hard requirement: the final prior is unchanged

`tests/test_rule_families.py::test_the_final_prior_is_unchanged_draw_for_draw` takes the code of `8fa4b74` from
git history, draws batches from it and from this code for the final launch's variables, and compares every array
exactly. It passes for seeds 0, 1 and 2, with the weights as `0 1` and as `0 1 0 0`. A second test does the same
for weights 0.5 : 0.5, for a generic share of 0, and for the virtual-trial family alone. All equal.

## How to launch

```bash
FAMILY_W="0 1 1 1" [RULE_GRADED=0.5] [RULE_ENROL=0.2:1] OBJECTIVE=new bash scripts/audit_20261028/g4c_launch.sh
```

Order of the weights: graded coverage rule, threshold coverage rule, volume, centroid. The latent-coordinate share
is `GENERIC_W` as before (0 leaves it out). `final1m_launch.sh` still fixes `FAMILY_W="0 1"`; a launch for job G
that keeps its md5 checks is not written.

## Conventions followed, from the threshold coverage family

- Efficacy and recovery: with e the effect, arm 1 suits a lesion with probability 0.5 + e/2, arm 0 with 0.5 - e/2,
  and the expected outcome is 1 - (1 - TE r)(1 - RE). At TE = 1, RE = 0 this is the definition exactly. TE and RE
  come from the same ranges as the threshold coverage family (`GILES_HYPER`).
- Outcome draws: Bernoulli in the context, both expected outcomes for the queries, same label noise.
- Allocation: the same three mechanisms and strength range. The ramp runs on the centroid coordinate or on the
  volume, with a random direction.
- No separate effect scale: the threshold coverage family has none.
- Context and queries are drawn from the enrolled lesions as from a slice.

## Tests

`tests/test_rule_families.py`, 18 tests. Run here with the neighbouring prior tests: 61 passed, 1 skipped,
1 failed (`test_hyper_prior_contains_the_paper_scenarios`, which fails at `8fa4b74` too).

## Not tested, not done

- **The full test suite was not run on this branch**, only the six prior test files. Run it before review.
- **`g4c_launch.sh` was syntax-checked only** (`bash -n`). The new `FAMILY_W`, `RULE_*` and `--absent` lines were
  not executed. The job D branch has a test that runs the launch with stand-ins; this branch has none.
- The two branches both edit `g4c_launch.sh`, in different places. A merge was not tried.
- The gate ran on a synthetic pool only.

## Choices to confirm

1. **Ties.** Lesions with equal values get the same effect; a tied group goes where its mean rank falls, and to 0
   when it sits exactly on a cut. The scoring-side code is not in this repository, so I could not match it.
2. **Thirds when n is not a multiple of 3:** both ends get the same size, the middle takes the rest.
3. **A coin per process** decides which arm the high end responds to. The definition fixes it to arm A.
4. **TE and RE jitter** as in the final prior (0.9 to 1, 0 to 0.05), so effects are not exactly -1, 0, +1.
5. **Allocation is confounded** as in the final prior, not randomised as in the scoring definition.
6. **Enrolment defaults to the whole pool.** Then every volume-threshold process cuts the pool at the same two
   values. `RULE_ENROL` draws a random part per process; thirds and the axis are then those of that part.
7. **Graded and threshold shapes share one family weight** (`RULE_GRADED` splits them).
8. **The generic share replaces the outcome rule of volume and centroid trials too**, as for the other families.
9. **Rejected trials redraw the family**, as before; the realised shares can differ from the weights.
10. **Commit messages are "trial"**, as in the rest of the history; this note is the description.
