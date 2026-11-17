# Job G: volume and centroid rule families, and leaving a family out (code only)

Branch `jobG-rule-families`, cut from `scale-budget-ckpt` (`8fa4b74`). Written 16 November 2026, revised 17 November
after the audit of the branch (`audit_branch_jobG.md`; its item numbers are in brackets below).
Nothing was trained or scored. No pool, latent, image or participant file was opened; tests use random arrays.
The experiment (which priors, which readings) is not fixed here.

## What changed

First round (16 Nov):

1. New `src/neurocausalpfn/prior/feature_rules.py`: families `volume` and `centroid`.
   The scalar is the lesion volume, or the centroid's coordinate on the first principal axis of the enrolled
   lesions' centroids. Threshold shape: top third +1, bottom third -1, middle 0. Graded shape: linear in the rank
   from -1 to +1.
2. `prior/cohort.py`: the two names are families (alone, or in the mixture). `family_weights` takes two numbers
   as before, or four. A weight of 0 leaves a family out.
3. `prior/neuro_prior.py`: fields in `HyperPrior` for the options below.
4. `scripts/audit_20261028/g4c_launch.sh`: `FAMILY_W` accepts two or four numbers, and the rule options.
   With two numbers the variables and the gate command are the same as before. `launch_myriad.qsub.sh` passes the
   new variables.
5. `scripts/prior_gate.py`: `--absent family ...` fails when a drawn process belongs to a family that must be
   absent. With four weights the launch passes every weight-0 family to it, and expects the rule options.
6. `train/train_pfn.py`: the family table of the staged curriculum knows the new names; comments.
7. One existing assertion changed: `tests/test_prior_families.py` compared `FAMILIES` with the old three names.

Second round (17 Nov):

8. **A lesion without a finite volume or centroid is an error** [21e]. `rule_effect` and `principal_coordinate`
   refuse values that are not finite, and a process refuses to enrol such a lesion, with a count. Before, NaN sorted
   last and took the top effect. I chose failing over leaving the lesion out: a pool with such a lesion is a fault
   to see. A lesion that is not enrolled is not read. The families of the final prior never reach this code.
9. **`RULE_GRADED` and `RULE_ENROL` were not added to the list `g4c_launch.sh` clears** [14a]. I did not do this
   one as asked, because it would disable them: that list holds the variables the launch itself sets for the
   trainer (`ITERS`, `BATCH`, ...). The rule options are inputs the launch reads, like `FAMILY_W` or `GENERIC_W`;
   cleared at the start, they could never be set. What a left-over value does is now tested and visible: with two
   weights the launch stops; with four it is used, and it shows in the run name, in a line the launch prints
   (`rule families: graded=... enrol_by=... enrol=... axis=... gate_require=...`) and in what the gate expects.
   `RULE_ENROL` with lo above hi is now refused by the launch, not later by the generator [14b].
10. **The tests that need the code before the change fail when they cannot run** [2]. In a clone without commit
    `8fa4b74` (or without git) they fail with a message naming the commit; they no longer skip.
11. **The gate's default requirements** [16]: see "The gate with `PRIOR_REPAIR=1`" below. The gate's defaults are
    untouched; a new launch variable, `GATE_REQUIRE`, names the ranges for a launch that draws the new families.
12. **Names** [11] and **allocation** [9] in this note: corrected below.
13. **Two options for decisions that are César's**: who a process enrols (`RULE_ENROL_BY`) and which centroids
    give the axis (`RULE_AXIS`). Defaults are what the branch did (tested draw for draw against its earlier tip).

## The hard requirement: the final prior is unchanged

`tests/test_rule_families.py::test_the_final_prior_is_unchanged_draw_for_draw` takes the code of `8fa4b74` from
git history, draws batches from it and from this code for the final launch's variables, and compares every array
exactly. It passes for seeds 0, 1 and 2, with the weights as `0 1` and as `0 1 0 0`. A second test does the same
for weights 0.5 : 0.5, for a generic share of 0, and for the family `giles` alone. All equal, on the code of the
second round too. The audit made its own 12 comparisons on the first round's tip, with the generators' internal
state, and found them equal; it has not yet seen the second round.

## The family weights, by the names in the code

`FAMILY_W` (the launch) becomes `FAMILY_WEIGHTS` and `family_weights`; the order is `cohort.MIXTURE_FAMILIES`.

| Position | Name in the code | What it is |
|---|---|---|
| 1 | `theory` | `prior/intersynth_theory.py`: a continuous effect built from two coverage scores of the lesion (D and S in that file). Ranges from `PRIOR_HYPER`. The launch prints it as "theory". |
| 2 | `giles` | `prior/neuro_prior.py`, the virtual-trial family: a lesion responds to the arm that matches the subnetwork it covers above a threshold; all or none. Ranges from `GILES_HYPER`. The launch prints it as "virtual-trial". |
| 3 | `volume` | `prior/feature_rules.py`, the lesion's volume. Ranges from `GILES_HYPER`. |
| 4 | `centroid` | `prior/feature_rules.py`, the centroid's coordinate on a principal axis. Ranges from `GILES_HYPER`. |

The first version of this note called positions 1 and 2 the "graded coverage rule" and the "threshold coverage
rule". Those are my words for `theory` and `giles`, not names in the code; to confirm or drop.
The latent-coordinate share is not a family weight: it is `GENERIC_W` (`generic_weight` in `PRIOR_HYPER`), applied
to every process after the family is drawn. So "0 1 1 0" with `GENERIC_W=0.5` is half latent-coordinate processes;
a prior with one family left out still holds that share unless `GENERIC_W=0`. The weights are renormalised to sum
to 1, so leaving a family out raises the shares of the others.

## How to launch

```bash
FAMILY_W="0 1 1 1" [RULE_GRADED=0.5] [RULE_ENROL=0.2:1 | RULE_ENROL_BY=network] [RULE_AXIS=pool] \
    [GATE_REQUIRE="share_zero=lo:hi share_unit=lo:hi"] OBJECTIVE=new bash scripts/audit_20261028/g4c_launch.sh
```

| Variable | Unset means | Meaning |
|---|---|---|
| `FAMILY_W` | `0.5 0.5` | two numbers (`theory giles`) or four (`theory giles volume centroid`) |
| `RULE_GRADED` | 0.5 | share of volume and centroid processes with the graded shape |
| `RULE_ENROL_BY` | `share` | who such a process enrols: `share` or `network` (below) |
| `RULE_ENROL` | the whole pool | with `share`: range lo:hi of the share of the pool a process enrols |
| `RULE_AXIS` | `enrolled` | centroids that give the principal axis: `enrolled` or `pool` (below) |
| `GATE_REQUIRE` | nothing added | ranges of the gate's metrics for this launch (below) |

All need four numbers in `FAMILY_W`; each is named in the run when set. `final1m_launch.sh` still fixes
`FAMILY_W="0 1"`; a launch for job G that keeps its md5 checks is not written.

## Three decisions that are César's

Each is an option with the branch's behaviour as default. I did not choose; this is what each does.

**a. Who a process enrols, and so where the thirds are cut** (`RULE_ENROL_BY`).

- `share`, the default. With `RULE_ENROL` unset a process enrols the whole pool. Every process of a family then
  ranks the same lesions, so its effects depend only on its shape (2) and its sign (2): **four functions per
  family**, cut at the pool's thirds. What varies between trials is efficacy, recovery, allocation, label noise
  and which lesions are drawn. A network can learn those four functions of the pool as such.
- `share` with `RULE_ENROL=lo:hi`: a process enrols a uniformly random part of the pool, of a share drawn in that
  range, and ranks within it. This is what `RULE_ENROL` did in the first round. It moves the cuts little: a random
  part of a pool has nearly the pool's thirds. In rank terms the cut of a part of n lesions has a standard error
  of sqrt(2/9 / n) around the pool's cut: 0.75 percentile points for 4,000 lesions, 1.5 for 1,000. So it is
  close to the four functions above, with some noise at the cuts.
- `network`, new. A process enrols the lesions that a virtual-trial process of the same launch would enrol:
  those its randomly drawn networks reach above its threshold (the slice of the family `giles`, built by the same
  code). Ranks, thirds and, by default, the axis are those of that slice. The slice is selected by anatomy, so its
  volumes and centroids are not the pool's and the cuts differ from trial to trial (tested on random arrays: 12
  processes, more than 3 distinct cuts and sizes). A slice below the smallest allowed is drawn again. How much the
  cuts move on the real pool is not known: nothing was run on it. `RULE_ENROL` does not apply and is refused.
  With `CONTEXT_POOL=all` no family restricts its rows and this one enrols the whole pool too.
- Not written: fixing the rule on the rows actually drawn into a trial (context and queries) instead of the slice
  they are drawn from. The trial's rows are a random draw from the enrolled lesions, so this differs from the
  above by sampling noise only.

**b. Which centroids give the principal axis** (`RULE_AXIS`).

- `enrolled`, the default: the first principal axis of the enrolled lesions' centroids. With the whole pool
  enrolled that is the pool's axis, which is what the audit saw [5]. With `network` enrolment it is the slice's own
  axis, which can point another way than the pool's (tested: a slab of a pool long in x has its own axis in y).
- `pool`: the axis of the whole pool's centroids, read at the enrolled lesions; ranks and thirds are still cut
  among the enrolled lesions. One anatomical direction for every trial.
- Which one matches analysis 2 of `siena_request_four_analyses.md`: the text says "its coordinate on the first
  principal axis of the centroids" and "on the same lesions and folds", and does not say whether the centroids are
  those of the whole scored cohort or of one trial. The scoring code is not in this repository. If scoring takes
  the axis of the whole cohort and cuts thirds per trial, the counterpart here is `RULE_ENROL_BY=network
  RULE_AXIS=pool`; if it takes both per trial, `RULE_ENROL_BY=network` alone; if both over the whole cohort, the
  defaults. In no case is the training pool the scored cohort, so the axis and the cuts are never the same numbers.
- The coordinate is not scaled: centroids are used in their own units, so the longest direction dominates.

**c. The direction of the axis.**

A principal axis has no direction of its own. `principal_coordinate` fixes it by a rule: the component of the axis
that is largest in size is made positive. That is deterministic for given centroids and does not depend on the
solver (tested: same call, rows reordered, points mirrored). It is arbitrary anatomically, and two sets of lesions
whose axes differ can get opposite directions. For the prior this does not matter: which arm the high end responds
to is a fair coin per process (`sign`), so flipping the convention only renames processes (tested). It would matter
if the sign coin were removed (choice 3 below): then the convention would decide which end responds to arm A.

## The gate with `PRIOR_REPAIR=1`

Without `PRIOR_REPAIR` the launch opens every range of the gate, as before, and the new families pass.
With `PRIOR_REPAIR=1` the launch passes no range, so the gate applies its defaults, which are the class shares of
an all-or-none trial: `share_zero` (queries with no effect) 0.35 to 0.65, `share_unit` (effect of size 0.95 or
more) 0.30 to 0.65, `share_small` at most 0.10. The new families do not have those shares: at full efficacy a
threshold rule gives one third with no effect and two thirds of size 1; a graded rule gives no exact zero and about
5 % of size 0.95 or more. On random arrays a prior of the two families alone fails `share_zero`
(tested: `test_the_gates_default_requirements_do_not_fit_these_families`); the audit measured 0.25 for
"0 1 1 1" without the generic share. So such a launch stops at the gate.

What to pass: `GATE_REQUIRE="share_zero=lo:hi share_unit=lo:hi"`, the two class shares, with the ranges the prior
that is wanted should have. The launch adds them after its own, so each replaces that metric's range and the other
metrics keep theirs (with `PRIOR_REPAIR=1`: the gate's defaults). `share_small` passes its default on random
arrays; whether it does on the real pool is not known. **Which ranges is César's decision**; opening both
(`0:1`) removes the check for those two. `GATE_REQUIRE` is refused unless the volume or centroid family is drawn,
so the launches of the earlier families keep the gate they had. It is not carried by `launch_myriad.qsub.sh`
(it holds spaces).

## Conventions followed from the virtual-trial family (`giles`), and where they differ

- Efficacy and recovery: with e the effect, arm 1 suits a lesion with probability 0.5 + e/2, arm 0 with 0.5 - e/2,
  and the expected outcome is 1 - (1 - TE r)(1 - RE). At TE = 1, RE = 0 this is the definition exactly. TE and RE
  come from the same ranges (`GILES_HYPER`).
- **A difference, the lesions with no effect** [7]: in the virtual trial a lesion that responds to neither arm has
  expected outcome RE under both. Here a lesion of the middle third has 1 - (1 - TE/2)(1 - RE) under both, 0.5 at
  TE = 1 and RE = 0. That follows the definition of analysis 2 (0.5 + effect/2), not the virtual trial.
- Outcome draws: Bernoulli in the context, both expected outcomes for the queries, same label noise.
- **Allocation is not the same in one mechanism** [9]. The names, weights and strength range of the three
  mechanisms are the virtual trial's, and `severity` (a ramp on volume with a random direction) and `agnostic` are
  the same. `axis` differs. In the virtual trial the ramp runs along the image axis (x, y or z) that best separates
  the two halves of the process's first network, towards the half that responds to arm 1: allocation always leans
  to the arm that works. Here the ramp runs along the principal coordinate of the centroids and its direction is
  a coin of its own, independent of the rule's sign. So in a centroid process it leans with the effect half the
  time and against it half the time (tested); in a volume process it runs along a quantity the rule does not
  read. Confounding in these families is therefore not distributed as in the final prior's `axis` case.
- No separate effect scale: the virtual-trial family has none.
- Context and queries are drawn from the enrolled lesions as from a slice.

## Tests

- `tests/test_rule_families.py`, 28 tests. First round (18): the unchanged-prior comparison; each rule's effects
  against its definition; thirds and ranks with ties; efficacy, recovery, outcome draws and allocation; weights of
  0 exclude a family; the launch variables reach the generator; the gate passes with the new keys and fails on a
  wrong one. Second round (10): values that are not finite are refused, in the functions, in a process and in a
  pool, and do not stop the families of the final prior; the new options unset draw what the first round drew,
  draw for draw (code of `7871305` from git history); enrolment by network; the two axes; the direction of the
  axis; the axis ramp's own direction; the options through the launch environment and the gate's keys; the gate's
  defaults failing these families and passing with the two shares named; a clone without the base commit fails.
- `tests/test_rule_families_launch.py`, 29 tests. `g4c_launch.sh` is run for real in a temporary checkout, with a
  stand-in `python` (the gate) and a stand-in PFN trainer that record what they receive.
  - With two family weights (the final launch's call, the default, `0.25 0.75`, a leave-networks-out call) the
    gate's command line, every launch variable and the printed run name are the same as with `g4c_launch.sh` of
    `8fa4b74`, which the test takes from git history.
  - With four weights: the weights and the rule options reach the gate and the trainer; the families with weight
    0 are passed to `--absent`; 21 bad inputs stop the launch before the gate; a rule option left in the shell
    stops or shows; `GATE_REQUIRE` reaches the gate with and without `PRIOR_REPAIR=1`.
  - `launch_myriad.qsub.sh` passes `FAMILY_W=0:1:1:1` and the rule options on.
- Full suite on this branch (Windows, Python 3.12, CPU): 463 passed, 5 failed, 2 skipped. The 5 failures are the
  5 that fail at `8fa4b74` (4 platform errors in `test_bundle_tools.py` and `test_disco_prior_match.py`, and
  `test_hyper_prior_contains_the_paper_scenarios`); the audit reran the suite there and found the same 5.

## With the job D branch

Both branches edit `g4c_launch.sh`. A trial merge of the two revised tips in a scratch clone (thrown away,
nothing merged in the repository) had no conflict. On the merged tree the test files of both branches and
`test_prior_families.py` passed: 97 tests, among them job D's comparison of the merged `g4c_launch.sh` with
`8fa4b74` for launches with two weights and no `RUN_TRAINER`.

## Not run anywhere

- A real training loop with the new families. The audit ran 4 steps on random arrays; nothing more.
- The real gate through the launch: the launch ran under Git Bash with stand-ins for `python` and the trainer.
  The gate's own code ran on random arrays only.
- The real pool and the real atlas. So nothing is known about the shares, the slices by network, the rejection
  rate or the cut points there.
- The scorer, and siena.
- `RUN_TRAINER=tabpfn` (job D) with four family weights.

## Choices to confirm

1. **Ties.** Lesions with equal values get the same effect; a tied group goes where its mean rank falls, and to 0
   when it sits exactly on a cut. The scoring-side code is not in this repository, so I could not match it.
2. **Thirds when n is not a multiple of 3:** the two ends have the same size, about n/3, and the middle takes the
   rest, so the middle can be the smallest group (n = 8: 3, 2, 3; n = 7: 2, 3, 2).
3. **A coin per process** decides which arm the high end responds to. The definition fixes it to arm A.
4. **TE and RE jitter** as in the final prior (0.9 to 1, 0 to 0.05), so effects are not exactly -1, 0, +1.
5. **Allocation is confounded**, not randomised as in the scoring definition, and its `axis` ramp is not tied to
   the effect as in the virtual trial (above).
6. **Who is enrolled** (decision a) and **which axis** (decision b).
7. **Graded and threshold shapes share one family weight** (`RULE_GRADED` splits them).
8. **The generic share replaces the outcome rule of volume and centroid trials too**, as for the other families;
   and a family left out does not remove the generic share.
9. **Weights are renormalised**, and **rejected trials redraw the family**, as before: the realised shares can
   differ from the weights, and the line "effective composition" the launch prints does not include rejection.
10. **The lesions with no effect sit at 0.5**, not at the recovery rate (above).
11. **Which ranges of `share_zero` and `share_unit` a prior with these families must meet** (`GATE_REQUIRE`).
12. **A lesion that cannot be ranked stops the run** rather than being left out.
13. **Commit messages are "trial"**, as in the rest of the history; this note is the description.

Not changed, from the same audit: the comparison with the code before the change covers the final prior's
settings and four others, not `CONTEXT_POOL=all`, `STD=context`, `PRIOR_REPAIR=1`, leave-networks-out, the staged
curriculum or several pools [3]; three diagnostic scripts read attributes only `giles` processes have [8].
