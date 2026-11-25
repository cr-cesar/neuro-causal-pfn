# Job D: TabPFN continued on the final prior (code only)

Branch `jobD-tabpfn-final-prior`, cut from `scale-budget-ckpt` (`8fa4b74`). Written 16 November 2026, revised 17 November after the audit of the branch; L2-SP and the drawn context length added 25 November.
Nothing was trained or scored. No pool, latent, image or participant file was opened; tests use random arrays.
The experiment (what is run, how it is read) is not fixed here.

## What the two existing pieces did (at `8fa4b74`)

`scripts/audit_20261028/tabpfn_ft_perfold.sh`:

| What | Value | Line |
|---|---|---|
| Pool | one per fold; `CACHE` must contain `{fold}` (a pool of UCLH and KCH without fold k's test patients, per the header) | 3-4, 20 |
| Models | one checkpoint per fold, folds 0 to 9 | 24 |
| Family weights | `--family mixture --family-weights 0.5 0.5`, not a variable | 36 |
| Threshold-rule ranges | `random_partitions=1 threshold=0.03:0.08 p_te=0.5:1` (so recovery and label noise at the class defaults, 0 to 0.5 and 0 to 0.05) | 21 |
| Other prior options | `two_arms=1 effect_scale=0.2:1 d_source=network s_source=genetics_1 s_b_source=genetics_2 effect_headroom=1 generic_weight=0.5` (no `binary_outcome`, no `generic_binary`) | 22 |
| Steps, context, estimators, seed | 2,000, 1,500, 1, 0 | 38-39 |
| Learning rate, trials per draw, queries | not passed: the script's 1e-5, 4, 256 | - |
| Checks | writes the commit and a count of changed files; no refusal, no md5, no prior gate | 31-35 |

The final prior differs in four places: family weights 0 : 1, `p_te=0.9:1 p_re=0:0.05 label_noise=0:0`,
`binary_outcome=1 generic_binary=1`, and the pool (the KCH pool, md5 `341e229d88f4`).

`scripts/finetune_tabpfn_prior.py`:

- Builds its own prior configuration, not the training launch's (`prior_from_args`, lines 46-70): `context_replace`,
  `standardize="pool"` and `augment` are written in the code (54-56); family weights are two numbers (57-61).
- `--n-query` defaults to 256 (line 244). The prior's smallest slice is queries + 8 (`prior/cohort.py:236`), so the
  script's prior only accepts slices of 264 lesions or more; the final model's accepts 72.
- Starts from the package checkpoint unless `--model-path` is given (138-140). AdamW through tabpfn's helper,
  learning rate 1e-5, weight decay 0.01, cosine schedule with 10 % warm-up, gradient clip 1 (147-150, 247-249).
- One update per simulated trial: a draw of the prior gives `--processes-per-draw` trials (default 4, line 161),
  each becomes one table, and the loader's batch size is 1 (169). There is no batch size beyond that.
- Outcomes: context outcomes that are already 0 or 1 are kept, others are drawn; query outcomes are drawn from the
  expected outcome (82-83). Context and query rows are then pooled and tabpfn's own splitter takes 20 % as queries.
- Writes `tabpfn_ft.pth` every 500 steps and at the end, under the same name, and `history.json` (208-221).
  No commit, no md5, no package version.
- Reads the pool cache and the atlas only. No fold, no scoring.

## What changed

First round (16 Nov):

1. `scripts/finetune_tabpfn_prior.py`: new option `--prior-from-env`. The prior comes from
   `launch_cfg.cfg_from_env()` and `train_pfn._build_prior`, the two functions the PFN training uses, so the same
   environment and seed give the same simulated trials (tested). In that mode the script also checks the pool's md5
   (`--pool-md5`) and writes `provenance.json` beside the checkpoint: commit, dirty flag, launch variables, library
   versions, pool path, size and md5, the atlas digest the launch computed, the tabpfn version, the starting
   checkpoint, the options of the run, the prior's configuration, and `"finished": true` once the last step is saved.
2. `scripts/audit_20261028/g4c_launch.sh`: new variable `RUN_TRAINER` (`pfn` by default, or `tabpfn`). With `tabpfn`
   the launch builds the same variables, runs the same gate, and then starts the script above with
   `--prior-from-env` and the options in `TABPFN_FT_ARGS` instead of the PFN training.
3. New `scripts/audit_20261028/tabpfn_final_launch.sh`: `final1m_launch.sh` with the other trainer. Same refusal of
   a tree with local changes, same md5 checks of the pool and atlas with the same defaults, same call of
   `g4c_launch.sh` (`OBJECTIVE=new FAMILY_W="0 1"`, the pool by path) plus `RUN_TRAINER=tabpfn`. One model per seed.
4. `tabpfn_ft_perfold.sh` is untouched.

Second round (17 Nov), after the audit of the branch (`audit_branch_jobD.md`, its item numbers in brackets):

5. **Prior options are refused, not ignored** [11]. With `--prior-from-env`, `--cache`, `--family`,
   `--family-weights`, `--prior-hyper`, `--giles-hyper`, `--n-query`, `--context-pool` and `--atlas-dir` stop the
   script with a message naming them, also when written with their default value (the command line is read, not the
   values).
6. **`RUN_TRAINER=tabpfn` has no default pool** [8]. `g4c_launch.sh` then needs `CACHE` (one pool, by path),
   `POOL_MD5` and `ATLAS_MD5`, compares them with the pool and with `data/atlases` using `final1m_launch.sh`'s
   commands, before the gate, and hands both digests to the script. The script itself now requires `--pool-md5`
   (which it checks against the file) and `--atlas-md5` (which it records; it does not recompute it).
7. **A finished model is compared, not trusted** [24]. The script compares the saved `provenance.json` with what is
   asked: commit, pool md5, atlas digest, tabpfn package, model version asked for, md5 of the starting file, seed,
   the prior's configuration, and every training option except `save_every`. Equal: the model is left as it is.
   Different: the run stops, says what differs (`finetune.steps: 2000 -> 300`), and touches nothing. The wrapper no
   longer skips by itself; every seed goes through the gate and the script.
8. **The wrapper exits with an error when a seed has no finished model** [25]. One after another, it stops at the
   first seed that fails and does not start the rest; with `PARALLEL=1` all run and any failure is an error. The
   count of "error lines" is gone (it matched words such as "nan" in paths). The log of a seed is appended to.
9. **The starting weights are named** [16, 17]. `--model-version` (default `v3.5`; `TABPFN_VERSION` in the
   wrapper, which also exports it to the package as `TABPFN_MODEL_VERSION`). Without `--model-path` the script
   asks the package's resolver for the file of that version, requires the package to be set to that version,
   requires the file to be on disk, hashes it, and gives it to TabPFN by path. If any of that fails the run stops
   and asks for `--model-path`; it no longer goes on with an unnamed checkpoint. With `--model-path` the file is
   hashed (`--model-md5` / `MODEL_MD5` checks it) and `provenance.json` says its version was not checked, unless it
   is byte-identical to the package's file for the version.
10. **torch is seeded per seed** [15]: `torch.manual_seed(seed)` before the model is built, in the
    `--prior-from-env` mode. The record says `torch_seed`.
11. **`FT_BATCH` is now `TRIALS_PER_DRAW`** [13]; `FT_BATCH` set stops the launch with a message. The help of
    `--processes-per-draw` says it is trials per draw of the prior, one update per trial, not trials per update.
12. **`history.json` without `--prior-from-env` has the keys it had** [10]: the options added for the other mode
    are left out of that record (tested against the option list of `8fa4b74`).
13. Two smaller ones from the same audit: checkpoints saved before the last step are named
    `tabpfn_ft.partial.pth` in the `--prior-from-env` mode, so `tabpfn_ft.pth` there is always a finished model
    [19]; `LR=0` is refused and the output folder is created after the checks, not before [27, 4].

Third round (25 Nov), two settings asked for by the registered request for job D
(`request_job_D_tabpfn_final_prior.md`, section "Two settings need code that the branch does not have").
Both are off unless set, and both belong to the `--prior-from-env` mode:

14. **L2-SP** (`--l2sp-alpha`, `L2SP_ALPHA` in the wrapper). Each update's loss gains
    (alpha / 2) * sum ||w - w0||^2 over the trainable weights; w0 is a copy of those weights made once, before the
    first update, never given to the optimizer. Unset or 0: no term and no copy; the updates are those of `6a7c6c3`
    (tested, loss for loss). With the term, every update records in `history.json` the loss (the cross-entropy, as
    before, without the term), the term (`l2sp_term`), alpha * ||w - w0|| (`l2sp_grad_norm`), the norm of the loss's
    own gradient before clipping, computed as the whole gradient minus alpha (w - w0) (`loss_grad_norm`), and
    ||w - w0|| / ||w0|| (`rel_dist`); each log line prints them for its last update.
15. **A drawn context length** (`--n-context-range LO:HI`, `N_CONTEXT_RANGE` in the wrapper). Each draw of the
    prior takes one length, uniform on the integers LO to HI inclusive, from a generator of its own seeded by the
    seed. Every length is printed with the log lines and recorded in `history.json` (`context_draws`: the length,
    the update count at which it was drawn and the number of updates that used it, 0 when the draw gave no usable
    table; and `n_context` in each update). Unset: the one length of `N_CONTEXT`.
    **Set together with `N_CONTEXT` (or `--n-context`) the launch refuses to start.** That is a proposal, the
    request left it to the commit: the other choice was to let the range win, which hides a variable that was set.
16. Both enter `provenance.json` (`finetune.l2sp_alpha`, `finetune.n_context_range`), the record inside the
    checkpoint, and the comparison with a finished model: another alpha or another range stops the rerun.
17. What an unset run writes compared with `6a7c6c3`: the same updates, and the same files plus those two keys
    recorded as off (`l2sp_alpha: 0.0`, `n_context_range: null`) in `history.json`'s `config`, in
    `provenance.json` and in the checkpoint's record. Nothing else differs (tested). The mode without
    `--prior-from-env` refuses both options and writes what it wrote.
18. Not checked by the code: the range against the launch's `CONTEXT_MAX` (3,000). A range above it is accepted,
    as a fixed `N_CONTEXT` above it already was. A range that starts at 1 is accepted too.
19. With mixed precision (a card), an update the scaler skips for a gradient that is not finite records guard
    quantities that are not finite either, while its loss stays finite. The loss is the quantity to read for
    "any loss nan or inf".

What stays as it was without the new option: the training loop, the prior of the other mode, the files written
and their names, `--seed` defaulting to 0, no torch seed, the package's default checkpoint.
Two visible differences remain in that mode: the message when `--cache` is missing, and the help text.

## How to launch

```bash
VENV=<environment with tabpfn> POOL=<the E13a KCH pool> ATLAS=<atlas folder> O=<output folder> \
    bash scripts/audit_20261028/tabpfn_final_launch.sh
```

| Variable | Default | Meaning |
|---|---|---|
| `SEEDS` | `0 1 2` | one model per seed, written to `<O>/s<seed>/` |
| `STEPS` | 2000 | updates; each reads one simulated trial |
| `LR` | 1e-5 | learning rate |
| `TRIALS_PER_DRAW` | 4 | simulated trials one draw of the prior holds (`--processes-per-draw`); not a batch |
| `N_CONTEXT` | 1500 | context rows drawn per trial, one length for the run |
| `N_CONTEXT_RANGE` | unset | `lo:hi`: the context length of each draw of the prior, uniform on the integers; not with `N_CONTEXT` |
| `L2SP_ALPHA` | unset (no term) | weight of the L2-SP term towards the starting weights |
| `N_EST` | 1 | tabpfn estimators |
| `TABPFN_VERSION` | `v3.5` | model version the run starts from |
| `MODEL_PATH`, `MODEL_MD5` | unset | another starting checkpoint by path, and its expected md5 |
| `POOL_MD5`, `ATLAS_MD5` | those of `final1m_launch.sh` | expected digests |
| `PARALLEL` | 0 | 1 runs the seeds at once |

Each seed leaves `tabpfn_ft.pth`, `history.json`, `provenance.json` and `<O>/logs/train_s<seed>.log`.
The exit status is 0 only when every seed has a finished model.
The shared launch can be called directly, with the same checks:

```bash
RUN_TRAINER=tabpfn CACHE=<pool> POOL_MD5=<12 characters> ATLAS_MD5=<12 characters> OUT=<folder> \
    [TABPFN_FT_ARGS="--steps 2000 ..."] bash scripts/audit_20261028/g4c_launch.sh
```

Scoring is not part of this branch. The existing panel reads one checkpoint for all folds when
`TABPFN_MODEL_PATH` has no `{fold}` in it (`pfn/panel.py:165-177`); I did not run it.

## Tests

`tests/test_tabpfn_final_prior.py`, 38 tests (the 29 of 17 Nov, 14 of them one refused option each, and 9 of
25 Nov), all passing here (Windows, Python 3.12, CPU, torch 2.3.1, tabpfn 6.3.1):

- the old options and defaults are unchanged; the old mode writes no new file;
- every prior option is refused with `--prior-from-env`, also when given its default value;
- `--prior-from-env` hands the launch's configuration to the PFN's builder, and on a synthetic pool the two priors
  give identical batches for the same environment and seed (64 queries, smallest slice 72, binary outcomes);
- the provenance record, the unfinished / finished marker and the two checkpoint names;
- **`finetune()` itself, both modes, with stand-ins for every tabpfn call** (a two-layer torch model, a splitter, a
  saver). These run the script's own lines: six updates on the launch's prior, the files written, the same seed
  giving the same losses and another seed other losses (this fails when the torch seed is removed; checked), a
  finished model left alone or stopping the run for ten kinds of difference, the starting weights refused when the
  package is set to another version, when the file is missing, when the resolver cannot be called. They say nothing
  about tabpfn's own API;
- `history.json` of the old mode has exactly the option names of the script at `8fa4b74`;
- the launch scripts, run for real in a temporary checkout with a stand-in `python` and a stand-in PFN trainer:
  every variable that reaches the prior is the same under `final1m_launch.sh` and under the new launch; three seeds
  give three folders; eleven bad inputs are refused before the output folder exists; a failing seed makes the launch
  exit with an error, one after another and at once; the shared launch refuses `RUN_TRAINER=tabpfn` without a pool,
  without digests, with wrong digests, with two pools;
- **`g4c_launch.sh` of `8fa4b74` against this branch's**, in one checkout, for four calls without `RUN_TRAINER`
  (default, the final model's call, a call without the generic share, a call with the TabPFN path's variables left
  in the shell): the gate's command line, every exported variable at the gate and at the trainer, and the printed
  lines are equal.

The 9 of 25 Nov, on a toy model and on the stand-ins:

- the term and its gradient against the formula, tensor by tensor; the recorded loss-gradient norm against the
  gradient of the loss taken alone; the starting weights left untouched by training, a frozen weight left out;
- alpha unset or 0 and no range: the script of `6a7c6c3`, taken from git history, and this one give the same
  losses and rates, exactly, and the files differ only by the two keys;
- the guard inside `finetune()`: the four quantities at every update and log line, zero distance before the first
  update, a stronger guard keeping the weights closer;
- the same quantities with a stand-in scaler that multiplies the loss by 1,024: measured after unscaling and before
  clipping (the order of the calls is checked), equal to the plain path, unchanged by a clip of 1e-6;
- lengths in the range, differing between draws, the same with the same seed, others with another seed, each
  printed and recorded, each table having that many context rows; a range of one length equal to the fixed length
  update for update; a draw that gives no table recorded with 0 updates;
- the range with a fixed length refused (also when the fixed length is the default value), bad ranges and weights
  refused, both options refused without `--prior-from-env`;
- a finished model compared on both settings, in both directions;
- the wrapper passing both, printing them, and refusing both lengths set, bad ranges and bad weights before the
  output folder exists.

The tests that need commit `8fa4b74` or `6a7c6c3` fail, with a message, in a clone that does not have it.

Full suite on this branch: 444 passed, 5 failed, 2 skipped (17 Nov: 435 passed, 5 failed, 2 skipped). The 5 failures are the 5 listed at the end of this note; they fail at `8fa4b74` too (the audit reran the suite there: 406 passed, the same 5 failed).

## With the job G branch

Both branches edit `g4c_launch.sh`. A trial merge of the two revised tips in a scratch clone (thrown away,
nothing merged in the repository) had no conflict, and the test files of both branches passed on the merged
tree (97 tests with `test_prior_families.py`). `RUN_TRAINER=tabpfn` with four family weights was not run.

## Not run anywhere

- The real training loop: tabpfn 9.2 or later and a GPU. Not one real update was made.
- **`resolve_model_path` on tabpfn 9.2.0 (siena's release) was not checked.** This machine has tabpfn 6.3.1, where
  it is `resolve_model_path(model_path, which, version)` and returns four lists, the first holding the path. The
  script calls it as `resolve_model_path(None, "classifier", "v3.5")[0][0]`. I had no way to read 9.2.0 here. If the
  call differs there, the run stops with "the package's resolver failed" and asks for `MODEL_PATH`; it does not
  start. `settings.tabpfn.model_version` is read the way `pfn/panel.py` reads it.
- Whether TabPFN 9.2.0 loads its own default file, given by path, exactly as it loads it by default.
- The real gate, the real pool, the real atlas, and any launch on siena. The scripts ran under Git Bash with
  stand-ins.
- The scorer on a checkpoint written here.
- Memory: whether one card holds three seeds at once (`PARALLEL=1`).
- L2-SP on a card: the real mixed-precision path (tested with a stand-in scaler only), and its cost. The guard
  holds a second copy of the trainable weights and, at each update, the differences from it; it reads every
  weight once more per update for the recorded quantities. Peak memory and seconds per update with the guard are
  for the dry run.
- A context of 3,000 rows plus the queries in the real loop: memory and time.
- Bit-for-bit repeatability on a GPU: the generators are seeded, the GPU kernels are not forced to be deterministic.

## Choices to confirm

1. **A switch in `g4c_launch.sh` instead of a copy of its prior strings.** It is now 27 added lines in the shared
   launch, all inactive without `RUN_TRAINER=tabpfn` (tested against `8fa4b74`).
2. **64 queries per trial, not 256.** They come from the launch, because they set the smallest slice the prior
   accepts (72, the final model's). Each table then has `N_CONTEXT` + 64 rows before tabpfn's 80 / 20 split.
3. **A fixed context of 1,500 rows** (the old wrapper's value). The final model saw 256 to 3,000. The script's own
   default is 2,000.
4. **No batch.** One update reads one trial; `TRIALS_PER_DRAW` only groups the draws. The final model's 4 trials
   per update would need gradient accumulation, which I did not add.
5. **Seeds run one after another**, and the launch stops at the first failure, unless `PARALLEL=1`.
6. **What seed s seeds**: the prior (as the final model's seed s), the outcome draws, tabpfn's `random_state`, the
   splits, the loader's shuffles (seed + step, so seed 0 at step k and seed 1 at step k - 1 share a shuffle seed;
   the trials differ) and now torch's global generator.
7. **`OBJECTIVE=new` is passed for parity with `final1m_launch.sh`.** It sets only the PFN's loss and output range,
   which this trainer does not use. The gate still checks the PFN keys (objective, 20,000 iterations, batch 4,
   bins, attention) although none of them trains anything here; `RUN_ITERS` is not set by this launch.
8. **No scoring hook.** Nothing here opens a fold of any cohort.
9. **Schedule, weight decay, clip, one estimator**: left as they were (cosine with 10 % warm-up, 0.01, 1.0, and the
   old wrapper's 1 estimator against the script's 2).
10. **The start is the package's v3.5 file unless `MODEL_PATH` is given.** With `MODEL_PATH` the version of the file
    is taken on trust; `MODEL_MD5` is the check.
11. **A finished model made at another commit stops the launch**, even if the commits differ in a file that does
    not matter. The strict reading; the other was to compare the configuration only.
12. **An unfinished folder is trained again from the start**, over what is there.
13. **Commit messages are "trial"**, as in the rest of the history; this note is the description.
14. **Both a range and a fixed length set: the launch refuses to start** (item 15 above). A proposal.
15. **The recorded `loss` is the cross-entropy without the L2-SP term**, so losses compare across values of alpha;
    the term is beside it.
16. **The lengths come from a generator of their own**, seeded by the seed, so a run with a range of one length is
    the run with that fixed length, update for update.
17. **The second-instance check of 25 Nov** (a reader given the diff and the request's section) asked for: a test
    of the mixed-precision path (added); the files of an unset run to be stated (item 17 above, tested); draws
    without updates to be told apart (the `updates` count, tested); a weight too large to be a number refused by
    the wrapper and a zero weight said to be no term (done); list equality on tensors replaced (done). Stated and
    not changed: items 18 and 19 above, and the cost of the guard (under "Not run anywhere").

Not changed, from the same audit: `ATLAS` and the default paths are siena's [26]; `POOL_MD5` and `ATLAS_MD5` can be
overridden by the caller, as in `final1m_launch.sh` [28]; no library list beyond python, torch, numpy and tabpfn in
the record [18].

## Seen on the way, not changed

- `tests/test_neuro_prior.py::test_hyper_prior_contains_the_paper_scenarios` fails at `8fa4b74` (scenario
  `noisy_te025` is outside the default efficacy range 0.3 to 1).
- Four more tests fail at `8fa4b74` on this Windows machine, in `test_bundle_tools.py` (3) and
  `test_disco_prior_match.py` (1). The errors are about the platform: a file mode of 666 where 600 is expected, and
  shell scripts that Windows cannot start by name. I did not check them on Linux.
