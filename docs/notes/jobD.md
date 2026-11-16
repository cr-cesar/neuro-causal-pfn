# Job D: TabPFN continued on the final prior (code only)

Branch `jobD-tabpfn-final-prior`, cut from `scale-budget-ckpt` (`8fa4b74`). Written 16 November 2026.
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

1. `scripts/finetune_tabpfn_prior.py`: new option `--prior-from-env`. The prior comes from
   `launch_cfg.cfg_from_env()` and `train_pfn._build_prior`, the two functions the PFN training uses, so the same
   environment and seed give the same simulated trials (tested). In that mode the script also checks the pool's md5
   (`--pool-md5`) and writes `provenance.json` beside the checkpoint: commit, dirty flag, launch variables, library
   versions, pool path, size and md5, the atlas digest the launch computed, the tabpfn version, the starting
   checkpoint (name, size, md5 when the file can be located), the options of the run, the prior's configuration, and
   `"finished": true` once the last step is saved. Without the option the script behaves as before; `--seed` still
   defaults to 0.
2. `scripts/audit_20261028/g4c_launch.sh`: new variable `RUN_TRAINER` (`pfn` by default, or `tabpfn`). With `tabpfn`
   the launch builds the same variables, runs the same gate, and then starts the script above with
   `--prior-from-env` and the options in `TABPFN_FT_ARGS` instead of the PFN training. Unset, nothing changes
   (tested: same run name, PFN trainer started).
3. New `scripts/audit_20261028/tabpfn_final_launch.sh`: `final1m_launch.sh` with the other trainer. Same refusal of
   a tree with local changes, same md5 checks of the pool and atlas with the same defaults, same call of
   `g4c_launch.sh` (`OBJECTIVE=new FAMILY_W="0 1"`, the pool by path) plus `RUN_TRAINER=tabpfn`. One model per seed.
4. `tabpfn_ft_perfold.sh` is untouched.

## How to launch

```bash
VENV=<environment with tabpfn> POOL=<the E13a KCH pool> ATLAS=<atlas folder> O=<output folder> \
    bash scripts/audit_20261028/tabpfn_final_launch.sh
```

| Variable | Default | Meaning |
|---|---|---|
| `SEEDS` | `0 1 2` | one model per seed, written to `<O>/s<seed>/` |
| `STEPS` | 2000 | updates |
| `LR` | 1e-5 | learning rate |
| `FT_BATCH` | 4 | simulated trials per draw of the prior (`--processes-per-draw`) |
| `N_CONTEXT` | 1500 | context rows drawn per trial |
| `N_EST` | 1 | tabpfn estimators |
| `MODEL_PATH` | package default | starting checkpoint, by path |
| `POOL_MD5`, `ATLAS_MD5` | those of `final1m_launch.sh` | expected digests |
| `PARALLEL` | 0 | 1 runs the seeds at once |

Each seed leaves `tabpfn_ft.pth`, `history.json`, `provenance.json` and `<O>/logs/train_s<seed>.log`.
A seed whose `provenance.json` says finished is skipped on a rerun; an interrupted one is trained again from the start.

Scoring is not part of this branch. The existing panel reads one checkpoint for all folds when
`TABPFN_MODEL_PATH` has no `{fold}` in it (`pfn/panel.py:165-177`); I did not run it.

## Tests

`tests/test_tabpfn_final_prior.py`, 6 tests, all passing here (Windows, Python 3.12, CPU):

- the old options and defaults are unchanged, and the old mode writes no new file;
- `--prior-from-env` hands the launch's configuration to the PFN's builder;
- on a synthetic pool, the script's prior and the PFN trainer's prior give identical batches for the same
  environment and seed, with 64 queries, smallest slice 72, binary outcomes, and no graded-rule trial;
- the provenance record and the unfinished / finished marker;
- the three launch scripts, run for real in a temporary checkout with a stand-in `python` and a stand-in PFN
  trainer: every variable that reaches the prior is the same under `final1m_launch.sh` and under the new launch;
  three seeds give three folders; wrong pool md5, wrong atlas md5, a bad option and a tree with local changes are
  refused; without `RUN_TRAINER` the PFN trainer starts as before.

## Not tested

- The training loop. It needs tabpfn 9.2 or later and a GPU; this machine has tabpfn 6.3.1 on CPU. I did not change
  the loop, only what is built before it and saved after it.
- Locating the package's default checkpoint on the server's tabpfn release. If the lookup fails, the run goes on
  and `provenance.json` says so (`resolve_error`, `md5: null`). Passing `MODEL_PATH` avoids the question.
- The launch scripts on siena. They ran here under Git Bash with stand-ins, never with the real gate, pool or atlas.
- Memory: whether one card holds three seeds at once.
- Loading the new checkpoint in the scorer.

## Choices to confirm

1. **A switch in `g4c_launch.sh` instead of a copy of its prior strings.** It is 13 added lines in the shared
   launch. The other way was to repeat the family weights and the two option strings in the new wrapper, which can
   drift from the final launch.
2. **64 queries per trial, not 256.** They come from the launch, because they set the smallest slice the prior
   accepts (72, the final model's). Each table then has `N_CONTEXT` + 64 rows before tabpfn's 80 / 20 split.
3. **A fixed context of 1,500 rows** (the old wrapper's value). The final model saw 256 to 3,000. The script's own
   default is 2,000.
4. **"Batch" is trials per draw.** The update is still one trial per step; I did not add gradient accumulation.
5. **Seeds run one after another** unless `PARALLEL=1`.
6. **Seed s seeds everything**: the prior (as the final model's seed s), tabpfn's `random_state`, the outcome draws
   and the splits.
7. **`OBJECTIVE=new` is passed for parity with `final1m_launch.sh`.** It sets only the PFN's loss and output range,
   which this trainer does not use; nothing in the prior depends on it. The gate still checks those PFN keys.
8. **No scoring hook.** Nothing here opens a fold of any cohort.
9. **Schedule, weight decay, clip, one estimator**: left as they were (cosine with 10 % warm-up, 0.01, 1.0, and the
   old wrapper's 1 estimator against the script's 2).
10. **Commit messages are "trial"**, as in the rest of the history; this note is the description.

## Seen on the way, not changed

- `tests/test_neuro_prior.py::test_hyper_prior_contains_the_paper_scenarios` fails at `8fa4b74` (scenario
  `noisy_te025` is outside the default efficacy range 0.3 to 1).
- Four more tests fail at `8fa4b74` on this Windows machine, in `test_bundle_tools.py` (3) and
  `test_disco_prior_match.py` (1). The errors are about the platform: a file mode of 666 where 600 is expected, and
  shell scripts that Windows cannot start by name. I did not check them on Linux.
