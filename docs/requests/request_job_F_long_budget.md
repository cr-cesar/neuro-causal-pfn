# Request: job F. The final recipe trained up to ten times longer, with a checkpoint on the way every 500,000 simulated trials (from Claude, auditor; 16 Nov 2026)

**For the agent on Myriad.** The training needs Gladys's go-ahead in your session, with the GPU hours she grants. Scoring on the validation set only (1,291 images, ten pseudo-folds); nothing on the UCLH test folds. Aggregates only. **Edit no script: if a step needs a change of code, stop and report. Code changes are Cesar's.**

**Whose choice is what.** Every value below is the auditor's proposal. Cesar may edit it before the commit on the machine; after it, it is fixed.

## The question

The final model saw one million simulated trials. Does the same recipe, on the same prior, get better with more of them?

What is known. From 500,000 to one million trials the PEHE fell by 0.007 in the dissertation (Table 4.10) and by 0.005 in the audit's scaling runs (reported). In those runs the values at 750,000 and at 1.5 million trials are within 0.0011 of the value at one million (reported). Beyond 1.5 million trials nothing is measured. The gap of the final model to TabPFN-3.5 as S-learner on validation is +0.0095 (0.0019 to 0.0171).

## The training

- **Code:** branch `scale-budget-ckpt`, commit 8fa4b74. Against the final model's commit (69250c3) the training path changes in one commit only, a8ed60d: two launch options and the saving of intermediate checkpoints (read by the auditor).
- **Unchanged from the final model:** the network (single axis, 6 layers), the prior and its shares (0 : 0.5 : 0.5), the pool (KCH lesions as E1-bin latents, code id E13a, on BCBtoolkit disconnectomes; the copy already on Myriad, md5 beginning `5c90906a`, whose seven members were checked against the original), the objective, the head, the optimiser and its constant learning rate, mixed precision.
- **Changed:** the length only. Options of `g4c_launch.sh`: `RUN_ITERS=2500000`, `RUN_RAMP_ITERS=125000`, `RUN_CKPT_EVERY=125000`. One update is four simulated trials, so the run reaches ten million trials and saves a checkpoint every 500,000 trials (`pfn_it<updates>.pt`; the last one, at 2,500,000 updates, is `pfn.pt`). The context ramp of 125,000 updates is the final model's, so the checkpoint at 250,000 updates follows the final model's schedule.
- **Seeds:** 0, 1 and 2, one job per seed, each on an A100 and with the longest wall time the machine allows (48 hours on Myriad; the header of the project's batch script says 36 and is not used here). The three seeds use the same card type. They may run at the same time, and need not wait for job K.
- **Launch:** `g4c_launch.sh` once per seed, with the variables `final1m_launch.sh` passes and the three options above, after the same link and the same checks as in the request for job K (clean tree at the commit, the pool, atlas md5 `fd1b531e7cf9`). If the machine needs a new wrapper script to queue a seed, that is new code: stop and report.
- **If a job reaches its wall time**, the checkpoints already written stay. That is expected, not a failure: the curve is then shorter. A seed has reached a budget only if its checkpoint at that budget loads; a file cut during saving is dropped and reported.
- **Stop rule.** If the loss of a seed becomes non-finite, stop that seed and report it. Do not restart it with other settings.

## Step 0, before the training (no GPU)

1. The commit on the machine and a clean tree. The configuration diff against the final model: among the settings of the network, the prior and the training it must touch `iters`, `ramp_iters` and `ckpt_every` only (the output folder, the job name and the paths differ).
2. After the first checkpoint of the first seed: the seconds per update, the updates forecast within the wall time, and the versions of Python, PyTorch and CUDA.

## The scoring (on siena, where the validation latents are)

As the final model in run 1 (the scoring of 12 November): the 32 virtual trials, efficacy 1 and recovery 0, every training patient of the trial in the context.

- Every complete checkpoint of every seed is scored. At each budget, the PEHE per trial is the mean of the three seeds. Predictions are not averaged across seeds.
- Paired differences over the 16 deficits, on the mean of the two splits, t with 15 degrees of freedom.
- The checkpoints travel from Myriad to siena. Gladys or Cesar orders that transfer.

## Readings, fixed now

**The reference** is the checkpoint at 250,000 updates (one million trials) of the same three runs, so that only the length differs. **The budget read** is the largest of 500,000, 1,000,000, 1,500,000, 2,000,000 and 2,500,000 updates that all three seeds reached. It depends on the speed of the cards, not on any score. Let f = PEHE at the budget read minus PEHE at the reference. The page states the reading with its budget, in trials. The yardstick is half of the gap to TabPFN-3.5, 0.00475, as in the requests for rung g and job K.

- **A. More trials close half of the gap or more:** f is −0.00475 or lower, and its paired 95 % interval excludes zero.
- **B. They do not:** the lower limit of the interval is above −0.00475. If the whole interval is above zero, add "and the longer training is worse".
- **Otherwise** the run does not decide at this precision, and the page says so.
- **If the three seeds do not all reach 500,000 updates,** no reading is given.

**Reported without a reading:**

- the curve: PEHE at every checkpoint, with the seed SD;
- the reference minus the stored final model of run 1 (0.2788), with its interval. If that interval excludes zero, the page says the run does not reproduce the final model. The machine, the PyTorch build and the pool file differ from the final model's, so a small difference is possible;
- the budget read minus TabPFN-3.5 as S-learner (0.2693), with its interval.

**What the run cannot show.** The learning rate is constant and the context ramp is fixed at the first 125,000 updates. If the curve is flat, more trials did not help this recipe; a schedule that lowers the rate might still gain, and this run does not test it. One prior and one network: it says nothing about a different prior. The pool is a copy with another md5, checked member by member against the original.

## What to send back

Step 0; the list of checkpoints per seed with updates and md5; one csv (budget, seed, PEHE by split and mean) and one of the paired differences with their intervals; the per-trial rows; one page ending with the reading; the last lines of the training logs and the commits. Each statement labelled verified, reported or conjecture.

## Registration

Registered by the auditor before any of these results exists; the sha256 is given to Gladys with the file. Keep the file as received, commit it unchanged on the branch, and report the commit and the sha256 you compute. If you received it as pasted text, do not commit the paste: ask for the file.
