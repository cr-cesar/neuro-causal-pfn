# Request: job K. A TabICL-style network trained from scratch on the final prior, one million simulated trials (from Claude, auditor; 16 Nov 2026)

**For the agent on the machine where the job runs.** Step 0 needs no GPU and may be done at once. The training needs Gladys's go-ahead in your session, with the GPU hours she grants; the 20-minute test of step 0 needs her go-ahead too. Scoring on the validation set only (1,291 images, ten pseudo-folds); nothing on the UCLH test folds. Aggregates only. Configurations on a branch, not merged. **Edit no script: if a step needs a change of code, stop and report. Code changes are Cesar's.**

**Whose choice is what.** Every value below is the auditor's proposal. Cesar may edit it before the commit on the machine; after it, it is fixed.

## The question

The final model is a single-axis transformer of 4.8M parameters. On validation it is 0.0095 above TabPFN-3.5 as S-learner (interval 0.0019 to 0.0171). Does another network, given the same prior and budget, do better?

The dissertation has one value for another network: the TabICL-style one (column-then-row attention, 12 in-context layers, 14.8M parameters). Its PEHE was 0.385, against 0.304 for the single-axis model in the same setting, with a seed SD of 0.047 on the receptor split and 0.085 on the genetics split (Table 4.9). That setting was not the final one: 500,000 simulated trials, prior shares 0.25 : 0.25 : 0.5, pool and scoring on E1 latents. Job K repeats it in the final setting.

**What job K is not.** It does not use TabICLv2's released weights; that is the job "TabICLv2 continued pre-training" on siena. Job K trains the project's own TabICL-style network from scratch.

## The training

- **Unchanged from the final model (commit 69250c3):** the prior and its shares (0 : 0.5 : 0.5); the pool (KCH lesions as E1-bin latents, code id E13a, on BCBtoolkit disconnectomes); the objective; the output head; 250,000 updates of 4 simulated trials; the context schedule; the optimiser and its constant learning rate; mixed precision.
- **Changed:** the network. Two launch options of `g4c_launch.sh`: `RUN_ARCH=tabicl` and `RUN_N_LAYERS=12`. Without the second the network has 6 in-context layers, not the 12 of Table 4.9. Its size changes with it: 14.8M against 4.8M.
- **Seeds:** 0, 1 and 2, one job per seed.
- **Launch.** `final1m_launch.sh` always trains a 500,000-trial control, which job K does not need. So call `g4c_launch.sh` once per seed with the variables that script passes (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VENV OBJECTIVE=new FAMILY_W="0 1" ATLAS_SRC SEED RUN_ITERS=250000 JOBNAME CACHE OUT`) and the two options above, after making the link `data/atlases` as that script does (line 21). Its checks (lines 22 to 27) are then done by hand, before the first seed: no local change in tracked files, pool, and atlas md5 `fd1b531e7cf9` computed with its command. This request also requires the tree to be at 69250c3, which that script only prints. If the machine needs a new wrapper script to queue a seed, that is new code: stop and report.
- **Memory and time.** The context grows to 3,000 rows. If a seed runs out of memory, stop. Do not lower the context. The training saves its checkpoint only at the end and cannot resume, so a seed that passes the wall time of its job is lost: ask for the longest wall time the machine allows.
- **The pool on another machine.** The pool may be a copy without the keys `files`, `images_dir` and `fold_file`, which hold file names and paths and are not read in training. Its md5 then differs. The check is then on the members the training uses: `overlaps`, `centroids`, `volumes`, `Z`, `overlaps_network`, `overlaps_receptor` and `overlaps_genetics`. All seven must be present, and each must have the same sha256 in the copy and in the file whose md5 begins `341e229d88f4`, computed on both machines as `unzip -p FILE KEY.npy | sha256sum`. Report both md5 and the seven digests.
- **Checkpoint:** the last one. None is chosen by a score on the validation trials.
- **Stop rule.** If the loss of a seed becomes non-finite, stop that seed and report it. Do not restart it with other settings.

## Step 0, before any training (no GPU; send this back first)

1. **The code.** Whether 69250c3 is on the machine, and whether the launch above runs there without editing or adding a script. If not, stop.
2. **The diff** between the configuration of the final model and that of job K. It must touch the network's keys only.
3. **The network.** The parameter count of the network as built, against 14.8M. Any setting of the network that differs between 69250c3 and the commits of Table 4.9.
4. **The pool.** The md5 of the pool file on the machine and of the one the final launch names. If the pool is not on the machine, stop: Cesar decides where the job runs.
5. **The earlier run.** Its learning rate and schedule, as description. Job K keeps the final model's.
6. **Cost.** With Gladys's go-ahead, a throughput test of about 20 minutes on the card type the seeds will use. The context ramps up over the first half of a run, so the test is a separate short run (a small `RUN_ITERS`, its own output folder, its checkpoint discarded) whose second half is at 3,000 rows. Report the seconds per update at 3,000 rows, the hours forecast per seed as that figure times 250,000 (an upper bound), and the peak memory. On Myriad a job may last 48 hours (reported): if a seed is forecast above 40 hours, stop. Report the Python, PyTorch and CUDA versions; the final model was trained with PyTorch 2.3.1 (reported).
7. **The scoring.** Where the validation latents of E1-bin (encoder seed 0) and the stored per-trial rows of run 1 are; both are reported to be on siena. If checkpoints must move between machines, say so: Gladys or Cesar makes the transfer.

## The scoring

As rung f of run 1 (the scoring of 12 November, registered request of Brief 13): the 32 virtual trials, efficacy 1 and recovery 0, every training patient of the trial in the context.

- Per trial, the PEHE of job K is the mean of the PEHE of its three seeds. Predictions are not averaged across seeds.
- The final model (0.2788) and TabPFN-3.5 as S-learner (0.2693) are reused from the stored per-trial rows of run 1. Confirm the trials, pseudo-folds and test sizes are the same.
- Paired differences over the 16 deficits, on the mean of the two splits, t with 15 degrees of freedom.

## Readings, fixed now

Let k = job K minus the final model. Lower is better. The yardstick is half of the gap to TabPFN-3.5: 0.00475, as in the request for rung g.

- **A. This network closes half of the gap or more:** k is −0.00475 or lower, and its paired 95 % interval excludes zero.
- **B. It does not:** the lower limit of the interval is above −0.00475. If the whole interval is above zero, add "and it is worse".
- **Otherwise** the run does not decide at this precision, and the page says so.
- **Seeds.** Give the PEHE of each seed (mean of the two splits) and their SD (three values, n − 1). If the SD is above 0.01, the page says "unstable across seeds" beside the reading. The 0.01 is the auditor's choice, ten times the seed SD of the final model in this scenario (about 0.001). If a seed was stopped, no reading is given: the other seeds are reported as description.

**Also reported, without a reading:** job K minus TabPFN-3.5 as S-learner, with its interval.

**What the run cannot show.** The network, its size and the fit of the learning rate change together, so a difference cannot be put down to the architecture alone. The dissertation found a 21.4M single-axis model equal to the 4.8M one (Section 4.6); that bounds the effect of size within one family only. One network and one learning rate say nothing about TabICLv2 or about architectures in general.

## What to send back

Step 0 and the diff; one csv (estimator, trials, PEHE by split and mean, seed, paired difference with its 95 % interval) and the per-trial rows per seed; one page ending with the reading; the last lines of the training logs, the commits and the md5 of each file. Each statement labelled verified, reported or conjecture.

## Registration

**Revision 2, 16 November, before any training.** Revision 1 (sha256 beginning `64db1ce6`) named one launch option and would have trained a 6-layer network. The Myriad agent's reading of the scripts showed it; the auditor checked the lines at 69250c3. Changed: the launch options, the launch by seed, the memory rule, point 1 of step 0. Nothing else.

**Revision 3, 16 November, before any training.** Added: the pool may travel without its three identifying keys, with the check by array digests; the note that the training saves only at the end (both from the siena agent's reading of the code, reported). Nothing else.

**Revision 4, 16 November, before any training.** The stop rule of point 6 was 20 hours per seed, set from timings on a faster card; the Myriad agent forecasts 22 to 29 hours on its best card (conjecture). The rule now follows the 48-hour limit of a job, with a margin. The hours are no longer stated here: Gladys grants them. The pool copy is made with `zip -d` and checked by member digests, so both machines compute the same thing. No reading changed.

**Revision 5, 16 November, before any training.** After a second check of revisions 2 to 4 against the code at 69250c3 (`audit_request_job_K_rev4.md`: nothing wrong, several points imprecise). Completed: the variables of the launch, the link to the atlas, the wording of the checks, the seven members of the pool check, and the throughput test as a separate short run, because a 20-minute start of a full run never reaches 3,000 rows. The pool copy may be made by any method; the check is the seven member digests. No reading changed.

Registered by the auditor before any of these results exists; the sha256 is given to Gladys with the file. Keep the file as received, commit it unchanged on the branch, and report the commit and the sha256 you compute. If you received it as pasted text, do not commit the paste: ask for the file.
