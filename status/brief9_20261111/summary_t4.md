Task 4 — Learning curve of the latent reference (validation only)
Decisions for César: none yet. On the only cells large enough to compare (20 matched cells: network 10 in both splits × 10 pseudo-folds), the reference is still falling between 2,000 patients and all, but by less each doubling; the full table is in learning_curve_summary_E1.csv.
Ran: commit 6cb459c (branch audit-brief9), siena, TabPFN 9.2.0 (model v3.5, ~/env_panel) on the GPU shared with the task 3 trainings, E1 seed 0 latents, validation pseudo-folds with clean context (0 of 1,291 queries with another image of the patient in the training side), sizes 500/1,000/2,000/all, 3 repeats, 7 Nov 21:47–22:52. Coverage cache recomputed from the disconnectome maps (kept on siena).
Table (learning_curve_summary_E1.csv), PEHE mean of splits (receptor / genetics):
- Matched cells with > 2,000 enrolled training patients (20 cells, median 2,604): n = 500 0.279 (0.278 / 0.280); 1,000 0.247 (0.237 / 0.257); 2,000 0.224 (0.213 / 0.235); all 0.216 (0.203 / 0.230).
- Cells with at least n (more cells at small n): 500 0.258 (230 cells); 1,000 0.234 (120); 2,000 0.224 (20); all 0.238 (320 cells, median 794 enrolled).
Reading:
- Verified (summary CSV): on the matched cells the latent reference keeps improving with more patients: −0.032 from 500 to 1,000, −0.022 from 1,000 to 2,000, −0.008 from 2,000 to all (~2,600).
- Conjecture: the reference is still somewhat sample-limited at the benchmark's size, with diminishing returns; the 0.242 (test) floor is not a hard floor. Only 20 matched cells, all from network 10 (receptor and genetics), support this, so the slope beyond 2,000 is uncertain.
- Verified: validation values are not comparable with the test 0.242/0.243 in level (runbook note); compare shapes only.
Deviations from this runbook: none in the commands. One CUDA allocation warning (OOM, then retry) at 21:57 while TabPFN shared the GPU with three trainings; all 1,430 cell fits completed without missing values and the trainings logged no error.
