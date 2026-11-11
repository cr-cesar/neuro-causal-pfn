Task 5b — R0 trained on KCH half A, scored on validation (Brief 9 point 8, task 5)
Decisions for César: whether to adopt the KCH split as the fresh test. Training R0 on half the pool (603 lesions) costs 0.0025 PEHE against final1M (interval [−0.0051, +0.0001], p 0.058). That is below the 0.005 screening rule and not significant, but the sign is consistent across splits. Together with 5a (thin hit-classes in 14 of 32 B-half trials), the split looks usable for PEHE but weak for prescriptive balanced accuracy.
Ran: commit 6cb459c (branch audit-brief9), sweep_axis.sh kchA, level R0, seeds 0, 1, 2, siena RTX 5090, 8 Nov 18:19 – 23:32 GMT; pool half A md5 b4104431c261, atlas md5 fd1b531e7cf9; checkpoints 225ca75355d9 / cfc5ea1e21c7 / 32cb5b2bd073, 0 error lines. Scored on validation (clean context, 0 of 1291 queries with a same-patient image in the training side), E13a seed 0 latents, score_val_clean.sh on CPU; stage1_screen.py. Nothing scored on half B.
Table (r0_half/kchA_vs_final1M.csv, md5 f63c7b57d61d), pehe_paper, 16 deficits, gain = final1M − kchA:
| | receptor | genetics | mean of splits |
| kchA s0 / s1 / s2 | 0.296 / 0.295 / 0.300 | 0.260 / 0.268 / 0.267 | |
| Mean, final1M → kchA | | | 0.2788 → 0.2813 |
| Gain | −0.0040 | −0.0009 | −0.0025 [−0.0051, +0.0001], p 0.058 (Holm 0.058) |
| Deficits where kchA is better | | | 5 of 16 |
| Screen | same sign on both splits: yes; passes 0.005 rule: no; claim bar 0.02: no | | |
Reading:
- Verified (kchA_vs_final1M.csv): halving the training pool costs at most ~0.005 PEHE on validation (upper end of the interval); the point estimate is 0.0025, about 1% of the PEHE level.
- Verified (score logs): the three seeds agree within 0.005 on each split; the cost is concentrated on receptor (−0.0040) rather than genetics (−0.0009).
- Reported (5a): half B has ≥ 50 enrolled in every trial, but 14 of 32 trials have a hit-class under 10.
- Conjecture: R0 depends weakly on pool size at this scale, so a model trained on half A is a fair stand-in for final1M when testing on half B; the test half will be informative for PEHE, noisy for balanced accuracy on the thin trials.
Deviations from this runbook: none (scoring on CPU with nice -n 10; the GPU was free before the launch).
