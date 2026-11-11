Task 3 — Leave-out at 1M with three seeds: held vs seen, run as a random effect (Brief 9 point 2)
Decisions for César: none needed to proceed. Leaving a network out of the prior costs nothing measurable on it: held − seen = +0.0004 PEHE, REML interval [−0.0015, +0.0023]. A penalty above ~0.002 for unseen networks is excluded; the old single-seed bound of ~0.008 shrinks fourfold. Criterion 4 ("keep the advantage with networks left out") is met at this prior.
Ran: commit 6cb459c (branch audit-brief9), lno3_launch.sh, 4 groups (networks 1–4, 5–8, 9–12, 13–16 left out of every channel; 60 atoms remain in each run) × seeds 0, 1, 2, 250,000 steps (1M trials), siena RTX 5090, 7 Nov 21:36 – 8 Nov 18:19, pool md5 341e229d88f4; 12 checkpoints, 0 error lines. Scored on validation (clean context), E13a seed 0 latents, lno3_score.sh; lno_cluster.py for the run effect.
Table (lno3_cluster.csv, paired/*.csv), pehe_paper, 12 runs × 16 deficits:
| Contrast | Estimate | Naive paired t | Cluster mean (n = 12) | REML (run random effect) |
| Held − seen, both splits | +0.0004 | [−0.0013, +0.0021], p 0.63 | [−0.0019, +0.0027] | [−0.0015, +0.0023], se 0.0009 |
| Held − seen, receptor | +0.0014 | [−0.0015, +0.0044] | [−0.0025, +0.0054] | [−0.0017, +0.0045] |
| Held − seen, genetics | −0.0007 | [−0.0033, +0.0020] | [−0.0026, +0.0013] | [−0.0034, +0.0021] |
Mean PEHE held 0.2795, seen 0.2791. Against final1M (3 seeds, no network left out): held +0.0007 (t p 0.45), seen +0.0003 (t p 0.76). REML components (both): sd_run 0.0014, sd_resid 0.0060. Prescriptive balanced accuracy, both: held − seen +0.0023, REML [−0.0109, +0.0154] (2 deficits undefined).
Reading:
- Verified (lno3_cluster.csv): with three seeds at the final budget, a network the prior never contained is scored as well as one it contained: the held-minus-seen gap is +0.0004, and every interval excludes a penalty above 0.0027.
- Verified (paired CSVs): removing four networks from the prior does not cost the model anything against final1M either (held +0.0007, seen +0.0003, both n.s.).
- Verified: the run effect is small (sd 0.0014 against 0.0060 residual); the REML interval is close to the naive one, so Claude's clustering concern (point 2) changes little here.
- Conjecture: the virtual-trial prior teaches a mechanism (threshold rule on network halves) that transfers to unseen networks, rather than memorising the benchmark networks; this does not test mechanisms outside the family (the E4 effect-shape cell is still needed for that).
- Verified: the bound (0.0023) is about 3 % of the anatomy gain (~0.08, Table 4.9); as in task 3a, that gain comes from another configuration, so the comparison is indicative only.
Deviations from this runbook: none. The 12 trainings ran 3 at a time; the first batch shared the CPU with task 2 (≈7 h instead of ≈5 h).
