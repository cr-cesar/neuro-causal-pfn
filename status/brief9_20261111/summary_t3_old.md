Task 3 (part 1) — Brief 9 point 2 on the old leave-out run: held vs seen with run as a random effect
Decisions for César: none. On the old single-seed run (2 Nov, 500k, 4 groups) the held-minus-seen gap stays uninformative under every interval: +0.0028, with the REML interval [−0.0022, +0.0077]. Superseded by the three-seed 1M runs of Task 3 (12 runs, finished 8 Nov 18:19): +0.0004, REML [−0.0015, +0.0023].
Ran: commit 6cb459c (branch audit-brief9), lno_cluster.py on the 2 Nov scorings (labels lno_g{1..4}_s0, validation, clean context), siena CPU, 7 Nov. Pipeline check: the naive interval reproduces the reported +0.0028 [−0.0012, +0.0067] exactly.
Table (lno_old_cluster.csv), pehe_paper, held − seen, 4 runs × 16 deficits:
| Split | Estimate | Naive paired t | Cluster mean (n = 4) | REML (run random effect) |
| Both | +0.0028 | [−0.0012, +0.0067], p 0.15 | [−0.0058, +0.0113], p 0.38 | [−0.0022, +0.0077], se 0.0016 |
| Receptor | +0.0021 | [−0.0016, +0.0058] | [−0.0058, +0.0100] | [−0.0027, +0.0069] |
| Genetics | +0.0034 | [−0.0023, +0.0092] | [−0.0081, +0.0149] | [−0.0043, +0.0111] |
REML components (both): sd_run 0.0033, sd_resid 0.0063. Prescriptive balanced accuracy, both: held − seen −0.0119, REML [−0.0339, +0.0140] (2 deficits undefined).
Reading:
- Verified (lno_old_cluster.csv): accounting for the shared run widens the interval from [−0.0012, +0.0067] to [−0.0022, +0.0077]; the estimate does not change. Claude's concern (point 2) holds in direction for the t-based REML interval, but the change is small.
- Verified: the upper bound is about 0.008, a tenth of the anatomy gain (~0.08 in Table 4.9), so a large region-specific share of that gain stays excluded, as Claude argued; but the gain and the leave-out runs differ in configuration (anatomy family present, E1 latents vs virtual-trial family, E13a), so this is a bound on this prior only (conjecture for the transfer).
- Verified: the run effect is half the residual spread (0.0033 vs 0.0063): runs do shift all deficits, but most of the spread is per deficit.
Deviations from this runbook: none.
