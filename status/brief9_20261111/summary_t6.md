Task 6 — Stage 1 sweep: 2^(4−1) factorial and generic share 0.25 (24 runs), screened against R0
Decisions for César: (1) no level passes the screen, so under the preliminary protocol the next prior stays at R0 (final1M); (2) wide virtual-trial ranges (factor B) cost 0.013 PEHE, a robust harm in every run that has them; (3) generic share 0.75, effect term 0 and cosine decay are flat (|effect| ≤ 0.0014, none significant); (4) the preliminary protocol (docs/protocol_stage1.md, "preliminary") needs your review, the rotation-averaging point first.
Ran: commit cc83a62 (branch audit-brief9 = 6cb459c + preliminary protocol, md5 d9dd61f71a08, local, not pushed), stage1_jobs.sh, jobs F2–F8 and gw025 × seeds 0, 1, 2 (24 runs; `dist` dropped by the protocol), siena RTX 5090, 9 Nov 15:21 – 11 Nov 04:49 GMT, pool md5 341e229d88f4, atlas md5 fd1b531e7cf9; 24 checkpoints, 0 error lines. Scored on validation (clean context), E13a seed 0 latents, CPU; stage1_screen.py.
Table (stage1_screen.csv), pehe_paper, 16 deficits, gain = R0 − level (positive = better than R0), R0 mean 0.2788:
| Level | Change | Gain receptor | Gain genetics | Gain mean [interval] | Holm p | Passes |
| F2 | A + D | +0.0012 | −0.0017 | −0.0003 [−0.0024, +0.0019] | 1.00 | no |
| F3 | B + D | −0.0129 | −0.0150 | −0.0140 [−0.0185, −0.0095] | < 0.001 | no (worse) |
| F4 | A + B | −0.0155 | −0.0124 | −0.0139 [−0.0171, −0.0108] | < 0.001 | no (worse) |
| F5 | C + D | +0.0016 | −0.0002 | +0.0007 [−0.0017, +0.0032] | 1.00 | no |
| F6 | A + C | −0.0041 | −0.0039 | −0.0040 [−0.0070, −0.0010] | 0.048 | no (worse) |
| F7 | B + C | −0.0145 | −0.0121 | −0.0133 [−0.0173, −0.0093] | < 0.001 | no (worse) |
| F8 | A + B + C + D | −0.0155 | −0.0123 | −0.0139 [−0.0168, −0.0110] | < 0.001 | no (worse) |
| gw025 | generic 0.25 | −0.0003 | +0.0004 | +0.0001 [−0.0026, +0.0027] | 1.00 | no |
Main effects (stage1_screen_factorial.csv; 12 runs against 12, Holm over four), gain of the high level: A generic 0.75 −0.0014 [−0.0030, +0.0003]; B wide ranges −0.0129 [−0.0153, −0.0105], Holm p < 0.001; C CATE_W 0 −0.0006 [−0.0022, +0.0010]; D cosine +0.0010 [−0.0013, +0.0032]. Aliased pairs all |·| ≤ 0.0013, none significant.
Reading:
- Verified (stage1_screen.csv): no level reaches a 0.005 gain; the best point estimate is F5 at +0.0007. The protocol's combination rule therefore keeps R0.
- Verified (stage1_screen_factorial.csv): wide virtual-trial ranges are the only factor with an effect, and it is a harm of about 0.013 PEHE (about 5 % of the PEHE level), consistent across both splits and all four runs that contain B.
- Verified: generic share is flat between 0.25 (gw025), 0.5 (R0) and 0.75 (main effect A), within ±0.0015.
- Conjecture: the clean ranges match the benchmark's trial design, so widening them spends capacity on trials the benchmark never shows; the PFN at this prior and budget sits on a plateau for the other three factors.
- Conjecture: F6 (A + C, −0.0040, Holm p 0.048) is a marginal harm that the main effects do not explain (A and C are each flat); it may be the aliased AC = BD interaction (−0.0013, p 0.08) or noise.
Deviations from this runbook: `dist` not run (protocol section 5, after task 1); the protocol is preliminary and written before the sweep, not by César.
