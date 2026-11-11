Task 2 — Wave 0 item 1: nine cells on validation, with pehe_implied
Decisions for César: (1) in the noisy cells, against the implied CATE TE(1 − RE)·τ, the final model is clearly worse than TabPFN-3.5 and CausalPFN (+0.10 to +0.17 PEHE): it does not shrink its effects as much as the noise requires, so the direction left open in Brief 8 (explanation 5) is "the final model overshoots"; (2) under location bias (0.95 and the default) the final model beats every estimator by 0.05–0.07, the result that supports the biased-cell criterion; (3) the primary biased cell to fix in the protocol (location bias 0.5 proposed) is the weakest one: at 0.5 TabPFN is still 0.014 better.
Ran: commit 6cb459c (branch audit-brief9); scorer ~/ncp_g8_b9 (99a3d8c + giles_replica from the Brief 8 patch, commit bdfd84b); per-fold code ~/ncp_pf_clean 79e8b3c; siena, 7–8 Nov 21:47–03:54; final1M seeds 0, 1, 2 (~/p2_bcb/pfn_final1M), E13a seed 0 latents; validation pseudo-folds with clean context. Gate: the patched scorer reproduces final1M seed 0 in the ideal cell exactly (difference 0.000 on both splits).
Table (wave0_tables.csv, wave0_tables_paired.csv): PEHE, mean of splits; d = final − other over 16 deficits; * = 95 % interval excludes 0. Metric: pehe_implied in the noisy cells, pehe_paper elsewhere.
| Cell | Final (3 seeds) | TabPFN S | CausalPFN | LR (T) | ET (T) |
| noisy_te075 | 0.309 | 0.211 (+0.097*) | 0.222 (+0.086*) | 0.326 (−0.017) | 0.251 (+0.058*) |
| noisy_te050 | 0.255 | 0.131 (+0.124*) | 0.132 (+0.122*) | 0.278 (−0.024*) | 0.188 (+0.067*) |
| noisy_te025 | 0.213 | 0.042 (+0.171*) | 0.050 (+0.162*) | 0.248 (−0.035*) | 0.148 (+0.064*) |
| location_bias_050 | 0.287 | 0.273 (+0.014*) | 0.289 (−0.001) | 0.316 (−0.029*) | 0.363 (−0.076*) |
| location_bias_095 | 0.345 | 0.406 (−0.061*) | 0.399 (−0.054*) | 0.396 (−0.052*) | 0.449 (−0.105*) |
| location_bias | 0.348 | 0.421 (−0.072*) | 0.401 (−0.053*) | 0.398 (−0.050*) | 0.454 (−0.106*) |
| responsiveness_060 | 0.282 | 0.274 (+0.008) | 0.283 (−0.001) | 0.313 (−0.030*) | 0.361 (−0.079*) |
| responsiveness_075 | 0.300 | 0.294 (+0.006) | 0.306 (−0.006) | 0.326 (−0.026*) | 0.383 (−0.083*) |
| responsiveness_090 | 0.358 | 0.367 (−0.008) | 0.386 (−0.028*) | 0.387 (−0.029*) | 0.453 (−0.095*) |
Against ±1 (pehe_paper), the noisy cells give final / TabPFN 0.500 / 0.470 (te075), 0.623 / 0.633 (te050), 0.692 / 0.688 (te025).
Reading:
- Verified (paired CSV): scored against the implied CATE, the final model's PEHE in the noisy cells is 0.21–0.31, against 0.04–0.22 for TabPFN and CausalPFN; the gap grows as TE falls (te025 largest). Against ±1 the estimators are close, as Table 4.7 reported.
- Conjecture: the final prior's virtual-trial effects are ±TE(1 − RE) with TE 0.9–1 and RE 0–0.05, so the model has never seen small effects and cannot shrink; the S-learners shrink because the outcomes are noisy. A wider TE range in the prior (the "wide ranges" factor B of stage 1) is the natural fix.
- Verified: under location bias at 0.95 and at the default strength the final model is better than every estimator (0.05–0.07 against TabPFN and CausalPFN); at 0.5 TabPFN is better by 0.014.
- Verified: in the responsiveness cells the final model is level with TabPFN and CausalPFN at 0.60 and 0.75, and better than CausalPFN at 0.90 (−0.028); it beats LR and ET everywhere.
Deviations from this runbook: wave0_cells.sh was not run as is, because on siena CausalPFN lives in ~/env_cpfn and TabPFN in ~/env_panel; the same calls (same labels and arguments) ran from ~/p2_bcb/brief9_t2_cells.sh with the matching env, two scorings at a time. TabPFN shared the GPU with the task 3 trainings (peak ~31 of 32.6 GB); no cell failed.
