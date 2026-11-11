Task 9 — Proximity check (stage 2 preparation)
Decisions for César: the threshold (fixed in advance by the plan). Two benchmark trials have a virtual network of the prior at |r| ≥ 0.9; sixteen at ≥ 0.8.
Ran: commit 6cb459c (branch audit-brief9), siena CPU, --draws 10000 (10,000 processes, 19,983 virtual networks), 7 Nov 21:36, pool md5 341e229d88f4. The script's self-check of its vectorised τ against the DGP's cate() on the first 20 draws passed (no stop).
Table (proximity_summary.csv): median over the 32 trials of the max |corr| of τ with a virtual network 0.802; trials with a virtual network at |r| ≥ 0.9: 2; at ≥ 0.8: 16.
Closest trials (proximity_by_trial.csv): genetics network 10 (948 enrolled) 0.951, 3 virtual networks ≥ 0.9, best agreement on enrolled 0.986; genetics network 13 (129 enrolled) 0.935, 1 network ≥ 0.9, agreement 0.977. Next: genetics 2 0.883, receptor 6 0.881, receptor 16 0.857.
Reading:
- Verified (files above): at 10,000 draws the prior produces effect vectors close to most benchmark trials (half at ≥ 0.8), and near-copies (≥ 0.9) for two genetics trials.
- Conjecture: these two trials are the ones where the in-prior/out-of-prior distinction matters most for stage 2; the decision depends on the threshold, not applied here.
Deviations from this runbook: none.
