Task 5a — KCH split checks (Brief 9 point 8)
Decisions for César: whether a 604-lesion test half is acceptable. No B-half trial has fewer than 50 enrolled (minimum 50, median 142), but 14 of 32 B-half trials have a class (only-half-1 or only-half-2) below 10, against 8 of 32 in the whole pool; 6 of those classes are structurally 0 in the whole pool too.
Ran: commit 6cb459c (branch audit-brief9), siena CPU, split seed 0, 7 Nov, pool md5 341e229d88f4 (E13a KCH; cache channel disconnectome, threshold 0.5). Patient check with a file-name regex (kept on siena, not in the report).
Table (kch_split_check.csv, kch_trial_sizes.csv):
- 1,207 lesions; 1,207 distinct latent rows, 1,207 distinct overlap rows, 1,207 distinct patients (every file matched the regex): one lesion per patient, no exact repeats.
- Half A (train) 603 lesions, md5 b4104431c261; half B (test) 604 lesions, md5 57bb79b68be8. Both stay on siena.
- B half: enrolled min 50 / median 142; trials < 50 enrolled: 0; smallest: genetics 12 (50), receptor 5 (51), genetics 5 (52), receptor 12 (53), genetics 13 (65), receptor 13 (68).
- B-half trials with a class < 10 (only-half-1 / only-half-2): receptor 2 (0/43), 4 (0/80), 5 (6/18), 6 (93/0), 7 (0/95), 8 (0/15), 9 (81/6), 10 (6/163), 13 (35/2), 14 (7/112); genetics 5 (9/15), 8 (5/13), 9 (116/1), 13 (52/1).
Reading:
- Verified (kch_split_check.csv): the pool has one lesion per patient, so a split by lesion is a split by patient.
- Verified (kch_trial_sizes.csv): enrolment per trial is adequate in the B half (≥ 50), but the arms defined by which half is hit become thin: 14 trials have a class under 10, and several are already empty in the whole pool (a network half that is never hit alone).
- Conjecture: the empty classes are anatomical (one half of the network is nested in the other's territory) and would affect any KCH-based test, not only the split; prescriptive balanced accuracy will be undefined or noisy on those trials.
Deviations from this runbook: none (the --patient-regex was given; neither it nor any name appears in outputs).
