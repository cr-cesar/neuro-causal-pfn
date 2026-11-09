# Stage 1 protocol (preliminary)

Status: preliminary. Fixed before the first stage 1 sweep run; to be reviewed.

## 1. Screening rule

A level passes the screen against R0 (final1M) when all three hold:

- a three-seed gain of at least 0.005 PEHE on the mean of the two splits (receptor, genetics), with gain = R0 − level;
- the same sign of the gain on both splits;
- a paired interval over the 16 deficits that excludes zero after Holm across the levels screened (Holm-adjusted p < 0.05).

A gain of at least 0.02 is required before any claim in the text. Scoring uses the validation pseudo-folds with clean context only; nothing is scored on the UCLH test folds.

## 2. Primary biased cell

Location bias at strength 0.5. It is not used in stage 1; it is fixed here for the later biased-cell tests.

## 3. Factorial and combination rule

- Design: 2^(4−1) fractional factorial, generator D = ABC, three seeds per run; F1 = R0 = final1M.
  - A: generic share 0.5 / 0.75
  - B: virtual-trial ranges clean / wide
  - C: effect term CATE_W 0.5 / 0
  - D: learning rate constant / cosine
- Main effects A–D are each estimated from 12 runs against 12, with Holm over the four; two-factor interactions are aliased in pairs and reported, not tested.
- Combination: the next prior combines the levels of the factors whose main effect passes the screening rule; factors that do not pass stay at their R0 level.
- Extra levels: generic share 0.25 (`gw025`), screened against R0 by the same rule.
- Size is re-tested on R1 only if R1's prior is more diverse than R0's.

## 4. Rotation averaging

Not part of the primary criterion. The primary score is the validation scoring as used for R0 and all earlier levels; the average over pseudo-fold rotations is reported as a secondary analysis.

## 5. Levels not run

- `dist` (DISTINCT=1): dropped. Item 0a found that 99–100 % of contexts over 1,024 rows are longer than their slice, so the level cannot make contexts distinct.
- "Allocation off": not run. Item 0a found 0.06 % agnostic processes after the verifier, below the 1 % rule.
- Generic share 0 on R0: not run in stage 1.
