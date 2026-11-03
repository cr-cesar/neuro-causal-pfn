# Per-fold study leaderboard

PEHE on the virtual-trial replica, paper definition, every representation fitted inside each fold (no anatomical leakage). Mean over seeds; seed SD = population SD of the seed means (not a CI); 95% CI = over the 16 deficits (Giles's interval). Lower is better.

| Exp | Variant | Budget | Task | Channel | Folds | Seeds | PEHE paper (seed SD) | 95% CI over deficits | Bal. acc. |
|-----|---------|--------|------|---------|-------|-------|----------------------|----------------------|-----------|
| E1 | E1+x3 | chain | disconnectome | disconnectome | grp-giles-singles | 1 | 0.296 (0.000) | 0.259–0.332 | 0.851 |
| E1 | E1 | chain | disconnectome | disconnectome | grp-giles-singles | 3 | 0.304 (0.001) | 0.268–0.339 | 0.838 |
| E1 | E1 | chain | disconnectome | disconnectome | grp-giles | 3 | 0.304 (0.001) | 0.269–0.340 | 0.838 |
| E5 | E5+x3 | chain | disconnectome | disconnectome | grp-giles-singles | 1 | 0.310 (0.000) | 0.271–0.348 | 0.883 |
| E5 | E5 | chain | disconnectome | disconnectome | grp-giles-singles | 3 | 0.322 (0.002) | 0.281–0.363 | 0.858 |
| E5 | E5 | chain | disconnectome | disconnectome | grp-giles | 3 | 0.323 (0.001) | 0.282–0.364 | 0.861 |
| ref | volume | - | disconnectome | - | grp-giles-singles | 1 | 0.518 (0.000) | 0.458–0.577 | 0.527 |
| ref | volume | - | disconnectome | - | grp-giles | 1 | 0.518 (0.000) | 0.458–0.578 | 0.526 |
