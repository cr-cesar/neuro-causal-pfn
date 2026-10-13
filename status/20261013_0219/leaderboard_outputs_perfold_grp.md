# Per-fold study leaderboard

PEHE on the virtual-trial replica, paper definition, every representation fitted inside each fold (no anatomical leakage). Mean over seeds (population std). Lower is better.

| Exp | Variant | Budget | Task | Channel | Folds | Seeds | PEHE paper | Bal. acc. |
|-----|---------|--------|------|---------|-------|-------|------------|-----------|
| E1 | E1+x3 | chain | disconnectome | disconnectome | grp-giles-singles | 1 | 0.296 (0.000) | 0.851 |
| E1 | E1 | chain | disconnectome | disconnectome | grp-giles-singles | 3 | 0.304 (0.001) | 0.838 |
| E1 | E1 | chain | disconnectome | disconnectome | grp-giles | 3 | 0.304 (0.001) | 0.838 |
| E5 | E5+x3 | chain | disconnectome | disconnectome | grp-giles-singles | 1 | 0.310 (0.000) | 0.883 |
| E5 | E5 | chain | disconnectome | disconnectome | grp-giles-singles | 3 | 0.322 (0.002) | 0.858 |
| E5 | E5 | chain | disconnectome | disconnectome | grp-giles | 3 | 0.323 (0.001) | 0.861 |
| ref | volume | - | disconnectome | - | grp-giles-singles | 1 | 0.518 (0.000) | 0.527 |
| ref | volume | - | disconnectome | - | grp-giles | 1 | 0.518 (0.000) | 0.526 |
