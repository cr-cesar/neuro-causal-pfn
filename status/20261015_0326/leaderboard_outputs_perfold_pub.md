# Per-fold study leaderboard

PEHE on the virtual-trial replica, paper definition, every representation fitted inside each fold (no anatomical leakage). Mean over seeds; seed SD = population SD of the seed means (not a CI); 95% CI = over the 16 deficits (Giles's interval). Lower is better.

| Exp | Variant | Budget | Task | Channel | Folds | Seeds | PEHE paper (seed SD) | 95% CI over deficits | Bal. acc. |
|-----|---------|--------|------|---------|-------|-------|----------------------|----------------------|-----------|
| E1 | E1 | published | disconnectome | disconnectome | grp-giles-singles | 3 | 0.294 (0.001) | 0.260–0.329 | 0.844 |
| E3 | E3[backbone=resnet18] | published | disconnectome | disconnectome | grp-giles-singles | 3 | 0.293 (0.001) | 0.259–0.327 | 0.847 |
| E3 | E3[backbone=resnet50] | published | disconnectome | disconnectome | grp-giles-singles | 3 | 0.295 (0.000) | 0.261–0.329 | 0.847 |
| E3 | E3[backbone=cnn] | published | disconnectome | disconnectome | grp-giles-singles | 3 | 0.296 (0.001) | 0.262–0.330 | 0.849 |
| E3 | E3[backbone=wide] | published | disconnectome | disconnectome | grp-giles-singles | 3 | 0.298 (0.001) | 0.263–0.333 | 0.847 |
| E11b | E11b[backbone=resnet18,w_dice=0.1] | published | disconnectome | disconnectome | grp-giles-singles | 3 | 0.283 (0.002) | 0.248–0.317 | 0.859 |
| E11b | E11b[backbone=resnet18,w_dice=1.0] | published | disconnectome | disconnectome | grp-giles-singles | 3 | 0.283 (0.002) | 0.248–0.318 | 0.859 |
| E11b | E11b[backbone=cnn,w_dice=1.0] | published | disconnectome | disconnectome | grp-giles-singles | 3 | 0.285 (0.001) | 0.249–0.321 | 0.854 |
| E11b | E11b[backbone=cnn,w_dice=0.1] | published | disconnectome | disconnectome | grp-giles-singles | 3 | 0.285 (0.001) | 0.250–0.321 | 0.854 |
| ref | volume | - | disconnectome | - | grp-giles-singles | 1 | 0.518 (0.000) | 0.458–0.577 | 0.527 |
