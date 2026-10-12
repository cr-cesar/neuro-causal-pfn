# Per-fold study leaderboard

PEHE on the virtual-trial replica, paper definition, every representation fitted inside each fold (no anatomical leakage). Mean over seeds (population std). Lower is better.

| Exp | Variant | Budget | Task | Channel | Folds | Seeds | PEHE paper | Bal. acc. |
|-----|---------|--------|------|---------|-------|-------|------------|-----------|
| E1 | E1 | published | disconnectome | disconnectome | grp-giles-singles | 3 | 0.279 (0.001) | 0.839 |
| E3 | E3[backbone=resnet18] | published | disconnectome | disconnectome | grp-giles-singles | 3 | 0.278 (0.000) | 0.842 |
| E3 | E3[backbone=resnet50] | published | disconnectome | disconnectome | grp-giles-singles | 3 | 0.279 (0.001) | 0.843 |
| E3 | E3[backbone=cnn] | published | disconnectome | disconnectome | grp-giles-singles | 3 | 0.280 (0.000) | 0.838 |
| E3 | E3[backbone=wide] | published | disconnectome | disconnectome | grp-giles-singles | 3 | 0.281 (0.001) | 0.844 |
| E11b | E11b[backbone=resnet18,w_dice=1.0] | published | disconnectome | disconnectome | grp-giles-singles | 3 | 0.293 (0.001) | 0.844 |
| E11b | E11b[backbone=resnet18,w_dice=0.1] | published | disconnectome | disconnectome | grp-giles-singles | 3 | 0.293 (0.001) | 0.843 |
| E11b | E11b[backbone=cnn,w_dice=1.0] | published | disconnectome | disconnectome | grp-giles-singles | 3 | 0.293 (0.001) | 0.841 |
| E11b | E11b[backbone=cnn,w_dice=0.1] | published | disconnectome | disconnectome | grp-giles-singles | 3 | 0.293 (0.001) | 0.840 |
