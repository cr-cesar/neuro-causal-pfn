# Per-fold study leaderboard

PEHE on the virtual-trial replica, paper definition, every representation fitted inside each fold (no anatomical leakage). Mean over seeds; seed SD = population SD of the seed means (not a CI); 95% CI = over the 16 deficits (Giles's interval). Lower is better.

| Exp | Variant | Budget | Task | Channel | Folds | Seeds | PEHE paper (seed SD) | 95% CI over deficits | Bal. acc. |
|-----|---------|--------|------|---------|-------|-------|----------------------|----------------------|-----------|
| E1 | E1 | published | disconnectome | disconnectome | grp-giles-singles | 3 | 0.279 (0.001) | 0.248–0.311 | 0.839 |
| E1 | E1 | published | disconnectome | both | grp-giles-singles | 3 | 0.305 (0.001) | 0.272–0.338 | 0.830 |
| E1 | E1 | published | disconnectome | lesion | grp-giles-singles | 3 | 0.346 (0.001) | 0.316–0.376 | 0.794 |
| E3 | E3[backbone=resnet18] | published | disconnectome | disconnectome | grp-giles-singles | 3 | 0.278 (0.000) | 0.247–0.309 | 0.842 |
| E3 | E3[backbone=resnet50] | published | disconnectome | disconnectome | grp-giles-singles | 3 | 0.279 (0.001) | 0.248–0.310 | 0.843 |
| E3 | E3[backbone=cnn] | published | disconnectome | disconnectome | grp-giles-singles | 3 | 0.280 (0.000) | 0.248–0.312 | 0.838 |
| E3 | E3[backbone=wide] | published | disconnectome | disconnectome | grp-giles-singles | 3 | 0.281 (0.001) | 0.249–0.312 | 0.844 |
| E4 | E4[40+60] | published | disconnectome | disconnectome | grp-giles-singles | 1 | 0.278 (0.000) | 0.246–0.311 | 0.847 |
| E4 | E4[25+75] | published | disconnectome | disconnectome | grp-giles-singles | 1 | 0.278 (0.000) | 0.246–0.311 | 0.847 |
| E4 | E4[75+75] | published | disconnectome | disconnectome | grp-giles-singles | 1 | 0.279 (0.000) | 0.247–0.310 | 0.847 |
| E4 | E4[50+50] | published | disconnectome | disconnectome | grp-giles-singles | 1 | 0.281 (0.000) | 0.249–0.313 | 0.836 |
| E4 | E4[60+40] | published | disconnectome | disconnectome | grp-giles-singles | 1 | 0.284 (0.000) | 0.251–0.316 | 0.833 |
| E4 | E4[25+25] | published | disconnectome | disconnectome | grp-giles-singles | 1 | 0.287 (0.000) | 0.256–0.319 | 0.835 |
| E4 | E4[75+25] | published | disconnectome | disconnectome | grp-giles-singles | 1 | 0.288 (0.000) | 0.257–0.320 | 0.826 |
| E4 | E4[100+100] | published | disconnectome | disconnectome | grp-giles-singles | 1 | 0.295 (0.000) | 0.262–0.328 | 0.838 |
| E11b | E11b[backbone=resnet18,w_dice=1.0] | published | disconnectome | disconnectome | grp-giles-singles | 3 | 0.293 (0.001) | 0.260–0.325 | 0.844 |
| E11b | E11b[backbone=resnet18,w_dice=0.1] | published | disconnectome | disconnectome | grp-giles-singles | 3 | 0.293 (0.001) | 0.261–0.324 | 0.843 |
| E11b | E11b[backbone=cnn,w_dice=1.0] | published | disconnectome | disconnectome | grp-giles-singles | 3 | 0.293 (0.001) | 0.260–0.326 | 0.841 |
| E11b | E11b[backbone=cnn,w_dice=0.1] | published | disconnectome | disconnectome | grp-giles-singles | 3 | 0.293 (0.001) | 0.260–0.326 | 0.840 |
