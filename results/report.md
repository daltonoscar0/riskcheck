## Headline: out-of-sample calibration, 2000 onwards

49 industry portfolios (49 assets), 1w horizon, 1345 non-overlapping observations. A well-calibrated model gives a bias statistic of 1.00; the band for the full test period is ±0.039. "In band" is the share of rolling 52-observation windows within ±0.196 of 1.

| Estimator | Random long-only | Random long-short | Min-variance | Min-var in band | Min-var vol (ann.) | Log score |
| :-- | --: | --: | --: | --: | --: | --: |
| Sample | 0.996 | 1.033 | **1.190** | 64% | 11.71% | -6.827 |
| EWMA | 0.961 | 0.999 | **1.108** | 72% | 11.40% | -6.895 |
| Ledoit-Wolf | 1.033 | 1.024 | **1.296** | 41% | 11.50% | -6.903 |
| PCA factor | 1.029 | 1.035 | **1.698** | 13% | 13.01% | -6.816 |
| FF5+Mom factor | 1.009 | 1.007 | **1.591** | 14% | 12.48% | -6.873 |
| EWMA + VRA | 0.941 | 0.985 | **1.082** | 67% | 11.40% | -6.899 |

## Minimum variance, by asset set and horizon

| Estimator | Asset set | Horizon | Obs | Min-var bias | Band | In band | Min-var vol (ann.) | Forecast vol (ann.) | Gross leverage | Log score |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| Sample | 49 assets | 1w | 1345 | 1.190 | ±0.039 | 64% | 11.71% | 10.05% | 3.38 | -6.827 |
| EWMA | 49 assets | 1w | 1345 | 1.108 | ±0.039 | 72% | 11.40% | 10.30% | 3.35 | -6.895 |
| Ledoit-Wolf | 49 assets | 1w | 1345 | 1.296 | ±0.039 | 41% | 11.50% | 8.96% | 3.30 | -6.903 |
| PCA factor | 49 assets | 1w | 1345 | 1.698 | ±0.039 | 13% | 13.01% | 7.80% | 3.15 | -6.816 |
| FF5+Mom factor | 49 assets | 1w | 1345 | 1.591 | ±0.039 | 14% | 12.48% | 7.87% | 3.00 | -6.873 |
| EWMA + VRA | 49 assets | 1w | 1345 | 1.082 | ±0.039 | 67% | 11.40% | 10.65% | 3.35 | -6.899 |
| Sample | 49 assets | 1m | 336 | 1.311 | ±0.077 | 46% | 12.47% | 10.06% | 3.38 | -5.463 |
| EWMA | 49 assets | 1m | 336 | 1.218 | ±0.077 | 57% | 12.16% | 10.29% | 3.35 | -5.524 |
| Ledoit-Wolf | 49 assets | 1m | 336 | 1.398 | ±0.077 | 16% | 12.07% | 8.97% | 3.30 | -5.516 |
| PCA factor | 49 assets | 1m | 336 | 1.800 | ±0.077 | 0% | 13.65% | 7.82% | 3.15 | -5.428 |
| FF5+Mom factor | 49 assets | 1m | 336 | 1.768 | ±0.077 | 0% | 13.20% | 7.88% | 3.00 | -5.481 |
| EWMA + VRA | 49 assets | 1m | 336 | 1.195 | ±0.077 | 60% | 12.16% | 10.65% | 3.35 | -5.523 |
| Sample | 25 assets | 1w | 1345 | 1.285 | ±0.039 | 56% | 13.34% | 10.91% | 5.40 | -8.301 |
| EWMA | 25 assets | 1w | 1345 | 1.167 | ±0.039 | 58% | 12.72% | 11.24% | 5.40 | -8.388 |
| Ledoit-Wolf | 25 assets | 1w | 1345 | 1.340 | ±0.039 | 44% | 13.46% | 10.07% | 4.18 | -8.340 |
| PCA factor | 25 assets | 1w | 1345 | 1.627 | ±0.039 | 17% | 13.51% | 8.53% | 5.77 | -8.315 |
| FF5+Mom factor | 25 assets | 1w | 1345 | 1.495 | ±0.039 | 23% | 13.24% | 9.21% | 5.10 | -8.322 |
| EWMA + VRA | 25 assets | 1w | 1345 | 1.128 | ±0.039 | 46% | 12.72% | 11.86% | 5.40 | -8.390 |
| Sample | 25 assets | 1m | 336 | 1.567 | ±0.077 | 37% | 15.28% | 10.90% | 5.40 | -6.841 |
| EWMA | 25 assets | 1m | 336 | 1.441 | ±0.077 | 41% | 14.64% | 11.23% | 5.40 | -6.937 |
| Ledoit-Wolf | 25 assets | 1m | 336 | 1.636 | ±0.077 | 25% | 15.13% | 10.06% | 4.18 | -6.885 |
| PCA factor | 25 assets | 1m | 336 | 2.052 | ±0.077 | 0% | 16.15% | 8.53% | 5.77 | -6.804 |
| FF5+Mom factor | 25 assets | 1m | 336 | 1.849 | ±0.077 | 7% | 15.07% | 9.20% | 5.10 | -6.827 |
| EWMA + VRA | 25 assets | 1m | 336 | 1.390 | ±0.077 | 36% | 14.64% | 11.86% | 5.40 | -6.936 |

## Random portfolios

| Estimator | Asset set | Horizon | Long-only bias | In band | Long-short bias | In band |
| :-- | --: | --: | --: | --: | --: | --: |
| Sample | 49 assets | 1w | 0.996 | 44% | 1.033 | 54% |
| EWMA | 49 assets | 1w | 0.961 | 53% | 0.999 | 60% |
| Ledoit-Wolf | 49 assets | 1w | 1.033 | 63% | 1.024 | 74% |
| PCA factor | 49 assets | 1w | 1.029 | 63% | 1.035 | 73% |
| FF5+Mom factor | 49 assets | 1w | 1.009 | 73% | 1.007 | 75% |
| EWMA + VRA | 49 assets | 1w | 0.941 | 57% | 0.985 | 57% |
| Sample | 49 assets | 1m | 0.978 | 25% | 1.001 | 71% |
| EWMA | 49 assets | 1m | 0.944 | 39% | 0.973 | 75% |
| Ledoit-Wolf | 49 assets | 1m | 1.003 | 80% | 1.001 | 89% |
| PCA factor | 49 assets | 1m | 0.999 | 80% | 1.012 | 89% |
| FF5+Mom factor | 49 assets | 1m | 0.986 | 85% | 0.986 | 88% |
| EWMA + VRA | 49 assets | 1m | 0.921 | 51% | 0.963 | 74% |
| Sample | 25 assets | 1w | 1.040 | 46% | 1.078 | 47% |
| EWMA | 25 assets | 1w | 0.995 | 55% | 1.022 | 53% |
| Ledoit-Wolf | 25 assets | 1w | 1.036 | 73% | 1.021 | 69% |
| PCA factor | 25 assets | 1w | 1.034 | 73% | 1.063 | 70% |
| FF5+Mom factor | 25 assets | 1w | 1.008 | 81% | 1.030 | 73% |
| EWMA + VRA | 25 assets | 1w | 0.963 | 53% | 0.989 | 49% |
| Sample | 25 assets | 1m | 1.133 | 34% | 1.112 | 58% |
| EWMA | 25 assets | 1m | 1.085 | 28% | 1.058 | 65% |
| Ledoit-Wolf | 25 assets | 1m | 1.117 | 64% | 1.047 | 77% |
| PCA factor | 25 assets | 1m | 1.115 | 64% | 1.091 | 79% |
| FF5+Mom factor | 25 assets | 1m | 1.110 | 67% | 1.064 | 79% |
| EWMA + VRA | 25 assets | 1m | 1.060 | 39% | 1.023 | 62% |

## Log score, levels and paired differences

| Estimator | Asset set | Log score | Own 95% interval | vs. best | Paired 95% interval | Worse than best? |
| :-- | --: | --: | --: | --: | --: | --: |
| Sample | industry49 | -6.827 | [-7.08, -6.62] | 0.0754 | [0.0178, 0.1360] | yes |
| EWMA | industry49 | -6.895 | [-7.13, -6.72] | 0.0078 | [-0.0260, 0.0381] | no |
| Ledoit-Wolf | industry49 | -6.903 | [-7.12, -6.74] | 0.0000 | [0.0000, 0.0000] | no |
| PCA factor | industry49 | -6.816 | [-7.05, -6.63] | 0.0869 | [0.0613, 0.1211] | yes |
| FF5+Mom factor | industry49 | -6.873 | [-7.09, -6.70] | 0.0298 | [0.0184, 0.0439] | yes |
| EWMA + VRA | industry49 | -6.899 | [-7.12, -6.74] | 0.0035 | [-0.0172, 0.0248] | no |
| Sample | size_bm_25 | -8.301 | [-8.59, -8.08] | 0.0895 | [0.0180, 0.1632] | yes |
| EWMA | size_bm_25 | -8.388 | [-8.64, -8.20] | 0.0021 | [-0.0330, 0.0362] | no |
| Ledoit-Wolf | size_bm_25 | -8.340 | [-8.60, -8.13] | 0.0502 | [-0.0134, 0.1321] | no |
| PCA factor | size_bm_25 | -8.315 | [-8.59, -8.10] | 0.0753 | [0.0123, 0.1444] | yes |
| FF5+Mom factor | size_bm_25 | -8.322 | [-8.58, -8.12] | 0.0684 | [0.0234, 0.1120] | yes |
| EWMA + VRA | size_bm_25 | -8.390 | [-8.63, -8.22] | 0.0000 | [0.0000, 0.0000] | no |

## Stress windows (minimum-variance bias statistic, daily horizon)

| Estimator | GFC 2008-2009<br>(n=505) | COVID crash (Mar 2020)<br>(n=22) | COVID crash (wide)<br>(n=52) | 2022 drawdown<br>(n=251) |
| :-- | --: | --: | --: | --: |
| Sample | 1.837 | 6.788 | 4.922 | 1.149 |
| EWMA | 1.613 | 5.471 | 4.019 | 1.181 |
| Ledoit-Wolf | 1.550 | 5.702 | 4.173 | 1.492 |
| PCA factor | 2.098 | 6.399 | 4.761 | 2.084 |
| FF5+Mom factor | 1.983 | 5.860 | 4.481 | 2.008 |
| EWMA + VRA | 1.412 | 5.162 | 3.790 | 1.085 |

## Hyperparameters, chosen on 1972-1999 and frozen

| Family | Chosen on validation | Grid searched | Validation log score |
| :-- | --: | --: | --: |
| sample | Sample (3y window) | 2 configurations | -7.8610 |
| ewma | EWMA (half-life 252d) | 7 configurations | -7.9266 |
| ledoit_wolf | Ledoit-Wolf (1y window) | 2 configurations | -7.8450 |
| pca | PCA factor (k=5, 1y) | 12 configurations | -7.8235 |
| fundamental | FF5+Mom factor (1y, factor HL 63d) | 10 configurations | -7.8673 |
| vra | EWMA (half-life 252d) + regime adj. (HL 504d) | 7 configurations | -7.9290 |

_Run 2026-11-25T17:47:43+00:00 on arm, JAX 0.10.2 (cpu), float64=True; evaluation wall time 95.96s._
