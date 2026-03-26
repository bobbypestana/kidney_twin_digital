# Performance Improvement Summary: vGFR ML Pipeline

This report summarizes how iterative improvements to the modeling code—such as robust outlier handling and non-linear interaction terms—have refined vGFR predictions across two distinct cohorts.

## New (Vascular Only)

| Round | Model | MAE | R2 | Story |
| --- | --- | --- | --- | --- |
| 10 | RandomForest | 8.95 | 0.521 | Round 10 for the new cohort used Random Forest to capture non-linearities in vascular-only data. |
| 11 | BayesianRidge | 8.91 | 0.421 | Round 11 leveraged Bayesian priors for stable coefficient estimation. |
| 13 | Ridge | 8.57 | 0.418 | Round 13 utilized non-linear interaction terms to recover performance on the new cohort's vascular data. |

**Key Finding:** Performance improved by **4.2%** in MAE from the initial baseline to the current champion model.

## Legacy (Morphology+Vascular)

| Round | Model | MAE | R2 | Story |
| --- | --- | --- | --- | --- |
| 3 | Ridge | 9.71 | 0.289 | Round 3 focused on traditional kidney morphology (HU and Volume) to establish a baseline. |
| 10 | Huber | 4.20 | 0.713 | Round 10 introduced the Huber Regressor specifically to handle outliers, achieving the best MAE results. |
| 13 | Ridge | 5.36 | 0.685 | Round 13 utilized non-linear interaction terms (e.g., Age x Flow) to capture complex physiological interplay. |

**Key Finding:** Performance improved by **56.7%** in MAE from the initial baseline to the current champion model.

