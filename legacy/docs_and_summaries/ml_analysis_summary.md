# vGFR ML Analysis Summary

This document summarizes the current state of the machine learning analysis aimed at predicting eGFRc (kidney function) using CT scan segmentation features. This summary is intended to be passed to future Antigravity sessions to maintain continuity.

## 1. Project Goal & Data Architecture
- **Objective:** Calibrate a kidney digital twin per patient by predicting `egfrc` (estimated Glomerular Filtration Rate from serum creatinine).
- **Data Source:** DuckDB (`database/egfr_data.duckdb`).
- **Core Tables:** The primary training dataset is `gold.anon_segmentations_with_egfr`, which contains patient info, calculated eGFRc, and segmentation metrics (HU stats for arterial, venous, and late phases). `bronze.*` tables contain raw volume and HU distributions.

## 2. Feature Engineering Strategies
The initial dataset involved extensive feature engineering derived from raw HU measurements:
- **Extraction ratios:** [(artery_HU - vein_HU) / artery_HU](file:///G:/My%20Drive/kvantify/DanQ_health/analysis/classical_ml/vgfr_improvements_r10.py#257-341) per phase.
- **Left-right symmetry:** Ratios of left vs. right kidney densities.
- **Phase contrast:** Differences between arterial and late phases.
- **Normalization:** Kidney densities normalized against the aorta.
- **Bilateral means:** Averages across both kidneys to reflect overall function.

*Note: All features related to calculated vGFR (`vgfr_`), leakage, or specific concentrations (`conc_`, `w_pv_`, `w_back_`) were explicitly excluded from ML training to avoid data leakage.*

## 3. Modeling Approach & Evolution (Rounds 1-10)
We performed 10 major iterations of ML improvements (`classical_ml/vgfr_improvements_r*.py`). These involved:

1. **Algorithm Testing:** 
   Evaluated multiple standard algorithms using LOOCV (Leave-One-Out Cross-Validation), including Ridge, Lasso, ElasticNet, Huber, BayesianRidge, RandomForest, and SVR.
   
2. **Feature Selection Tactics:**
   - Started with simple correlation-based filtering.
   - Progressed to **"Blended Rank Stepwise Selection"**: A greedy, forward-stepwise feature selector designed to optimize a composite score of MAE, RMSE, and R2 simultaneously.

3. **Inclusion of Bronze Features (Round 10):**
   In the final rounds, we expanded the feature set to include raw `Volume` and `HU distribution` data (Mean, Std, Median, Percentiles) directly from the [bronze](file:///G:/My%20Drive/kvantify/DanQ_health/analysis/classical_ml/vgfr_improvements_r10.py#55-90) segmentation tables, vastly increasing the candidate feature pool.

4. **Tracking & Reproducibility:**
   - **MLflow** was heavily utilized to track run hashes, parameters, JSON feature importances, residuals, and LOOCV performance graphs. Logs reside in `C:/tmp/mlflow_vgfr`.

## 4. Final Champions
As detailed in `plot_final_champions.py`, the final top-performing configurations were:

### 🏆 1. BayesianRidge (Highest R² and Lowest RMSE)
- **R² Score:** Highest overall variance explained.
- **Key Features (11 total):** `current_age`, `arterial_venecava_between_kidney_hepatic`, `E_late_right`, `norm_vein_arterial_right`, `arterial_portal_vein`, `LR_ratio_vein_venous`, etc.
- **Color profile:** Steelblue.

### 🎯 2. SVR (Highest Precision / Lowest MAE)
- **Model:** SVR(kernel='rbf', C=10.0)
- **Key Features (4 total):** `current_age`, `arterial_venacava_below_kidney`, `arterial_right_kidney_vein`, `norm_vein_late_right`.
- **Strength:** Excellent precision resulting in the lowest Mean Absolute Error (MAE).
- **Color profile:** Indigo.

### ⚖️ 3. Ridge (Best Balanced Trade-off)
- **Model:** Ridge(alpha=10.0)
- **Key Features (7 total):** `current_age`, `arterial_venecava_between_kidney_hepatic`, `E_late_right`, `E_arterial_right`, `arterial_portal_vein`, `norm_artery_venous_right`, `LR_ratio_vein_venous`.
- **Strength:** Balanced metrics across MAE, RMSE, and R2.
- **Color profile:** Forestgreen.

## Next Steps for New Agent
- Always work out of `G:\My Drive\kvantify\DanQ_health\analysis`.
- Review the `ml_results/` models and charts if new analysis parameters are required.
- Continue tracking any newly designed iterations with the existing MLflow pipeline set up in the `.py` scripts.
