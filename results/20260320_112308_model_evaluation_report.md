# vGFR ML Pipeline: Model Evaluation & Feature Engineering Report

This document details the machine learning approaches, evaluation metrics, top-performing models for each cohort, and the underlying feature engineering mechanics used to predict virtual Glomerular Filtration Rate (vGFR).

---

## 1. How the Best Models Work

The pipeline evaluates multiple algorithmic approaches using Leave-One-Out Cross-Validation (LOOCV). Through rigorous ranking, three model architectures consistently rose to the top:

### Huber Regressor (Rank-Blended Selection)
The **Huber Regressor** is a linear model that is highly **robust to outliers**. Unlike standard Ordinary Least Squares (OLS) which heavily penalizes large errors via squared loss, Huber applies absolute loss to errors above a certain threshold. 
- **How it's used:** We feed it into a custom "Rank-Blended" stepwise selection algorithm. Instead of just picking features that boost $R^2$, the selector ranks all potential features by how much they improve MAE, RMSE, and $R^2$ simultaneously, picking the feature with the best average rank. This results in incredibly stable, generalizable feature sets that aren't overly sensitive to a single noisy patient.

### Stacking Ensemble
**Stacking** is an ensemble learning technique that combines multiple diverse algorithms.
- **How it works:** We train three distinct "base" models on the data: a Bayesian Ridge (probabilistic linear), a Huber Regressor (outlier-robust linear), and a Support Vector Regressor (SVR, non-linear margin-based). Their predictions are then passed to a "meta-learner" (a Ridge Regressor) alongside the original features. The meta-learner essentially learns *when to trust which base model*, combining their strengths.

### Ridge Regression (Non-Linear Interactions)
**Ridge Regression** is a linear model with $L2$ regularization, which shrinks the coefficients of less important features towards zero to prevent overfitting.
- **How it's used:** In Round 13, we expose the Ridge model to a highly engineered dataset containing "non-linear interactions" (such as `age * excretion_rate`). Because Ridge naturally handles multicollinearity (highly correlated features), it safely navigates the overlapping interaction terms to find strong signals without destabilizing the coefficients.

### Bayesian Ridge
**Bayesian Ridge** estimates a probabilistic model of the regression problem.
- **How it works:** Instead of predicting a single deterministic weight for a feature, it assumes a prior distribution (usually Gaussian) over the weights, providing built-in regularization and a measure of uncertainty. It naturally prunes irrelevant features by pushing their distributions toward zero.

---

## 2. Understanding the Evaluation Metrics

During LOOCV (Leave-One-Out Cross-Validation), every patient is predicted exactly once using a model trained on all *other* patients. The resulting predictions are compared against the true eGFRc using three key metrics:

| Metric | Meaning | Interpretation focus |
| :--- | :--- | :--- |
| **MAE** (Mean Absolute Error) | The simple average of the absolute difference between predicted vGFR and true eGFRc in $ml/min/1.73m^2$. | **Clinical Interpretability**. An MAE of 5.0 means the model is, on average, off by exactly 5 units. Focusing on MAE ensures the model is practically accurate for the *majority* of normal patients. |
| **RMSE** (Root Mean Sq. Error) | The square root of the average of squared differences. | **Outlier Penalization**. Because errors are squared before averaging, a single prediction that is off by 20 units hurts RMSE much more than four predictions off by 5 units. Focusing on RMSE forces the model to avoid "catastrophic" misses. |
| **$R^2$** (Coefficient of Det.) | The proportion of variance in the true eGFRc that is explained by the model's predictions. | **Signal Strength**. An $R^2$ of 1.0 is perfect; 0.0 means the model is no better than just guessing the average eGFRc of the cohort. Positive $R^2$ proves the model has found a real, mathematical signal. |

---

## 3. Top Three Results by Cohort

### Legacy Cohort (25-11-2025)
*Data availability: Age, Sex, Organ Segment Volumes, HU Statistics, Vascular Point Values.*

1. **Huber Regressor (Rank-Blended) — Round 10**
   - **MAE:** 4.20
   - **$R^2$:** 0.713
   - **Features:** `current_age`, `venous_kidney_hu_std`, `arterial_right_kidney_vein`, `venous_right_hepatic_vein`, `norm_art_artery_right`, `arterial_kidney_hu_mean`, `late_left_kidney_vein`, `arterial_venecava_between_kidney_hepatic`, `venous_kidney_hu_mean`, `late_portal_vein`
   - **Comment:** The absolute champion. By heavily prioritizing image texture (HU Standard Deviation) and vascular flow, the robust Huber model achieved an incredible 4.2 MAE. It is highly resistant to noise.
   - ![Legacy R10 Champion](figures/25112025_round_10_champion.png)

2. **Stacking Ensemble — Round 12**
   - **MAE:** 6.47
   - **$R^2$:** 0.695
   - **Features:** `current_age`, `arterial_venecava_between_kidney_hepatic`, `E_late_right`, `venous_kidney_hu_std`, `E_arterial_right`, `venous_kidney_vol`, `late_right_kidney_vein`, `vol_per_age`, `arterial_portal_vein`, `age_x_E_arterial`, `E_arterial_mean`, `norm_ven_artery_right`
   - **Comment:** Leverages the interaction terms found in Round 13. While the MAE is higher than Huber, the $R^2$ is phenomenal, indicating excellent proportional prediction across the entire spectrum of kidney function.
   - ![Legacy R12 Champion](figures/25112025_round_12_champion.png)

3. **Ridge Regression (Non-Linear) — Round 13**
   - **MAE:** 5.36
   - **$R^2$:** 0.685
   - **Features:** `current_age`, `arterial_venecava_between_kidney_hepatic`, `E_late_right`, `venous_kidney_hu_std`, `E_arterial_right`, `venous_kidney_vol`, `late_right_kidney_vein`, `vol_per_age`, `arterial_portal_vein`, `age_x_E_arterial`, `E_arterial_mean`, `norm_ven_artery_right` (same as Stacking)
   - **Comment:** A simpler, more interpretable alternative to the Stacking Ensemble using the exact same interaction features, bringing the MAE back down toward 5.0.
   - ![Legacy R13 Champion](figures/25112025_round_13_champion.png)

### New Cohort (12-03-2026)
*Data availability: Age, Sex, Vascular Point Values.* **(No segment volumes or HU statistics available)**.

1. **Random Forest (Rank-Blended) — Round 10**
   - **MAE:** 8.52
   - **$R^2$:** 0.534
   - **Features:** `current_age`, `late_left_hepatic_vein`, `art_flow_efficiency`
   - **Comment:** The highest $R^2$ was achieved by a non-linear tree-based ensemble. Relying on just 3 features, it successfully finds an interaction pattern between age and arterial flow efficiency.
   - ![New Cohort R10 Champion](figures/12032026_round_10_champion.png)

2. **BayesianRidge (Rank-Blended) — Round 11**
   - **MAE:** 8.91
   - **$R^2$:** 0.421
   - **Features:** `current_age`, `E_late_right`, `arterial_portal_vein`, `arterial_venecava_above_hepatic`, `arterial_right_kidney_artery`, `arterial_left_kidney_artery`
   - **Comment:** Bayesian parameter estimation proves highly stable when working with restricted feature sets, performing almost identically to the Huber Regressor.
   - ![New Cohort R11 Champion](figures/12032026_round_11_champion.png)

3. **Ridge Regression (Non-Linear) — Round 13**
   - **MAE:** 8.57
   - **$R^2$:** 0.418
   - **Features:** `current_age`, `E_late_right`, `arterial_venecava_above_hepatic`, `arterial_portal_vein`, `E_arterial_right`, `ven_flow_efficiency`
   - **Comment:** Adding non-linear vascular-efficiency interactions allowed a standard Ridge model to edge out slightly better absolute error (MAE) than the Bayesian approach, though explaining slightly less overall variance ($R^2$).
   - ![New Cohort R13 Champion](figures/12032026_round_13_champion.png)

---

## 4. Feature Engineering Mechanics

Below is a detailed breakdown of exactly how the key features used in the champion models are calculated from the raw annotations, as defined in `01_ingestion_pipeline/gold_layer.py`.

### A. Raw Demographics & Morphology
- **`current_age`**: The patient's age in years at the time the scan was taken.
- **`venous_kidney_vol`**: The total continuous segmented volume ($cm^3$) of the kidney during the venous phase.
- **`*_kidney_hu_std` / `hu_mean`**: The standard deviation and mean of Hounsfield Units within the entire 3D kidney mask during the specified phase (arterial, venous, late). Captures whole-organ parenchymal texture.

### B. Raw Vascular Point HU Measurements
The raw data contains single-point click annotations tracking radiotracer concentration (in HU) at specific anatomical locations during specific scan phases.
- **Examples:** `arterial_right_kidney_vein`, `venous_right_hepatic_vein`, `late_left_kidney_vein`, `arterial_venecava_between_kidney_hepatic`, `arterial_portal_vein`.

### C. Calculated Mathematical Ratios
To capture *flow* and *clearance kinetics* rather than just static brightness, several complex ratios are computed during the `gold` data synthesis:

**1. Excretion Ratios (`E_*`)**
Calculates the relative drop in contrast concentration between the artery feeding the kidney and the vein exiting it. Higher ratios mathematically model successful extraction by the nephrons.
- **`E_arterial_right`**: `(arterial_right_kidney_artery - arterial_right_kidney_vein) / arterial_right_kidney_artery`
- **`E_late_right`**: `(late_right_kidney_artery - late_right_kidney_vein) / late_right_kidney_artery`
- **`E_late_mean` / `E_arterial_mean`**: The simple average of the left and right kidney excretion ratios.

**2. Aortic Normalization (`norm_*`)**
Normalizes the tracer concentration arriving at the kidney against the global systemic concentration in the aorta to account for dosing/timing variations.
- **`norm_art_artery_right`**: `arterial_right_kidney_artery / arterial_aorta`
- **`norm_ven_artery_right`**: `venous_right_kidney_artery / venous_aorta`

**3. Advanced Non-Linear Interactions**
Engineered specifically for Round 13/14 to capture complex physiological interplay:
- **`vol_per_age`**: `arterial_kidney_vol / current_age`. A proxy for age-related parenchymal atrophy.
- **`age_x_E_arterial`**: `current_age * E_arterial_mean`. Models the interactions between declining vascular elasticity (age) and current filtration capability.
- **`ven_flow_efficiency`**: `E_venous_mean * mean_artery_venous`. A composite index: the average extraction factor multiplied by the raw volume of tracer arriving via the artery during the venous phase.
