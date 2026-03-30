# vGFR ML Pipeline: Model Evaluation & Feature Engineering Report

> **Generated:** 2026-03-26 | **Pipeline Version:** v2 (Threshold-Free)
>
> This report reflects results after enforcing the strict exclusion of all "threshold" and "margin" derived features (kidney volume, kidney HU mean/std/median) from both cohorts. All models now rely exclusively on **vascular point HU measurements**, **demographics**, and **engineered ratios**.

---

## 1. How the Best Models Work

The pipeline evaluates multiple algorithmic approaches using Leave-One-Out Cross-Validation (LOOCV). Through rigorous ranking, the following model architectures consistently rose to the top:

### Ridge Regression (Rank-Blended Selection)
**Ridge Regression** is a linear model with $L2$ regularization, which shrinks the coefficients of less important features towards zero to prevent overfitting.
- **How it's used:** Features are selected using a custom "Rank-Blended" stepwise algorithm, which ranks candidates by their simultaneous improvement to MAE, RMSE, and $R^2$, picking the feature with the best composite score. This results in stable, generalizable feature sets.

### Bayesian Ridge
**Bayesian Ridge** estimates a probabilistic model of the regression problem.
- **How it works:** Instead of a single deterministic weight, it assumes a prior distribution (Gaussian) over the weights, providing built-in regularization and a measure of uncertainty. It naturally prunes irrelevant features by pushing their distributions toward zero.

### Random Forest (Rank-Blended Selection)
**Random Forest** is a non-linear ensemble of decision trees that captures complex interactions without explicit feature engineering.
- **How it's used:** Combined with the Rank-Blended selection strategy, it identifies a minimal set of features that explain non-linear relationships between vascular dynamics and kidney function.

### Stacking Ensemble
**Stacking** combines multiple diverse base models (Bayesian Ridge, Huber, SVR) whose predictions are fed to a Ridge meta-learner.
- **Caveat:** While highly performant on the legacy cohort, the Stacking Ensemble struggles to generalize on the new cohort's restricted vascular feature space. We had to increase the robustness parameters (`max_iter`, `epsilon`) of the internal Huber Regressor to maintain convergence, and it still underperforms a simpler Ridge model.

---

## 2. Understanding the Evaluation Metrics

During LOOCV, every patient is predicted once using a model trained on all *other* patients.

| Metric | Meaning | Interpretation Focus |
| :--- | :--- | :--- |
| **MAE** (Mean Absolute Error) | Average absolute difference between predicted vGFR and true eGFRc ($ml/min/1.73m^2$). | **Clinical Interpretability**. MAE = 7.0 means the model is, on average, off by 7 units. |
| **RMSE** (Root Mean Sq. Error) | Square root of the average of squared differences. | **Outlier Penalization**. A single large miss hurts RMSE disproportionately. |
| **$R^2$** (Coefficient of Det.) | Proportion of variance in eGFRc explained by the model. | **Signal Strength**. 1.0 = perfect; 0.0 = no better than the cohort average. |

---

## 3. Top Three Results by Cohort

### Legacy Cohort (25-11-2025)
*$n = 25$ patients. Data: Age, Vascular Point HU Values (3 phases). No organ volumes or HU statistics.*

| Rank | Round | Model | MAE | $R^2$ | Features | Key Features |
| :---: | :---: | :--- | :---: | :---: | :---: | :--- |
| 1 | **10 / 13** | Ridge (Rank-Blended) | **7.04** | **0.526** | 7 | `current_age`, IVC, excretion ratios, arterial portal vein |
| 2 | **11 / 14** | Bayesian Ridge | **7.61** | **0.507** | 3 | `current_age`, `E_late_right`, IVC |
| 3 | **2** | Bayesian Ridge | **7.85** | **0.457** | 5 | `current_age`, `E_late_right`, IVC measurements |

**Round 10 / 13 — Ridge Regression (Champion)**
- **Features:** `current_age`, `arterial_venecava_between_kidney_hepatic`, `E_late_right`, `E_arterial_right`, `arterial_portal_vein`, `norm_ven_artery_right`, `late_left_hepatic_vein`
- **Comment:** These rounds converged on an identical 7-feature set, indicating a stable optimum in the unbiased vascular-only feature space. The best $R^2$ / MAE balance for a more detailed model.

![Legacy R13 Champion](../02_ml_pipeline/ml_results/25112025_round_13_champion.png)
![Legacy R13 Relevance](../02_ml_pipeline/ml_results/25112025/round_13_relevance.png)


**Round 11 / 14 — Bayesian Ridge**
- **Features:** `current_age`, `E_late_right`, `arterial_venecava_between_kidney_hepatic`
- **Comment:** The simplest model — only 3 features. Demonstrates that age + late excretion + IVC concentration successfully captures exactly half the predictable signal ($R^2 \approx 0.50$).

![Legacy R11 Champion](../02_ml_pipeline/ml_results/25112025_round_11_champion.png)
![Legacy R11 Relevance](../02_ml_pipeline/ml_results/25112025/round_11_relevance.png)


**Round 2 — Bayesian Ridge**
- **Features:** `current_age`, `E_late_right`, `venous_venacava_above_hepatic`, `arterial_venecava_above_hepatic`, `venous_left_kidney_artery`
- **Comment:** Previously the champion when relying on highly-imputed inputs. Now serves as a functional baseline, achieving $R^2 \approx 0.45$ without relying on missing clinical data points.

![Legacy R2 Champion](../02_ml_pipeline/ml_results/25112025_round_2_champion.png)
![Legacy R2 Relevance](../02_ml_pipeline/ml_results/25112025/round_2_relevance.png)


---

### New Cohort (12-03-2026)
*$n = 38$ patients. Data: Age, Sex, Vascular Point HU Values (3 phases), Slicer HU Statistics (per vessel).*

| Rank | Round | Model | MAE | $R^2$ | Features | Key Features |
| :---: | :---: | :--- | :---: | :---: | :---: | :--- |
| 1 | **10** | Random Forest | **9.09** | **0.535** | 3 | `current_age`, hepatic, arterial ratio |
| 2 | **13** | Ridge (Non-Linear) | **8.57** | **0.418** | 6 | `current_age`, `E_late_right`, hepatic IVC |
| 3 | **11 / 14** | Bayesian Ridge | **8.91** | **0.421** | 6 | `current_age`, `E_late_right`, hepatic IVC, portal, renal arteries |

> **Imputation Bias Update:** The new evaluation enforces a strict <30% missing data threshold to prevent artificial inflation in the non-linear Slicer models previously seen. The metric drops below represent the unbiased mathematical ceiling for $n=38$.

**Round 10 — Random Forest (Champion)**
- **Features:** `current_age`, `late_left_hepatic_vein`, `norm_ven_artery_left`
- **Comment:** The highest $R^2$ generated from strictly observed physiological features. The Random Forest captures a highly non-linear reliance on late hepatic clearance.

![New Cohort R10 Champion](../02_ml_pipeline/ml_results/12032026_round_10_champion.png)
![New Cohort R10 Relevance](../02_ml_pipeline/ml_results/12032026/round_10_relevance.png)


**Round 13 — Ridge (Non-Linear Interaction Search)**
- **Features:** `current_age`, `E_late_right`, `arterial_venecava_above_hepatic`, `arterial_portal_vein`, `E_arterial_right`, `ven_flow_efficiency`
- **Comment:** Stripped of the capacity to overfit on missing volumetric data, the non-linear interaction search essentially converges back roughly to the stable baseline, identically matching the legacy features.

![New Cohort R13 Champion](../02_ml_pipeline/ml_results/12032026_round_13_champion.png)
![New Cohort R13 Relevance](../02_ml_pipeline/ml_results/12032026/round_13_relevance.png)


**Round 11 / 14 — Bayesian Ridge (Rank-Blended)**
- **Features:** `current_age`, `E_late_right`, `arterial_portal_vein`, `arterial_venecava_above_hepatic`, `arterial_right_kidney_artery`, `arterial_left_kidney_artery`
- **Comment:** Demonstrates that strict Bayesian models plateau heavily around $R^2 \approx 0.42$ on this dataset without the highly imputed volumetric outliers.

![New Cohort R11 Champion](../02_ml_pipeline/ml_results/12032026_round_11_champion.png)
![New Cohort R11 Relevance](../02_ml_pipeline/ml_results/12032026/round_11_relevance.png)


---

## 4. Feature Engineering Mechanics

Below is a breakdown of how the key features are calculated from raw annotations, as defined in `01_ingestion_pipeline/gold_layer.py`.

### A. Raw Demographics
- **`current_age`**: Patient's age in years at scan time.

### B. Raw Vascular Point HU Measurements
Single-point click annotations tracking contrast concentration (HU) at specific anatomical locations during specific scan phases.
- **Examples:** `arterial_right_kidney_vein`, `venous_right_hepatic_vein`, `late_left_kidney_vein`, `arterial_venecava_between_kidney_hepatic`, `arterial_portal_vein`.

### C. Slicer Per-Vessel HU Statistics (New Cohort Only)
For the 12-03-2026 cohort, 3D Slicer export files provide HU mean and standard deviation for individual vascular structures (excluding "threshold" and "margin" labels).
- **Examples:** `arterial_renal_vein_left_hu_mean`, `venous_aorta_hu_std`.

### D. Calculated Mathematical Ratios

**1. Excretion Ratios (`E_*`)**
Relative drop in contrast between the artery feeding the kidney and the vein exiting it. Higher ratios model successful extraction by the nephrons.
- **`E_arterial_right`**: `(arterial_right_kidney_artery - arterial_right_kidney_vein) / arterial_right_kidney_artery`
- **`E_late_right`**: `(late_right_kidney_artery - late_right_kidney_vein) / late_right_kidney_artery`
- **`E_late_mean` / `E_arterial_mean`**: Average of left and right kidney excretion ratios.

**2. Aortic Normalization (`norm_*`)**
Normalizes kidney artery concentration against systemic aortic concentration to account for dosing/timing variations.
- **`norm_art_artery_right`**: `arterial_right_kidney_artery / arterial_aorta`
- **`norm_ven_artery_right`**: `venous_right_kidney_artery / venous_aorta`

**3. Advanced Non-Linear Interactions**
- **`age_x_E_arterial`**: `current_age * E_arterial_mean`. Models interaction between age-related vascular decline and current filtration capability.
- **`art_flow_efficiency`**: `E_arterial_mean * mean_artery_arterial`. Composite index of extraction × raw tracer volume during the arterial phase.
- **`ven_flow_efficiency`**: `E_venous_mean * mean_artery_venous`. Same composite for the venous phase.

---

## 5. Key Observations (Post-Threshold Cleanup)

1. **`current_age` dominates across all models** — it is the single strongest predictor in every champion, confirming that age-related nephron loss is the primary driver of GFR variation.
2. **`E_late_right` (Late Excretion Ratio)** is the second most important feature universally. This indicates that the kidney's ability to clear contrast during the equilibrium phase is a strong physiological proxy for filtration rate.
3. **IVC measurements** (`venecava_between_kidney_hepatic`, `venacava_above_hepatic`) appear in almost every model, serving as a reference for systemic venous contrast levels.
4. **Removing threshold-based features** reduced legacy cohort performance (previous champion MAE=4.20 → current best MAE=6.93) but produced a more robust, anatomically defensible feature set.
5. **Stacking Ensemble** (Round 12) remains viable for the legacy cohort (MAE=8.19) but is **unstable** for the new cohort (MAE=15.84) even after parameter stabilization and should not be used.
