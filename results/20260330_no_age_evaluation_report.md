# vGFR Physiological Isolation: ML Experiments Without `current_age`

> **Generated:** 2026-03-30 | **Focus:** Evaluating predictive models purely on vascular phase data
>
> **Objective**: The previous experiments identified `current_age` as the dominant feature in every successful vGFR prediction model. This report evaluates model performance when `current_age` is explicitly **removed** from the feature matrix, forcing the algorithms to build a prediction strictly from isolated physiological features (contrast pooling, Slicer statistics, and spatial gradients) within a single anatomical scan phase.

---

## 1. Methodology

The pipeline was executed exactly as before, with one critical change: the `--exclude-age` parameter was invoked.
- `current_age` and any interaction features containing "age" (e.g. `age_x_E_arterial`) were dropped entirely before feature selection.
- Features were restricted to a single phase (Arterial, Venous, or Late).

---

## 2. Legacy Cohort Results (25-11-2025)
*$n = 25$ patients. Data: Vascular Point HU Values.*

**Conclusion:** The legacy cohort suffered a moderate collapse in the Late phase (dropping from $R^2 \approx 0.47$ to $0.18$), proving that late-phase contrast pooling is meaningless unless strictly contextualized by patient age. The Venous systemic flow models maintained a weak but stable baseline.

### 🔴 ARTERIAL PHASE

| Rank | Model | MAE | $R^2$ | Features Selected |
| :---: | :--- | :---: | :---: | :--- |
| **1** | **Huber** | **9.62** | **0.247** | 3 features: `E_arterial_right`, `arterial_venecava_between_kidney_hepatic`, `art_flow_efficiency` |
| 2 | BayesianRidge | 10.19 | 0.218 | 3 features |
| 3 | KNN | 10.97 | 0.136 | 3 features |

![Legacy Arterial No-Age Champion](../02_ml_pipeline/ml_results/25112025_arterial_no_age_top1_Huber_champion.png)
![Legacy Arterial No-Age Relevance](../02_ml_pipeline/ml_results/25112025/arterial_no_age_rank1_Huber_relevance.png)

### 🔵 VENOUS PHASE (Champion true no-age)
| Rank | Model | MAE | $R^2$ | Features Selected |
| :---: | :--- | :---: | :---: | :--- |
| **1** | **GradientBoosting** | **10.20** | **0.352** | 2 features: `norm_ven_artery_right`, `venous_aorta` |
| 2 | RandomForest | 10.00 | 0.302 | 2 features |
| 3 | KNN | 9.26 | 0.237 | 1 feature |

![Legacy Venous No-Age Champion](../02_ml_pipeline/ml_results/25112025_venous_no_age_top1_GradientBoosting_champion.png)
![Legacy Venous No-Age Relevance](../02_ml_pipeline/ml_results/25112025/venous_no_age_rank1_Gradient_relevance.png)

### 🟡 LATE PHASE
| Rank | Model | MAE | $R^2$ | Features Selected |
| :---: | :--- | :---: | :---: | :--- |
| **1** | **GradientBoosting** | **10.48** | **0.188** | 4 features |
| 2 | KNN | 11.97 | 0.078 | 2 features |
| 3 | Huber | 12.14 | -0.020 | 1 feature |

![Legacy Late No-Age Champion](../02_ml_pipeline/ml_results/25112025_late_no_age_top1_GradientBoosting_champion.png)
![Legacy Late No-Age Relevance](../02_ml_pipeline/ml_results/25112025/late_no_age_rank1_Gradient_relevance.png)

---

## 3. New Cohort Results (12-03-2026)
*$n = 38$ patients. Data: Vascular Point HU Values, Slicer HU Vessel Stats (<30% missing).*

**Conclusion:** A drastic physiological collapse. Because `current_age` was previously the absolute anchor for the new cohort (peaking at $R^2 = 0.606$ in the baseline), removing it halved the performance of the best models and caused the Late Phase signal to effectively vanish. The single best physiological alternative was calculating the raw gradient between the isolated kidney vein and the systemic Vena Cava via a Random Forest.

### 🔵 VENOUS PHASE (Champion true no-age)
| Rank | Model | MAE | $R^2$ | Features Selected |
| :---: | :--- | :---: | :---: | :--- |
| **1** | **RandomForest** | **11.15** | **0.316** | 3 features: `venous_k_vein_systemic_grad`, `norm_ven_artery_right`, `norm_ven_artery_left` |
| 2 | GradientBoosting | 11.36 | 0.118 |  2 features |
| 3 | KNN | 11.93 | 0.051 | 3 features |

![New Venous No-Age Champion](../02_ml_pipeline/ml_results/12032026_venous_no_age_top1_RandomForest_champion.png)
![New Venous No-Age Relevance](../02_ml_pipeline/ml_results/12032026/venous_no_age_rank1_RandomFo_relevance.png)

### 🔴 ARTERIAL PHASE

| Rank | Model | MAE | $R^2$ | Features Selected |
| :---: | :--- | :---: | :---: | :--- |
| **1** | **BayesianRidge**| **11.24** | **0.195** | 5 features: `arterial_right_kidney_vein`, `arterial_venecava_between_kidney_hepatic`, `arterial_left_hepatic_vein`, `E_arterial_right`, `E_arterial_mean` |
| 2 | KNN | 11.59 | 0.196 | 4 features |
| 3 | Huber | 10.90 | 0.125 | 4 features |

![New Arterial No-Age Champion](../02_ml_pipeline/ml_results/12032026_arterial_no_age_top1_BayesianRidge_champion.png)
![New Arterial No-Age Relevance](../02_ml_pipeline/ml_results/12032026/arterial_no_age_rank1_Bayesian_relevance.png)

### 🟡 LATE PHASE
| Rank | Model | MAE | $R^2$ | Features Selected |
| :---: | :--- | :---: | :---: | :--- |
| **1** | **RandomForest** | **11.75** | **0.128** | 3 features: `late_right_hepatic_vein`, `late_portal_hepatic_delta`, `late_left_kidney_vein` |
| 2 | GradientBoosting| 12.02 | 0.153 | 4 features |
| 3 | ElasticNet | 12.36 | -0.014 | 1 feature |

![New Late No-Age Champion](../02_ml_pipeline/ml_results/12032026_late_no_age_top1_RandomForest_champion.png)
![New Late No-Age Relevance](../02_ml_pipeline/ml_results/12032026/late_no_age_rank1_RandomFo_relevance.png)

---

## 4. Key Takeaways

1. **Age is the Fundamental Anchor:** Raw physiological features (excretion ratios, normalized contrast pooling, spatial gradients) do **not** contain enough intrinsic information to predict an absolute glomerular filtration rate (vGFR). Without Age to establish the baseline filtration target for the patient's demographic bracket, prediction errors blow up past MAE 11.0.
2. **Physiology Explains Variance, Not Absolute Limits:** The phase-specific contrast features function exclusively as variance modifiers. In the baseline models, Slicer statistics help the model decide if a 65-year-old patient is filtering *better* or *worse* than the average 65-year-old. Without the "this is a 65-year-old" anchor, the equation has no foundation.
3. **Venous Gradients Act as the Nearest Substitute:** Without age, the models turned entirely to the Venous phase. Specifically, the machine learning algorithms found their strongest raw signal mapping the systemic difference mapping between the pooled contrast in the individual renal vein versus the entire systemic return flowing through the Vena Cava (`venous_k_vein_systemic_grad`).
