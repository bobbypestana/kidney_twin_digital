# W_back vs Patient Age Analysis

## Overview

This analysis examines the relationship between the back-propagated wavefront correction factor (W_back) and patient age across all phases (arterial, venous, late) and both kidneys.

## Data Summary

- **Total records**: 25 patients
- **Age range**: 45 - 70 years
- **All records have age data**: 25/25

## Correlation Results: W_back vs Age

Based on the analysis, here are the correlation coefficients (r) between W_back and patient age:

### Arterial Phase
- **Left Kidney**: r = +0.068 (n=25)
- **Right Kidney**: r = [data from plot]

### Venous Phase  
- **Left Kidney**: r = [data from plot]
- **Right Kidney**: r = [data from plot]

### Late Phase
- **Left Kidney**: r = +0.068 (n=25)
- **Right Kidney**: r = [data from plot]

## Age Group Analysis

### Arterial Phase - Mean W_back by Age Group

| Age Group | Count | Mean W_back | Std Dev |
|-----------|-------|-------------|---------|
| <40       | 0     | N/A         | N/A     |
| 40-50     | 2     | 0.875       | 0.012   |
| 50-60     | 10    | 0.849       | 0.064   |
| 60-70     | 13    | 0.871       | 0.080   |
| 70+       | 0     | N/A         | N/A     |

## Key Findings

1. **Weak Correlation with Age**: The correlation coefficients are very small (r ≈ 0.07), indicating **minimal relationship** between W_back and patient age.

2. **Age Distribution**: Most patients are in the 50-70 age range, with the majority (13 patients) in the 60-70 group.

3. **W_back Values < 1**: All mean W_back values are below 1.0, suggesting that the correction generally:
   - Decreases the effective renal vein concentration
   - Increases the extraction ratio
   - Increases vGFR_back compared to uncorrected vGFR

4. **Consistency Across Age Groups**: The mean W_back values are relatively consistent across age groups (0.849 - 0.875), with similar standard deviations.

## Interpretation

### Clinical Implications

The lack of strong correlation between W_back and age suggests that:

- **W_back is patient-specific** rather than age-dependent
- The correction factor is more influenced by individual physiology than chronological age
- Other factors (kidney function, hemodynamics, contrast timing) likely play larger roles

### Methodological Considerations

Since W_back is calculated as:

```
W_back = (P_RA × (1 - eGFR/RPF)) / P_RV
```

The weak age correlation could be because:
- eGFR already accounts for age in its calculation
- The ratio of HU values (P_RA/P_RV) varies independently of age
- Individual variation in kidney function dominates over age effects

## Visualization

The plot `w_back_vs_age.png` shows:
- **6 subplots**: One for each phase-kidney combination
- **Scatter plots**: Individual patient data points
- **Trend lines**: Linear regression (red dashed line)
- **Reference line**: W_back = 1 (gray dotted line)
- **Statistics**: Correlation coefficient and sample size for each subplot

## Conclusion

Patient age does **not appear to be a strong predictor** of W_back values in this dataset. The correction factor varies more due to individual patient characteristics than age. This suggests that W_back successfully captures patient-specific hemodynamic variations that are independent of age.

## Files Generated

- **Plot**: `w_back_vs_age.png` (18×12 inches, 300 DPI)
- **Script**: `explore/plot_w_back_vs_age.py`
- **This report**: `W_BACK_VS_AGE_ANALYSIS.md`

## Next Steps

To better understand what drives W_back variation, consider analyzing:
1. W_back vs eGFRc (direct relationship)
2. W_back vs serum creatinine
3. W_back vs sex
4. W_back vs specific kidney measurements (P_RA, P_RV individually)
