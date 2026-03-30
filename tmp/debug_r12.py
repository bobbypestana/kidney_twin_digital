"""Diagnose Round 12 Stacking failure on 12-03-2026 cohort."""
import sys
sys.path.append(r'g:\My Drive\kvantify\DanQ_health\analysis\02_ml_pipeline')

import numpy as np
import pandas as pd
from ml_utils import load_cohort, get_feature_matrix, TARGET
from sklearn.linear_model import Ridge, HuberRegressor, BayesianRidge
from sklearn.ensemble import StackingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

df, _ = load_cohort("12-03-2026")
X, y = get_feature_matrix(df)

# R13 features (from the JSON)
r13_feats = ["current_age", "E_late_right", "arterial_portal_vein", "arterial_renal_vein_left_hu_mean"]

print(f"Cohort size: {len(df)}")
print(f"R13 features: {r13_feats}")
print(f"Feature matrix shape for stacking: {X[r13_feats].shape}")
print()

# Test individual base models first
for name, model in [("Bayesian", BayesianRidge()), ("Huber", HuberRegressor()), ("SVR", SVR(kernel='rbf', C=10.0))]:
    pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
    y_pred = cross_val_predict(pipe, X[r13_feats].values, y.values, cv=LeaveOneOut(), n_jobs=-1)
    print(f"  {name}: MAE={mean_absolute_error(y, y_pred):.2f}, R2={r2_score(y, y_pred):.3f}")

print()

# Stacking with passthrough=True (default)
print("--- Stacking (default passthrough=True) ---")
estimators = [
    ('bayesian', BayesianRidge()),
    ('huber', HuberRegressor()),
    ('svr', SVR(kernel='rbf', C=10.0))
]
stack = StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=1.0))
pipe = Pipeline([('scaler', StandardScaler()), ('model', stack)])
y_pred = cross_val_predict(pipe, X[r13_feats].values, y.values, cv=LeaveOneOut(), n_jobs=-1)
print(f"  MAE={mean_absolute_error(y, y_pred):.2f}, R2={r2_score(y, y_pred):.3f}")
print(f"  y_pred range: [{y_pred.min():.1f}, {y_pred.max():.1f}]")
print(f"  y_true range: [{y.min():.1f}, {y.max():.1f}]")

print()

# Stacking with passthrough=False
print("--- Stacking (passthrough=False) ---")
stack2 = StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=1.0), passthrough=False)
pipe2 = Pipeline([('scaler', StandardScaler()), ('model', stack2)])
y_pred2 = cross_val_predict(pipe2, X[r13_feats].values, y.values, cv=LeaveOneOut(), n_jobs=-1)
print(f"  MAE={mean_absolute_error(y, y_pred2):.2f}, R2={r2_score(y, y_pred2):.3f}")
print(f"  y_pred range: [{y_pred2.min():.1f}, {y_pred2.max():.1f}]")
