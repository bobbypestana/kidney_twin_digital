"""
Plot eGFRc vs vGFR (Predicted) for v2 Pipeline.
Simplified: Loads pre-engineered features from gold.ml_features.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def plot_egfrc_vs_vgfr(y_actual, y_predicted, model_label, features_used, metrics, output_path):
    """
    Generate publication-quality eGFRc vs vGFR (Predicted) scatter plot.
    """
    fig, ax = plt.subplots(figsize=(9, 9))

    # Scatter
    sc = ax.scatter(y_predicted, y_actual, s=120, alpha=0.8,
                    edgecolors='black', linewidth=0.8, c='steelblue', zorder=5)

    # Identity line
    margin = 10
    lims = [min(y_actual.min(), y_predicted.min()) - margin,
            max(y_actual.max(), y_predicted.max()) + margin]
    ax.plot(lims, lims, 'r--', linewidth=2, alpha=0.7, label='Identity (perfect prediction)', zorder=3)

    # +/- MAE band
    mae = metrics['MAE']
    ax.fill_between(lims, [l - mae for l in lims], [l + mae for l in lims],
                    alpha=0.08, color='red', label=f'+/- MAE ({mae:.1f})', zorder=2)

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('vGFR (Predicted from CT) [mL/min/1.73m$^2$]', fontsize=13)
    ax.set_ylabel('eGFRc (Actual - Creatinine) [mL/min/1.73m$^2$]', fontsize=13)
    ax.set_title('eGFRc vs vGFR (Predicted from CT)', fontsize=15, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=11)

    # Metrics annotation box
    textstr = (f'Model: {model_label}\n'
               f'R$^2$ = {metrics["R2"]:.3f}\n'
               f'MAE = {metrics["MAE"]:.2f} \n'
               f'RMSE = {metrics["RMSE"]:.2f} \n'
               f'n = {len(y_actual)} patients\n'
               f'Features: {len(features_used)}')
    props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.97, 0.03, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {output_path}")
