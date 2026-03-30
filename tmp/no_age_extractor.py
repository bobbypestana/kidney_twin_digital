import json
from pathlib import Path

def report(folder: Path, out_path: Path):
    if not folder.exists():
        return
    results = []
    for f in folder.glob("*_no_age_*_summary.json"):
        with open(f) as fp:
            try:
                d = json.load(fp)
                metrics = d.get('metrics', {})
                mae = metrics.get('MAE', 'N/A')
                r2 = metrics.get('R2', 'N/A')
                if isinstance(mae, float): mae = f"{mae:.2f}"
                if isinstance(r2, float): r2 = f"{r2:.3f}"
                features = d['features']
                results.append((d['round'], d['model'], mae, r2, len(features), features))
            except Exception as e:
                pass
    
    with open(out_path, 'a') as f_out:
        for p in ['arterial', 'venous', 'late']:
            f_out.write(f"\nPhase: {p.upper()}\n")
            for r in sorted(results, key=lambda x: str(x[0])):
                if p in str(r[0]):
                    name = r[0].replace(f'{p}_no_age_rank', 'Rank ')
                    f_out.write(f"  {name:15s} | {r[1]:15s} | MAE: {r[2]} | R2: {r[3]} | Feats ({r[4]}): {r[5]}\n")

out_file = Path('g:/My Drive/kvantify/DanQ_health/analysis/tmp/no_age_results.txt')
if out_file.exists(): out_file.unlink()

with open(out_file, 'w') as f:
    f.write("====================================\n--- LEGACY COHORT (25-11-2025) ---\n====================================\n")
report(Path(r'g:\My Drive\kvantify\DanQ_health\analysis\02_ml_pipeline\ml_results\25112025'), out_file)

with open(out_file, 'a') as f:
    f.write("\n\n====================================\n--- NEW COHORT (12-03-2026) ---\n====================================\n")
report(Path(r'g:\My Drive\kvantify\DanQ_health\analysis\02_ml_pipeline\ml_results\12032026'), out_file)
