import json
from pathlib import Path

def report(folder: Path):
    if not folder.exists():
        return
    for f in folder.glob("*_summary.json"):
        with open(f) as fp:
            try:
                d = json.load(fp)
                metrics = d.get('metrics', {})
                mae = metrics.get('mae', 'N/A')
                r2 = metrics.get('r2_composite', 'N/A')
                if isinstance(mae, float): mae = f"{mae:.2f}"
                if isinstance(r2, float): r2 = f"{r2:.3f}"
                print(f"{f.stem:30s} | MAE={mae} R2={r2} | {d['model']} | Feats: {len(d['features'])}")
            except Exception as e:
                print(f"Error parsing {f.name}: {e}")

print("--- LEGACY 25-11-2025 ---")
report(Path(r'g:\My Drive\kvantify\DanQ_health\analysis\02_ml_pipeline\ml_results\25112025'))

print("\n--- NEW 12-03-2026 ---")
report(Path(r'g:\My Drive\kvantify\DanQ_health\analysis\02_ml_pipeline\ml_results\12032026'))
