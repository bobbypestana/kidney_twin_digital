import json
from pathlib import Path

def report(folder: Path):
    if not folder.exists():
        print(f"Directory {folder} does not exist.")
        return
    for f in folder.glob("*_summary.json"):
        with open(f) as fp:
            d = json.load(fp)
            mae = d.get('metrics', {}).get('mae')
            if mae is not None:
                r2 = d.get('metrics', {}).get('r2_composite')
                print(f"{f.stem:25s} | MAE={mae:.2f} R2={r2:.3f} | {d['model']} | Feats: {len(d['features'])}")

print("--- LEGACY 25-11-2025 ---")
report(Path(r'g:\My Drive\kvantify\DanQ_health\analysis\02_ml_pipeline\ml_results\25112025'))

print("\n--- NEW 12-03-2026 ---")
report(Path(r'g:\My Drive\kvantify\DanQ_health\analysis\02_ml_pipeline\ml_results\12032026'))
