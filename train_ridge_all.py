# train_ridge_all.py
import sys
from pathlib import Path

from models.ridge.ridge_regressor import train_ridge_regressor, TrainConfig

DATASETS_DIR = Path("data/datasets")
OUT_DIR = Path("models/ridge")


def is_regression_dataset(name: str) -> bool:
    return ".reg_" in name  # your convention


def main():
    sym = sys.argv[1].strip().upper() if len(sys.argv) > 1 else None

    cfg = TrainConfig(
        train_ratio=0.80,
        tscv_splits=5,
        alpha_grid=(0.1, 1.0, 10.0, 100.0),
        with_scaler=True,
        baseline_zero=True,
    )

    files = sorted(DATASETS_DIR.glob("*.dataset.csv"))
    files = [p for p in files if is_regression_dataset(p.name)]
    if sym:
        files = [p for p in files if p.name.startswith(sym + ".")]

    if not files:
        print("No regression datasets found (expected *.reg_*.dataset.csv).")
        return

    ok = 0
    failed = []
    for ds in files:
        try:
            symbol = ds.name.split(".")[0]
            task_key = ".".join(ds.name.split(".")[1:-2])
            out = OUT_DIR / symbol / task_key

            res = train_ridge_regressor(ds, out, cfg=cfg)
            print(f"✅ {symbol} {task_key} -> {out} | TEST={res['test_metrics']} | BEST={res['best_params']}")
            ok += 1
        except Exception as e:
            print(f"❌ {ds.name} failed: {e}")
            failed.append(ds.name)

    print(f"\nDone: {ok}/{len(files)} Ridge models trained")
    if failed:
        print("Failed:")
        print("\n".join(failed))


if __name__ == "__main__":
    main()