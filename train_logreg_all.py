# train_logreg_all.py
import sys
from pathlib import Path

from models.logreg.logreg_model import train_logreg, TrainConfig

DATASETS_DIR = Path("data/datasets")
OUT_DIR = Path("models/logreg")


def is_classification_dataset(name: str) -> bool:
    return ".cls_" in name


def main():
    sym = sys.argv[1].strip().upper() if len(sys.argv) > 1 else None

    cfg = TrainConfig(
        train_ratio=0.80,
        C_grid=(0.01, 0.05, 0.1, 0.5, 1.0),
        tscv_splits=5,
        class_weight="balanced",
        max_iter=2000,
        proba_threshold=0.50,
    )

    files = sorted(DATASETS_DIR.glob("*.dataset.csv"))
    files = [p for p in files if is_classification_dataset(p.name)]
    if sym:
        files = [p for p in files if p.name.startswith(sym + ".")]

    if not files:
        print("No classification datasets found (expected *.cls_*.dataset.csv).")
        return

    ok = 0
    failed = []
    for ds in files:
        try:
            symbol = ds.name.split(".")[0]
            task_key = ".".join(ds.name.split(".")[1:-2])  # between SYMBOL and dataset.csv
            out = OUT_DIR / symbol / task_key
            res = train_logreg(ds, out, cfg=cfg)
            print(f"✅ {symbol} {task_key} -> {out} | {res['test_metrics']}")
            ok += 1
        except Exception as e:
            print(f"❌ {ds.name} failed: {e}")
            failed.append(ds.name)

    print(f"\nDone: {ok}/{len(files)} logreg models trained")
    if failed:
        print("Failed:")
        print("\n".join(failed))


if __name__ == "__main__":
    main()