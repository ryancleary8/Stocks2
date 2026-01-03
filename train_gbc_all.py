# train_gbc_all.py
import sys
from pathlib import Path

from models.gbc.gbc_model import train_gbc, TrainConfig

DATASETS_DIR = Path("data/datasets")
OUT_DIR = Path("models/gbc")


def is_classification_dataset(name: str) -> bool:
    return ".cls_" in name


def main():
    sym = sys.argv[1].strip().upper() if len(sys.argv) > 1 else None

    cfg = TrainConfig(
        train_ratio=0.80,
        tscv_splits=4,
        thr_long=0.60,
        thr_short=0.40,
        n_estimators_grid=(200, 400),
        learning_rate_grid=(0.05, 0.1),
        max_depth_grid=(3, 4),
        subsample_grid=(0.8, 1.0),
        min_samples_leaf_grid=(3, 5),
        random_state=42,
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
            task_key = ".".join(ds.name.split(".")[1:-2])
            out = OUT_DIR / symbol / task_key

            res = train_gbc(ds, out, cfg=cfg, scoring="roc_auc")
            print(f"✅ {symbol} {task_key} -> {out} | CV={res['best_cv_score']:.4f} | TEST={res['test_metrics']}")
            ok += 1
        except Exception as e:
            print(f"❌ {ds.name} failed: {e}")
            failed.append(ds.name)

    print(f"\nDone: {ok}/{len(files)} GBC models trained")
    if failed:
        print("Failed:")
        print("\n".join(failed))


if __name__ == "__main__":
    main()