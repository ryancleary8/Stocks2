# train_cnn_all.py
import sys
from pathlib import Path

from models.cnn.train_cnn import train_cnn, TrainConfig

DATASETS_DIR = Path("data/datasets")
OUT_DIR = Path("models/cnn")


def task_from_name(name: str) -> str:
    if ".reg_" in name:
        return "reg"
    if ".cls_" in name:
        return "cls"
    raise ValueError(f"Unknown dataset type: {name}")


def main():
    sym = sys.argv[1].strip().upper() if len(sys.argv) > 1 else None
    seq_len = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    cfg = TrainConfig(seq_len=seq_len, epochs=25, batch_size=128, patience=5)

    files = sorted(DATASETS_DIR.glob("*.dataset.csv"))
    if sym:
        files = [p for p in files if p.name.startswith(sym + ".")]

    ok = 0
    failed = []
    for ds in files:
        try:
            symbol = ds.name.split(".")[0]
            task_key = ".".join(ds.name.split(".")[1:-2])  # everything between SYMBOL and dataset.csv
            task = task_from_name(ds.name)

            out = OUT_DIR / symbol / task_key
            res = train_cnn(
                dataset_csv=ds,
                out_dir=out,
                task=task,
                cfg=cfg,
                model_kwargs={"conv_channels": 64, "kernel_size": 5, "dropout": 0.15},
            )
            print(f"✅ {symbol} {task_key} -> {out} | {res['test_metrics']}")
            ok += 1
        except Exception as e:
            print(f"❌ {ds.name} failed: {e}")
            failed.append(ds.name)

    print(f"\nDone: {ok}/{len(files)} CNN models trained")
    if failed:
        print("Failed:")
        print("\n".join(failed))


if __name__ == "__main__":
    main()