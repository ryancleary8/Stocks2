# train_lstm_all.py
import sys
from pathlib import Path

from models.rnn.train_lstm import train_lstm, TrainConfig

DATASETS_DIR = Path("data/datasets")
OUT_DIR = Path("models/lstm")


def task_from_name(name: str) -> str:
    if ".reg_" in name:
        return "reg"
    if ".cls_" in name:
        return "cls"
    raise ValueError(f"Unknown dataset type: {name}")


def main():
    sym = sys.argv[1].strip().upper() if len(sys.argv) > 1 else None
    seq_len = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    cfg = TrainConfig(seq_len=seq_len, epochs=40, batch_size=128, patience=6)

    files = sorted(DATASETS_DIR.glob("*.dataset.csv"))
    if sym:
        files = [p for p in files if p.name.startswith(sym + ".")]

    ok = 0
    failed = []
    for ds in files:
        try:
            symbol = ds.name.split(".")[0]
            task_key = ".".join(ds.name.split(".")[1:-2])
            task = task_from_name(ds.name)

            out = OUT_DIR / symbol / task_key
            res = train_lstm(
                dataset_csv=ds,
                out_dir=out,
                task=task,
                cfg=cfg,
                model_kwargs={"hidden_dim": 64, "num_layers": 1, "dropout": 0.25},
            )
            print(f"✅ {symbol} {task_key} -> {out} | {res['test_metrics']}")
            ok += 1
        except Exception as e:
            print(f"❌ {ds.name} failed: {e}")
            failed.append(ds.name)

    print(f"\nDone: {ok}/{len(files)} LSTM models trained")
    if failed:
        print("Failed:")
        print("\n".join(failed))


if __name__ == "__main__":
    main()