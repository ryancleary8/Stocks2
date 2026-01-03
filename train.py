"""train.py

Unified training entrypoint.

This script loops through each stock symbol listed in ``stocks.txt`` and trains
all available model architectures using pre-built datasets in ``data/datasets``.

- Datasets are assumed to already exist. If a symbol has no datasets, it is skipped.
- Outputs are written to ``trained_models/{model}/{symbol}/{task_key}/``.

Supported model families (when present in your repo):
  - Logistic Regression (classification)
  - Ridge Regression (regression)
  - Gradient Boosting Classifier (classification)
  - Random Forest (classification + regression)
  - XGBoost (classification + regression)
  - CNN (classification + regression)
  - LSTM/RNN (classification + regression)
  - Transformer (classification + regression)
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from datasets import load_symbols


def _optional_import(module_path: str, *names: str) -> list[Any]:
    """Import attributes defensively, returning ``None`` when missing.

    This keeps ``train.py`` runnable even when optional model dependencies or
    ignored directories (e.g., ``models/``) are absent. A friendly notice is
    printed once for each missing module.
    """

    try:
        module = importlib.import_module(module_path)
    except Exception as exc:  # noqa: BLE001
        missing = ", ".join(names)
        print(f"ℹ️  Skipping unavailable module '{module_path}' ({missing}): {exc}")
        return [None for _ in names]

    resolved = []
    for name in names:
        try:
            resolved.append(getattr(module, name))
        except AttributeError:
            print(f"ℹ️  Module '{module_path}' missing attribute '{name}', skipping.")
            resolved.append(None)
    return resolved


# -----------------------------
# Model trainers (imported lazily so the script can run even without them)
# -----------------------------
CnnTrainConfig, train_cnn = _optional_import("models.cnn.train_cnn", "TrainConfig", "train_cnn")
GbcTrainConfig, train_gbc = _optional_import("models.gbc.gbc_model", "TrainConfig", "train_gbc")
LogRegTrainConfig, train_logreg = _optional_import("models.logreg.logreg_model", "TrainConfig", "train_logreg")
RidgeTrainConfig, train_ridge_regressor = _optional_import(
    "models.ridge.ridge_regressor", "TrainConfig", "train_ridge_regressor"
)
LstmTrainConfig, train_lstm = _optional_import("models.rnn.train_lstm", "TrainConfig", "train_lstm")

RfClsTrainConfig, train_rf_classifier = _optional_import(
    "models.rf.rf_classifier", "TrainConfig", "train_rf_classifier"
)
RfRegTrainConfig, train_rf_regressor = _optional_import(
    "models.rf.rf_regressor", "TrainConfig", "train_rf_regressor"
)

XGBClassifierConfig, XGBOpenDirectionClassifier = _optional_import(
    "models.xgb.xgb_classifier", "XGBClassifierConfig", "XGBOpenDirectionClassifier"
)
XGBRegressorConfig, XGBOpenReturnRegressor = _optional_import(
    "models.xgb.xgb_regressor", "XGBRegressorConfig", "XGBOpenReturnRegressor"
)

 # Transformer trainer lives in models/transformer/ts_transformer.py
TransformerTrainConfig, train_transformer = _optional_import(
    "models.transformer.ts_transformer", "TrainConfig", "train_transformer"
)


DATASETS_DIR = Path("data/datasets")
TRAINED_MODELS_DIR = Path("trained_models")


def task_from_name(name: str) -> str:
    if ".reg_" in name:
        return "reg"
    if ".cls_" in name:
        return "cls"
    return "unknown"


def collect_symbol_datasets(symbol: str) -> list[Path]:
    pattern = f"{symbol.upper()}.*.dataset.csv"
    return sorted(DATASETS_DIR.glob(pattern))


def extract_task_key(ds: Path) -> str:
    return ".".join(ds.name.split(".")[1:-2])


def _load_df(ds: Path) -> pd.DataFrame:
    df = pd.read_csv(ds)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def train_for_symbol(symbol: str) -> None:
    symbol = symbol.upper().strip()
    datasets = collect_symbol_datasets(symbol)

    if not datasets:
        print(f"⚠️  {symbol}: no datasets found in {DATASETS_DIR}")
        return

    # -----------------------------
    # Configs
    # -----------------------------
    cnn_cfg = (
        CnnTrainConfig(seq_len=30, epochs=25, batch_size=128, patience=5)
        if CnnTrainConfig is not None
        else None
    )
    lstm_cfg = (
        LstmTrainConfig(seq_len=20, epochs=40, batch_size=128, patience=6)
        if LstmTrainConfig is not None
        else None
    )

    gbc_cfg = (
        GbcTrainConfig(
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
        if GbcTrainConfig is not None
        else None
    )

    logreg_cfg = (
        LogRegTrainConfig(
            train_ratio=0.80,
            C_grid=(0.01, 0.05, 0.1, 0.5, 1.0),
            tscv_splits=5,
            class_weight="balanced",
            max_iter=2000,
            proba_threshold=0.50,
        )
        if LogRegTrainConfig is not None
        else None
    )

    ridge_cfg = (
        RidgeTrainConfig(
            train_ratio=0.80,
            tscv_splits=5,
            alpha_grid=(0.1, 1.0, 10.0, 100.0),
            with_scaler=True,
            baseline_zero=True,
        )
        if RidgeTrainConfig is not None
        else None
    )

    rf_cls_cfg = RfClsTrainConfig() if RfClsTrainConfig is not None else None
    rf_reg_cfg = RfRegTrainConfig() if RfRegTrainConfig is not None else None

    xgb_cls_cfg = XGBClassifierConfig() if XGBClassifierConfig is not None else None
    xgb_reg_cfg = XGBRegressorConfig() if XGBRegressorConfig is not None else None

    transformer_cfg = (
        TransformerTrainConfig(seq_len=30, epochs=20, batch_size=128, patience=5)
        if TransformerTrainConfig is not None
        else None
    )

    # -----------------------------
    # Tiny wrappers for XGBoost class-based trainers
    # -----------------------------
    def _train_xgb_cls(ds: Path, _task: str, task_key: str) -> dict[str, Any]:
        if XGBOpenDirectionClassifier is None or xgb_cls_cfg is None:
            raise RuntimeError("XGBoost classifier module not available")
        out_dir = _ensure_dir(TRAINED_MODELS_DIR / "xgb_cls" / symbol / task_key)
        df = _load_df(ds)
        model = XGBOpenDirectionClassifier(xgb_cls_cfg)
        model.fit(df, target_col="y")
        model_path = out_dir / "model.json"
        model.save(model_path)
        return {"model_path": str(model_path)}

    def _train_xgb_reg(ds: Path, _task: str, task_key: str) -> dict[str, Any]:
        if XGBOpenReturnRegressor is None or xgb_reg_cfg is None:
            raise RuntimeError("XGBoost regressor module not available")
        out_dir = _ensure_dir(TRAINED_MODELS_DIR / "xgb_reg" / symbol / task_key)
        df = _load_df(ds)
        model = XGBOpenReturnRegressor(xgb_reg_cfg)
        model.fit(df, target_col="y")
        model_path = out_dir / "model.json"
        model.save(model_path)
        return {"model_path": str(model_path)}

    # -----------------------------
    # Trainer registry
    # -----------------------------
    trainers: list[tuple[str, set[str], Callable[[Path, str, str], Any]]] = []

    if train_cnn is not None and cnn_cfg is not None:
        trainers.append(
            (
                "cnn",
                {"reg", "cls"},
                lambda ds, task, task_key: train_cnn(
                    dataset_csv=ds,
                    out_dir=TRAINED_MODELS_DIR / "cnn" / symbol / task_key,
                    task=task,
                    cfg=cnn_cfg,
                    model_kwargs={"conv_channels": 64, "kernel_size": 5, "dropout": 0.15},
                ),
            )
        )
    else:
        print("ℹ️  CNN trainer unavailable; skipping CNN training.")

    if train_lstm is not None and lstm_cfg is not None:
        trainers.append(
            (
                "lstm",
                {"reg", "cls"},
                lambda ds, task, task_key: train_lstm(
                    dataset_csv=ds,
                    out_dir=TRAINED_MODELS_DIR / "lstm" / symbol / task_key,
                    task=task,
                    cfg=lstm_cfg,
                    model_kwargs={"hidden_dim": 64, "num_layers": 1, "dropout": 0.25},
                ),
            )
        )
    else:
        print("ℹ️  LSTM trainer unavailable; skipping LSTM training.")

    if train_gbc is not None and gbc_cfg is not None:
        trainers.append(
            (
                "gbc",
                {"cls"},
                lambda ds, task, task_key: train_gbc(
                    ds,
                    TRAINED_MODELS_DIR / "gbc" / symbol / task_key,
                    cfg=gbc_cfg,
                    scoring="roc_auc",
                ),
            )
        )
    else:
        print("ℹ️  Gradient Boosting Classifier unavailable; skipping GBC training.")

    if train_logreg is not None and logreg_cfg is not None:
        trainers.append(
            (
                "logreg",
                {"cls"},
                lambda ds, task, task_key: train_logreg(
                    ds,
                    TRAINED_MODELS_DIR / "logreg" / symbol / task_key,
                    cfg=logreg_cfg,
                ),
            )
        )
    else:
        print("ℹ️  Logistic Regression unavailable; skipping logreg training.")

    if train_ridge_regressor is not None and ridge_cfg is not None:
        trainers.append(
            (
                "ridge",
                {"reg"},
                lambda ds, task, task_key: train_ridge_regressor(
                    ds,
                    TRAINED_MODELS_DIR / "ridge" / symbol / task_key,
                    cfg=ridge_cfg,
                ),
            )
        )
    else:
        print("ℹ️  Ridge Regression unavailable; skipping ridge training.")

    # Random Forest (only if modules exist)
    if train_rf_classifier is not None and rf_cls_cfg is not None:
        trainers.append(
            (
                "rf_cls",
                {"cls"},
                lambda ds, task, task_key: train_rf_classifier(
                    ds,
                    TRAINED_MODELS_DIR / "rf_cls" / symbol / task_key,
                    cfg=rf_cls_cfg,
                ),
            )
        )
    else:
        print("ℹ️  RF classifier not available (models/rf/rf_classifier.py missing or import failed)")

    if train_rf_regressor is not None and rf_reg_cfg is not None:
        trainers.append(
            (
                "rf_reg",
                {"reg"},
                lambda ds, task, task_key: train_rf_regressor(
                    ds,
                    TRAINED_MODELS_DIR / "rf_reg" / symbol / task_key,
                    cfg=rf_reg_cfg,
                ),
            )
        )
    else:
        print("ℹ️  RF regressor not available (models/rf/rf_regressor.py missing or import failed)")

    # XGBoost (only if modules exist)
    if XGBOpenDirectionClassifier is not None and xgb_cls_cfg is not None:
        trainers.append(("xgb_cls", {"cls"}, _train_xgb_cls))
    else:
        print("ℹ️  XGBoost classifier not available (models/xgb/xgb_classifier.py missing or import failed)")

    if XGBOpenReturnRegressor is not None and xgb_reg_cfg is not None:
        trainers.append(("xgb_reg", {"reg"}, _train_xgb_reg))
    else:
        print("ℹ️  XGBoost regressor not available (models/xgb/xgb_regressor.py missing or import failed)")

    # Transformer (only if module exists)
    if train_transformer is not None and transformer_cfg is not None:
        trainers.append(
            (
                "transformer",
                {"reg", "cls"},
                lambda ds, task, task_key: train_transformer(
                    dataset_csv=ds,
                    out_dir=TRAINED_MODELS_DIR / "transformer" / symbol / task_key,
                    task=task,
                    cfg=transformer_cfg,
                ),
            )
        )
    else:
        print("ℹ️  Transformer not available (models/transformer/train_transformer.py missing or import failed)")

    # -----------------------------
    # Execute
    # -----------------------------
    if not trainers:
        print("⚠️  No trainers available; ensure model implementations are present.")
        return

    ok = 0
    failed: list[str] = []

    for ds in datasets:
        task = task_from_name(ds.name)
        if task == "unknown":
            print(f"⚠️  {symbol}: skipping unknown dataset type -> {ds.name}")
            continue

        task_key = extract_task_key(ds)

        for name, supported_tasks, fn in trainers:
            if task not in supported_tasks:
                continue
            try:
                res = fn(ds, task, task_key)
                print(f"✅ {symbol} {task_key} [{name}] -> {res}")
                ok += 1
            except Exception as exc:  # noqa: BLE001
                failed.append(f"{ds.name} ({name}) -> {exc}")
                print(f"❌ {symbol} {task_key} [{name}] failed: {exc}")

    print(f"\n{symbol}: {ok} models trained")
    if failed:
        print("Failures:")
        print("\n".join(failed))


def main() -> None:
    symbols = load_symbols()
    TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    for sym in symbols:
        train_for_symbol(sym)


if __name__ == "__main__":
    main()
