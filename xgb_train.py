"""Train only XGBoost-based models.

This script focuses solely on XGBoost classifiers/regressors. It mirrors the
dataset discovery conventions used elsewhere in the project and adds
user-friendly error/debug logging to make troubleshooting easier when imports or
training steps fail.
"""
from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

from datasets import load_symbols

logger = logging.getLogger("xgb_train")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DATASETS_DIR = Path("data/datasets")
TRAINED_MODELS_DIR = Path("trained_models")


# Import XGBoost helpers lazily so we can surface clear debug information when
# modules are missing or broken.
def _load_xgb_components() -> tuple[Any | None, Any | None, Any | None, Any | None]:
    cls_cfg_cls = cls_model_cls = reg_cfg_cls = reg_model_cls = None

    try:
        from models.xgb.xgb_classifier import XGBClassifierConfig, XGBOpenDirectionClassifier

        cls_cfg_cls = XGBClassifierConfig
        cls_model_cls = XGBOpenDirectionClassifier
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to import XGBoost classifier components: %s", exc)
        logger.debug("Classifier import traceback:\n%s", traceback.format_exc())

    try:
        from models.xgb.xgb_regressor import XGBRegressorConfig, XGBOpenReturnRegressor

        reg_cfg_cls = XGBRegressorConfig
        reg_model_cls = XGBOpenReturnRegressor
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to import XGBoost regressor components: %s", exc)
        logger.debug("Regressor import traceback:\n%s", traceback.format_exc())

    return cls_cfg_cls, cls_model_cls, reg_cfg_cls, reg_model_cls


def _task_from_name(name: str) -> str:
    if ".reg_" in name:
        return "reg"
    if ".cls_" in name:
        return "cls"
    return "unknown"


def _extract_task_key(ds: Path) -> str:
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


def _train_xgb_classifier(ds: Path, symbol: str, task_key: str, cfg_cls: Any, model_cls: Any) -> dict[str, Any]:
    logger.info("[CLS] %s %s -> starting", symbol, task_key)
    out_dir = _ensure_dir(TRAINED_MODELS_DIR / "xgb_cls" / symbol / task_key)
    df = _load_df(ds)
    model = model_cls(cfg_cls())
    model.fit(df, target_col="y")
    model_path = out_dir / "model.json"
    model.save(model_path)
    logger.info("[CLS] %s %s -> saved to %s", symbol, task_key, model_path)
    return {"model_path": str(model_path)}


def _train_xgb_regressor(ds: Path, symbol: str, task_key: str, cfg_cls: Any, model_cls: Any) -> dict[str, Any]:
    logger.info("[REG] %s %s -> starting", symbol, task_key)
    out_dir = _ensure_dir(TRAINED_MODELS_DIR / "xgb_reg" / symbol / task_key)
    df = _load_df(ds)
    model = model_cls(cfg_cls())
    model.fit(df, target_col="y")
    model_path = out_dir / "model.json"
    model.save(model_path)
    logger.info("[REG] %s %s -> saved to %s", symbol, task_key, model_path)
    return {"model_path": str(model_path)}


def train_for_symbol(symbol: str) -> None:
    cls_cfg_cls, cls_model_cls, reg_cfg_cls, reg_model_cls = _load_xgb_components()
    if cls_cfg_cls is None and reg_cfg_cls is None:
        logger.error("XGBoost modules are unavailable; aborting %s", symbol)
        return

    symbol = symbol.upper().strip()
    pattern = f"{symbol}.*.dataset.csv"
    datasets = sorted(DATASETS_DIR.glob(pattern))
    if not datasets:
        logger.warning("%s: no datasets found in %s", symbol, DATASETS_DIR)
        return

    ok = 0
    failed: list[str] = []

    for ds in datasets:
        task = _task_from_name(ds.name)
        if task == "unknown":
            logger.warning("%s: skipping unknown dataset type -> %s", symbol, ds.name)
            continue

        task_key = _extract_task_key(ds)
        try:
            if task == "cls":
                if cls_cfg_cls is None or cls_model_cls is None:
                    raise RuntimeError("XGBoost classifier components missing; see earlier errors")
                res = _train_xgb_classifier(ds, symbol, task_key, cls_cfg_cls, cls_model_cls)
            else:
                if reg_cfg_cls is None or reg_model_cls is None:
                    raise RuntimeError("XGBoost regressor components missing; see earlier errors")
                res = _train_xgb_regressor(ds, symbol, task_key, reg_cfg_cls, reg_model_cls)

            logger.info("✅ %s %s [%s] -> %s", symbol, task_key, task, res)
            ok += 1
        except Exception as exc:  # noqa: BLE001
            logger.error("❌ %s %s [%s] failed: %s", symbol, task_key, task, exc)
            logger.debug("Traceback for %s: %s", ds.name, traceback.format_exc())
            failed.append(f"{ds.name} ({task}) -> {exc}")

    logger.info("%s: %s models trained", symbol, ok)
    if failed:
        logger.error("Failures for %s:\n%s", symbol, "\n".join(failed))


def main() -> None:
    symbols = load_symbols() if len(sys.argv) < 2 else [sys.argv[1].strip().upper()]
    TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    for sym in symbols:
        train_for_symbol(sym)


if __name__ == "__main__":
    main()
