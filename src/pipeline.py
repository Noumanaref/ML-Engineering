"""End-to-end pipeline orchestrator for validation backtesting, training, and inference."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

from src.anomalies import flag_corrupted_labels
from src.config import (
    FORWARD_FOLDS,
    BlendConfig,
    PipelineConfig,
    TARGET_COLUMN,
)
from src.data_cleaning import DataCleaner, find_data_file
from src.metrics import LookupTableBaseline, compute_metrics
from src.models import GeometricBlendModel, HybridFreightModel

logger = logging.getLogger(__name__)


def build_production_model(blend_config: BlendConfig | None = None) -> GeometricBlendModel:
    """Instantiate the 70/30 geometric blend of with-time and no-time hybrid models."""
    config = blend_config or BlendConfig()
    model_with_time = HybridFreightModel(include_time=True)
    model_no_time = HybridFreightModel(include_time=False)
    return GeometricBlendModel(
        models=[model_with_time, model_no_time],
        weights=config.weights,
    )


def run_cross_validation(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None = None,
) -> Dict[str, Any]:
    """Execute honest forward-chaining backtest across temporal folds V1-V4."""
    # Construct complete market calendar using combined dates
    combined = pd.concat([train_df, val_df]) if val_df is not None else train_df
    daily_market = (
        combined.groupby(pd.to_datetime(combined["date"]).dt.normalize())["market_index"]
        .median()
        .ffill()
        .bfill()
    )

    cleaner = DataCleaner().fit(train_df)
    cleaned_train = cleaner.transform(train_df, daily_market=daily_market)
    cleaned_train["is_corrupted"] = flag_corrupted_labels(cleaned_train)

    fold_results = []
    for fold_name, (val_start, val_end) in FORWARD_FOLDS.items():
        train_mask = (cleaned_train["date"] < val_start) & (~cleaned_train["is_corrupted"])
        test_mask = (cleaned_train["date"] >= val_start) & (cleaned_train["date"] <= val_end)

        fold_train = cleaned_train[train_mask]
        fold_test = cleaned_train[test_mask].copy()

        # Fit baselines and final blend
        baseline = LookupTableBaseline().fit(fold_train)
        model = build_production_model().fit(fold_train)

        fold_test["pred_lookup"] = baseline.predict(fold_test)
        fold_test["pred_final"] = model.predict(fold_test)
        fold_test["fold"] = fold_name
        fold_results.append(fold_test)

    eval_df = pd.concat(fold_results, ignore_index=True)
    clean_eval = eval_df[~eval_df["is_corrupted"]]

    summary = {
        "clean_rows": {
            "lookup": compute_metrics(clean_eval[TARGET_COLUMN], clean_eval["pred_lookup"]),
            "final": compute_metrics(clean_eval[TARGET_COLUMN], clean_eval["pred_final"]),
        },
        "all_rows": {
            "lookup": compute_metrics(eval_df[TARGET_COLUMN], eval_df["pred_lookup"]),
            "final": compute_metrics(eval_df[TARGET_COLUMN], eval_df["pred_final"]),
        },
        "per_fold_wape": {
            f: compute_metrics(
                clean_eval[clean_eval["fold"] == f][TARGET_COLUMN],
                clean_eval[clean_eval["fold"] == f]["pred_final"],
            )["WAPE"]
            for f in FORWARD_FOLDS
        },
    }
    return summary


def run_pipeline(
    data_dir: Path | str = "data",
    output_dir: Path | str = ".",
    expected_rows: int = 12_000,
) -> Tuple[Path, Path]:
    """Execute complete end-to-end pipeline and emit submission files."""
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("Resolving source files in %s...", data_path)
    train_file = find_data_file(data_path, "train_test")
    val_file = find_data_file(data_path, "validation")
    template_file = find_data_file(data_path, "validation_predictions_template")
    dec_file = find_data_file(data_path, "december_chart_inputs")

    # 1. Load data
    train_raw = pd.read_csv(train_file, parse_dates=["date"])
    val_raw = pd.read_csv(val_file, parse_dates=["date"])
    template = pd.read_csv(template_file)
    dec_raw = pd.read_csv(dec_file, parse_dates=["date"])

    # 2. Build continuous daily market index series across train and validation
    combined = pd.concat([train_raw, val_raw], ignore_index=True)
    daily_market = (
        combined.groupby(combined["date"].dt.normalize())["market_index"]
        .median()
        .ffill()
        .bfill()
    )

    # 3. Clean records strictly fitting preprocessing statistics on train
    cleaner = DataCleaner().fit(train_raw)
    train_clean = cleaner.transform(train_raw, daily_market=daily_market)
    val_clean = cleaner.transform(val_raw, daily_market=daily_market)

    # Prepare December inputs (remove existing predicted_rate column before transform)
    dec_features = dec_raw.drop(columns=["predicted_rate"], errors="ignore")
    dec_clean = cleaner.transform(dec_features, daily_market=daily_market)

    # 4. Isolate training noise and train production model
    train_clean["is_corrupted"] = flag_corrupted_labels(train_clean)
    clean_training_set = train_clean[~train_clean["is_corrupted"]]
    logger.info(
        "Fitting production blend on %d clean historical records (%d corrupted records purged)...",
        len(clean_training_set),
        int(train_clean["is_corrupted"].sum()),
    )

    production_model = build_production_model().fit(clean_training_set)

    # 5. Predict on validation set and align with template order
    logger.info("Generating predictions for %d validation loads...", len(val_clean))
    val_preds = pd.DataFrame({
        "load_id": val_clean["load_id"],
        "predicted_rate": production_model.predict(val_clean).round(2),
    })

    submission = template[["load_id"]].merge(val_preds, on="load_id", how="left")
    if len(submission) != expected_rows:
        raise ValueError(f"Expected {expected_rows} predictions, generated {len(submission)}")
    if submission["predicted_rate"].isna().any():
        raise ValueError("Missing rate predictions detected in validation submission.")
    if (submission["predicted_rate"] <= 0).any():
        raise ValueError("Non-positive rate predictions encountered in validation submission.")

    val_sub_path = out_path / "validation_predictions.csv"
    submission.to_csv(val_sub_path, index=False)
    logger.info("Successfully written: %s", val_sub_path)

    # 6. Predict on fixed December loads for scoring and trend visualization
    logger.info("Generating December benchmark predictions...")
    dec_out = dec_raw.copy()
    dec_out["predicted_rate"] = production_model.predict(dec_clean).round(2)
    dec_out["date"] = dec_out["date"].dt.strftime("%Y-%m-%d")

    dec_sub_path = out_path / "december_chart_inputs.csv"
    dec_out.to_csv(dec_sub_path, index=False)
    logger.info("Successfully written: %s", dec_sub_path)

    return val_sub_path, dec_sub_path
