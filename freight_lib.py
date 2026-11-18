"""Shared code for the freight-rate project (cleaning, features, models, metrics).

Provides backward-compatible interfaces to the production `src` package.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd

from src.anomalies import flag_corrupted_labels
from src.config import (
    DISTANCE_BINS,
    EQUIPMENT_TYPES,
    FORWARD_FOLDS,
    START_EPOCH,
    TARGET_COLUMN,
    LightGBMConfig,
)
from src.data_cleaning import DataCleaner, find_data_file
from src.features import FeatureEngineer, STATIC_GBM_FEATURES
from src.metrics import LookupTableBaseline, compute_metrics
from src.models import GeometricBlendModel, HybridFreightModel

# Aliases for backward compatibility
TARGET = TARGET_COLUMN
EQUIPMENT = list(EQUIPMENT_TYPES)
START_DATE = START_EPOCH
DIST_BINS = list(DISTANCE_BINS)
DEFAULT_GBM = LightGBMConfig().to_dict()
STATIC = STATIC_GBM_FEATURES
FOLDS = FORWARD_FOLDS


def find(data_dir: Path | str, stem: str) -> str:
    """Find train_test.csv / train-test.csv / 'train-test (1).csv'."""
    return str(find_data_file(data_dir, stem))


class Cleaner:
    """Fix weights and market_index. Fill values are learned from train only."""

    def __init__(self) -> None:
        self._cleaner = DataCleaner()

    def fit(self, df: pd.DataFrame) -> "Cleaner":
        self._cleaner.fit(df)
        self.weight_by_equipment = self._cleaner.equipment_weight_medians
        self.weight_overall = self._cleaner.overall_weight_median
        return self

    def transform(self, df: pd.DataFrame, market_by_day: pd.Series | None = None) -> pd.DataFrame:
        return self._cleaner.transform(df, daily_market=market_by_day)


def flag_corrupted(df: pd.DataFrame, low: float = 0.50, high: float = 2.00) -> pd.Series:
    """True where $/mile is far from the typical value for its equipment x distance band."""
    return flag_corrupted_labels(df, target_col=TARGET, low_ratio=low, high_ratio=high)


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate tabular feature matrix from raw load records."""
    fe = FeatureEngineer()
    raw = fe.extract_features(df)
    X = pd.DataFrame(index=df.index)
    X["log_dist"] = raw["log_dist"]
    X["weight"] = raw["weight"]
    X["equip"] = raw["equip"]
    X["market"] = raw["market"]
    X["t"] = raw["t"]
    if "quote_signal" in df.columns:
        X["quote"] = df["quote_signal"].astype(float)
    return X


def score(y: Sequence[float], p: Sequence[float]) -> dict[str, float]:
    """Calculate MAE, RMSE, MAPE, and WAPE."""
    m = compute_metrics(y, p)
    return {"MAE": m["MAE"], "RMSE": m["RMSE"], "MAPE": m["MAPE"], "WAPE": m["WAPE"]}


class LookupBaseline:
    """Lookup table baseline matching equipment and distance bands."""

    def __init__(self) -> None:
        self._baseline = LookupTableBaseline()

    def fit(self, df: pd.DataFrame) -> "LookupBaseline":
        self._baseline.fit(df, target_col=TARGET)
        self.table = self._baseline.table
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self._baseline.predict(df)


class Hybrid:
    """Step 1: Ridge line on market (and optionally time). Step 2: LightGBM on residuals."""

    def __init__(
        self,
        linear_cols: Sequence[str],
        gbm_cols: Sequence[str] = STATIC,
        alpha: float = 1.0,
        gbm_params: dict | None = None,
    ) -> None:
        self.linear_cols = list(linear_cols)
        self.gbm_cols = list(gbm_cols)
        self.alpha = alpha
        self.gbm_params = {**DEFAULT_GBM, **(gbm_params or {})}
        include_time = "t" in self.linear_cols
        self._model = HybridFreightModel(
            include_time=include_time,
            gbm_features=self.gbm_cols,
            alpha=self.alpha,
            gbm_params=self.gbm_params,
        )

    def fit(self, df: pd.DataFrame) -> "Hybrid":
        self._model.fit(df, target_col=TARGET)
        return self

    def predict_log(self, df: pd.DataFrame) -> np.ndarray:
        return self._model.predict_log(df)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self._model.predict(df)


class Blend:
    """Average of several models in log space (a geometric mean)."""

    def __init__(
        self,
        models: Sequence[Hybrid],
        weights: Sequence[float] | None = None,
    ) -> None:
        self.models = list(models)
        self.weights = list(weights) if weights is not None else [1.0 / len(models)] * len(models)
        self._blend = GeometricBlendModel(
            models=[m._model for m in self.models],
            weights=self.weights,
        )

    def fit(self, df: pd.DataFrame) -> "Blend":
        self._blend.fit(df, target_col=TARGET)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self._blend.predict(df)

    def predict_log(self, df: pd.DataFrame) -> np.ndarray:
        return self._blend.predict_log(df)
