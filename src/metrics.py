"""Evaluation metrics and benchmark baselines for freight spot rate models."""
from __future__ import annotations

from typing import Dict, Sequence
import numpy as np
import pandas as pd

from src.config import DISTANCE_BINS, TARGET_COLUMN


def compute_metrics(
    y_true: Sequence[float] | np.ndarray | pd.Series,
    y_pred: Sequence[float] | np.ndarray | pd.Series,
) -> Dict[str, float]:
    """Compute industry-standard regression metrics including WAPE and financial bias."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)

    abs_err = np.abs(y - p)
    sq_err = (y - p) ** 2
    rel_err = abs_err / np.maximum(y, 1e-9)

    sum_true = float(np.sum(y))
    sum_pred = float(np.sum(p))
    sum_abs_err = float(np.sum(abs_err))

    return {
        "MAE": float(np.mean(abs_err)),
        "RMSE": float(np.sqrt(np.mean(sq_err))),
        "MAPE": float(np.mean(rel_err) * 100.0),
        "WAPE": float((sum_abs_err / sum_true) * 100.0) if sum_true > 0 else 0.0,
        "Bias": float(sum_pred / sum_true) if sum_true > 0 else 1.0,
        "Within_5pct": float(np.mean(rel_err < 0.05) * 100.0),
        "Within_10pct": float(np.mean(rel_err < 0.10) * 100.0),
    }


class LookupTableBaseline:
    """Benchmark baseline predicting $/mile by equipment and distance band."""

    def __init__(self) -> None:
        self.table: pd.Series | None = None
        self.default_rpm: float = 2.0

    def fit(self, df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> "LookupTableBaseline":
        bins = pd.cut(df["distance"], bins=list(DISTANCE_BINS))
        rpm = df[target_col] / df["distance"]
        self.table = rpm.groupby([df["equipment"], bins], observed=False).median()
        self.default_rpm = float(rpm.median())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.table is None:
            raise RuntimeError("LookupTableBaseline must be fitted before predict.")
        bins = pd.cut(df["distance"], bins=list(DISTANCE_BINS))
        keys = pd.MultiIndex.from_arrays([df["equipment"], bins])
        lookup_rpm = self.table.reindex(keys).fillna(self.default_rpm).values
        return lookup_rpm * df["distance"].values
