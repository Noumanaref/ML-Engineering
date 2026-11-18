"""Detection and isolation of synthetic rate anomalies and corrupted training labels."""
from __future__ import annotations

import pandas as pd
from src.config import DISTANCE_BINS, TARGET_COLUMN


def flag_corrupted_labels(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    low_ratio: float = 0.50,
    high_ratio: float = 2.00,
) -> pd.Series:
    """Identify corrupted load rates where $/mile diverges severely from equipment-distance norms.
    
    Approximately 1.4% of development records contain corrupted labels (3-5x spikes or ~0.28x drops).
    These are purged strictly from training sets to protect gradient estimators, while retained
    in validation backtests to realistically assess production noise tolerance.
    """
    if target_col not in df.columns:
        return pd.Series(False, index=df.index)

    distance_bins = pd.cut(df["distance"], bins=list(DISTANCE_BINS))
    rate_per_mile = df[target_col] / df["distance"]

    typical_rpm = rate_per_mile.groupby(
        [df["equipment"], distance_bins], observed=False
    ).transform("median")

    rpm_ratio = rate_per_mile / typical_rpm
    is_anomaly = (rpm_ratio < low_ratio) | (rpm_ratio > high_ratio)
    return is_anomaly.fillna(False)


class LabelAnomalyDetector:
    """Fitted detector storing reference rate-per-mile medians per equipment and distance band."""

    def __init__(self, low_ratio: float = 0.50, high_ratio: float = 2.00) -> None:
        self.low_ratio = low_ratio
        self.high_ratio = high_ratio
        self.reference_table: pd.Series | None = None

    def fit(self, df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> "LabelAnomalyDetector":
        distance_bins = pd.cut(df["distance"], bins=list(DISTANCE_BINS))
        rate_per_mile = df[target_col] / df["distance"]
        self.reference_table = rate_per_mile.groupby(
            [df["equipment"], distance_bins], observed=False
        ).median()
        return self

    def flag(self, df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> pd.Series:
        if target_col not in df.columns or self.reference_table is None:
            return pd.Series(False, index=df.index)

        distance_bins = pd.cut(df["distance"], bins=list(DISTANCE_BINS))
        rpm = df[target_col] / df["distance"]
        lookup_keys = pd.MultiIndex.from_arrays([df["equipment"], distance_bins])
        expected_rpm = self.reference_table.reindex(lookup_keys).values
        ratio = rpm.values / expected_rpm
        return pd.Series((ratio < self.low_ratio) | (ratio > self.high_ratio), index=df.index).fillna(False)
