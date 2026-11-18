"""Robust data cleaning, imputation, and file resolution for freight pricing data."""
from __future__ import annotations

import glob
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.config import EQUIPMENT_TYPES


def find_data_file(data_dir: Path | str, stem: str) -> Path:
    """Resolve file paths handling hyphens, underscores, and browser copy suffixes.
    
    Example: stem='train_test' matches 'train_test.csv', 'train-test.csv', or 'train-test (1).csv'.
    """
    directory = Path(data_dir)
    variants = [stem, stem.replace("_", "-")]
    for name in variants:
        target = directory / f"{name}.csv"
        if target.is_file():
            return target

    for name in variants:
        matches = sorted(glob.glob(str(directory / f"{name} (*).csv")))
        if matches:
            return Path(matches[0])

    raise FileNotFoundError(f"Could not locate dataset file for '{stem}' in directory: {directory}")


class DataCleaner:
    """Preprocesses raw freight records with zero data leakage.
    
    - Resolves negative weight records via sign inversion (abs).
    - Imputes missing weights using equipment-specific medians learned strictly on train.
    - Rebuilds a continuous calendar of daily median market_index values with forward/backward fill.
    """

    def __init__(self) -> None:
        self.equipment_weight_medians: Dict[str, float] = {}
        self.overall_weight_median: float = 30_000.0
        self.market_calendar: Optional[pd.Series] = None
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "DataCleaner":
        """Learn imputation statistics strictly from the training partition."""
        clean_weights = df["weight"].abs()
        self.equipment_weight_medians = (
            clean_weights.groupby(df["equipment"]).median().to_dict()
        )
        self.overall_weight_median = float(clean_weights.median())

        # Build clean daily market index calendar if market_index exists
        if "market_index" in df.columns:
            daily_medians = df.groupby(pd.to_datetime(df["date"]).dt.normalize())["market_index"].median()
            full_dates = pd.date_range(daily_medians.index.min(), daily_medians.index.max(), freq="D")
            self.market_calendar = daily_medians.reindex(full_dates).ffill().bfill()

        self.is_fitted = True
        return self

    def transform(
        self, df: pd.DataFrame, daily_market: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Apply cleaning transformations without modifying the input DataFrame in-place."""
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"])

        # 1. Clean and impute load weights
        out["weight"] = out["weight"].abs()
        equipment_map = out["equipment"].map(self.equipment_weight_medians)
        out["weight"] = (
            out["weight"]
            .fillna(equipment_map)
            .fillna(self.overall_weight_median)
        )

        # 2. Reconstruct or impute market_index
        normalized_dates = out["date"].dt.normalize()
        if daily_market is not None:
            market_source = daily_market
        elif self.market_calendar is not None:
            market_source = self.market_calendar
        else:
            market_source = pd.Series(dtype=float)

        if "market_index" in out.columns:
            out["market_index"] = out["market_index"].fillna(normalized_dates.map(market_source))
        else:
            out["market_index"] = normalized_dates.map(market_source)

        # Fallback if any date extends beyond known market calendar
        if out["market_index"].isna().any():
            fallback_val = (
                self.market_calendar.median()
                if self.market_calendar is not None and not self.market_calendar.empty
                else 1.0
            )
            out["market_index"] = out["market_index"].ffill().bfill().fillna(fallback_val)

        return out
