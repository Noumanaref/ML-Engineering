"""Feature extraction and engineering for freight rate prediction models."""
from __future__ import annotations

from typing import Sequence
import numpy as np
import pandas as pd

from src.config import EQUIPMENT_TYPES, START_EPOCH

# Canonical static feature subset for residual trees
STATIC_GBM_FEATURES: list[str] = ["log_dist", "weight", "equip"]


class FeatureEngineer:
    """Transforms raw load records into structured feature matrices.
    
    Generates:
    - Log distance (capturing scale economies where $/mile falls non-linearly with distance).
    - Normalized load weight.
    - Ordinal and one-hot equipment encodings.
    - Macro market index and normalized time progression (months since Jan 1, 2025).
    - Day-of-week and weekend indicators.
    """

    def __init__(self) -> None:
        self.equipment_categories = list(EQUIPMENT_TYPES)

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate comprehensive candidate feature set from cleaned load data."""
        features = pd.DataFrame(index=df.index)

        # 1. Non-linear distance and weight transformations
        features["distance"] = df["distance"].astype(float)
        features["log_dist"] = np.log(df["distance"].astype(float))
        features["weight"] = df["weight"].astype(float)
        features["log_weight"] = np.log(np.maximum(df["weight"].astype(float), 1.0))

        # 2. Equipment encodings
        equip_cat = pd.Categorical(df["equipment"], categories=self.equipment_categories)
        features["equip"] = equip_cat.codes
        features["equip_Flatbed"] = (features["equip"] == 1).astype(float)
        features["equip_Reefer"] = (features["equip"] == 2).astype(float)

        # 3. Temporal & Macro index
        date_series = pd.to_datetime(df["date"])
        features["market"] = df["market_index"].astype(float)
        features["t"] = (date_series - START_EPOCH).dt.days / 30.0

        # 4. Weekly seasonality
        features["dow"] = date_series.dt.dayofweek.astype(int)
        features["is_weekend"] = features["dow"].isin([5, 6]).astype(float)

        return features

    def build_linear_matrix(
        self, df: pd.DataFrame, include_time: bool = True
    ) -> pd.DataFrame:
        """Extract design matrix for Stage-1 Ridge regression."""
        feat = self.extract_features(df)
        cols = ["log_dist", "market"]
        if include_time:
            cols.append("t")
        cols.extend(["equip_Flatbed", "equip_Reefer"])
        return feat[cols].copy()

    def build_gbm_matrix(
        self, df: pd.DataFrame, feature_names: Sequence[str] = STATIC_GBM_FEATURES
    ) -> pd.DataFrame:
        """Extract feature matrix for Stage-2 residual LightGBM regressor."""
        feat = self.extract_features(df)
        return feat[list(feature_names)].copy()
