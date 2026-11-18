"""Configuration, type definitions, and domain constants for the Freight Rate system."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence
import pandas as pd

# Domain Constants
TARGET_COLUMN: str = "posted_rate"
DATE_COLUMN: str = "date"
LOAD_ID_COLUMN: str = "load_id"

EQUIPMENT_TYPES: tuple[str, ...] = ("Dry Van", "Flatbed", "Reefer")
DISTANCE_BINS: tuple[float, ...] = (0.0, 150.0, 300.0, 500.0, 800.0, 1200.0, 2000.0, 5000.0)
START_EPOCH: pd.Timestamp = pd.Timestamp("2025-01-01")

# Forward Chaining Validation Folds (Simulating 1-2 month out-of-distribution forecasting)
FORWARD_FOLDS: dict[str, tuple[str, str]] = {
    "V1": ("2025-07-01", "2025-08-31"),
    "V2": ("2025-08-01", "2025-09-30"),
    "V3": ("2025-09-01", "2025-10-31"),
    "V4": ("2025-10-01", "2025-10-31"),
}

# Domain Plausibility Guardrails (Rate per Mile)
MIN_RATE_PER_MILE: float = 0.50
MAX_RATE_PER_MILE: float = 8.00


@dataclass(frozen=True)
class LightGBMConfig:
    """Hyperparameters for the residual gradient boosting regressor."""
    n_estimators: int = 300
    learning_rate: float = 0.05
    num_leaves: int = 15
    min_child_samples: int = 50
    subsample: float = 0.80
    subsample_freq: int = 1
    colsample_bytree: float = 0.80
    random_state: int = 42
    verbose: int = -1

    def to_dict(self) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_child_samples": self.min_child_samples,
            "subsample": self.subsample,
            "subsample_freq": self.subsample_freq,
            "colsample_bytree": self.colsample_bytree,
            "random_state": self.random_state,
            "verbose": self.verbose,
        }


@dataclass(frozen=True)
class RidgeConfig:
    """Hyperparameters for the regularized macro linear model."""
    alpha: float = 1.0


@dataclass(frozen=True)
class BlendConfig:
    """Weights for ensembling models with and without macro time trend."""
    with_time_weight: float = 0.70
    no_time_weight: float = 0.30

    @property
    def weights(self) -> list[float]:
        return [self.with_time_weight, self.no_time_weight]


@dataclass
class PipelineConfig:
    """Top-level pipeline paths and runtime settings."""
    data_dir: Path = Path("data")
    output_dir: Path = Path(".")
    expected_validation_rows: int = 12_000
    lgbm: LightGBMConfig = field(default_factory=LightGBMConfig)
    ridge: RidgeConfig = field(default_factory=RidgeConfig)
    blend: BlendConfig = field(default_factory=BlendConfig)
