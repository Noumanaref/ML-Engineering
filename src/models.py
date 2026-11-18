"""Hybrid Residual Model and Geometric Ensembling for freight rate prediction."""
from __future__ import annotations

from typing import Any, Sequence
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.config import (
    LightGBMConfig,
    MAX_RATE_PER_MILE,
    MIN_RATE_PER_MILE,
    RidgeConfig,
    TARGET_COLUMN,
)
from src.features import FeatureEngineer, STATIC_GBM_FEATURES


class HybridFreightModel(BaseEstimator, RegressorMixin):
    """Two-stage residual hybrid model for robust out-of-distribution freight forecasting.
    
    Architecture:
    1. Stage 1 (Regularized Linear Core): Ridge regression in log space captures macro market level
       and baseline equipment costs. Being linear, it reliably extrapolates trends to future dates.
    2. Stage 2 (Non-linear Residual Booster): LightGBM Regressor fitted on the Stage 1 residuals
       captures non-linear distance economies of scale and equipment-weight interactions.
    3. Prediction: exp(Stage 1 + Stage 2) bounded by physical rate-per-mile feasibility constraints.
    """

    def __init__(
        self,
        include_time: bool = True,
        gbm_features: Sequence[str] = STATIC_GBM_FEATURES,
        alpha: float = 1.0,
        gbm_params: dict[str, Any] | None = None,
        min_rpm: float = MIN_RATE_PER_MILE,
        max_rpm: float = MAX_RATE_PER_MILE,
    ) -> None:
        self.include_time = include_time
        self.gbm_features = list(gbm_features)
        self.alpha = alpha
        self.gbm_params = gbm_params or LightGBMConfig().to_dict()
        self.min_rpm = min_rpm
        self.max_rpm = max_rpm

        self.feature_engineer = FeatureEngineer()
        self.scaler = StandardScaler()
        self.linear_model = Ridge(alpha=self.alpha)
        self.gbm_model = lgb.LGBMRegressor(**self.gbm_params)
        self.is_fitted = False

    def fit(self, df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> "HybridFreightModel":
        """Fit Stage 1 on log(target), compute residuals, and fit Stage 2 on residuals."""
        y_log = np.log(df[target_col].astype(float).values)

        # Stage 1: Linear model on standardized macro design matrix
        X_linear = self.feature_engineer.build_linear_matrix(df, include_time=self.include_time)
        X_scaled = self.scaler.fit_transform(X_linear)
        self.linear_model.fit(X_scaled, y_log)

        # Compute Stage 1 residuals
        linear_preds = self.linear_model.predict(X_scaled)
        residuals = y_log - linear_preds

        # Stage 2: LightGBM on route and equipment static features
        X_gbm = self.feature_engineer.build_gbm_matrix(df, feature_names=self.gbm_features)
        self.gbm_model.fit(X_gbm, residuals)

        self.is_fitted = True
        return self

    def predict_log(self, df: pd.DataFrame) -> np.ndarray:
        """Predict expected rate in log-space."""
        if not self.is_fitted:
            raise RuntimeError("HybridFreightModel must be fitted before predict.")

        X_linear = self.feature_engineer.build_linear_matrix(df, include_time=self.include_time)
        X_scaled = self.scaler.transform(X_linear)
        linear_log_pred = self.linear_model.predict(X_scaled)

        X_gbm = self.feature_engineer.build_gbm_matrix(df, feature_names=self.gbm_features)
        residual_pred = self.gbm_model.predict(X_gbm)

        return linear_log_pred + residual_pred

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict posted freight rates ($) with domain guardrails."""
        pred_rates = np.exp(self.predict_log(df))

        # Apply domain rate-per-mile bounds to prevent edge distortions
        distance = df["distance"].astype(float).values
        lower_bound = distance * self.min_rpm
        upper_bound = distance * self.max_rpm

        return np.clip(pred_rates, lower_bound, upper_bound)


class GeometricBlendModel(BaseEstimator, RegressorMixin):
    """Ensemble averaging multiple models in log space (geometric mean).
    
    Ensembling a model with a linear time trend (70%) with a model without time trend (30%)
    mitigates the risk of long-horizon drift while retaining recent trend momentum.
    """

    def __init__(
        self,
        models: Sequence[HybridFreightModel],
        weights: Sequence[float] | None = None,
    ) -> None:
        self.models = list(models)
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            total_weight = sum(weights)
            self.weights = [w / total_weight for w in weights]

    def fit(self, df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> "GeometricBlendModel":
        for model in self.models:
            model.fit(df, target_col=target_col)
        return self

    def predict_log(self, df: pd.DataFrame) -> np.ndarray:
        weighted_logs = sum(
            w * m.predict_log(df) for w, m in zip(self.weights, self.models)
        )
        return weighted_logs

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        rates = np.exp(self.predict_log(df))
        distance = df["distance"].astype(float).values
        return np.clip(rates, distance * MIN_RATE_PER_MILE, distance * MAX_RATE_PER_MILE)
