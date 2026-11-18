"""Unit tests for FeatureEngineer."""
import numpy as np
import pandas as pd
import pytest

from src.features import FeatureEngineer, STATIC_GBM_FEATURES


def test_feature_engineer_extracts_correct_columns():
    df = pd.DataFrame({
        "distance": [350.0, 720.0],
        "weight": [32000.0, 44000.0],
        "equipment": ["Dry Van", "Reefer"],
        "date": pd.to_datetime(["2025-03-01", "2025-03-02"]),
        "market_index": [1.02, 0.98],
    })
    fe = FeatureEngineer()
    extracted = fe.extract_features(df)

    expected_cols = {"distance", "log_dist", "weight", "log_weight", "equip",
                     "equip_Flatbed", "equip_Reefer", "market", "t", "dow", "is_weekend"}
    assert expected_cols.issubset(set(extracted.columns))

    lin_matrix = fe.build_linear_matrix(df, include_time=True)
    assert "t" in lin_matrix.columns
    assert "log_dist" in lin_matrix.columns

    gbm_matrix = fe.build_gbm_matrix(df, STATIC_GBM_FEATURES)
    assert list(gbm_matrix.columns) == STATIC_GBM_FEATURES
