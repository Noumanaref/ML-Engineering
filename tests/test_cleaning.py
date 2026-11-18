"""Unit tests for DataCleaner and file finder utilities."""
import numpy as np
import pandas as pd
import pytest

from src.data_cleaning import DataCleaner, find_data_file


def test_cleaner_handles_negative_and_missing_weights():
    df = pd.DataFrame({
        "equipment": ["Dry Van", "Reefer", "Flatbed", "Dry Van"],
        "weight": [-32000.0, np.nan, -44000.0, 31000.0],
        "distance": [300.0, 500.0, 200.0, 800.0],
        "date": pd.date_range("2025-01-01", periods=4),
        "market_index": [1.0, 1.05, 0.98, 1.02],
    })
    cleaner = DataCleaner().fit(df)
    transformed = cleaner.transform(df)

    assert (transformed["weight"] > 0).all()
    assert transformed["weight"].notna().all()
    assert transformed.loc[0, "weight"] == 32000.0
    assert transformed.loc[2, "weight"] == 44000.0


def test_cleaner_reconstructs_market_calendar_and_ffills():
    train_df = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=5, freq="D"),
        "equipment": ["Dry Van"] * 5,
        "weight": [30000.0] * 5,
        "distance": [500.0] * 5,
        "market_index": [1.0, 1.02, np.nan, 1.06, 1.08],
    })
    cleaner = DataCleaner().fit(train_df)
    transformed = cleaner.transform(train_df)

    assert transformed["market_index"].notna().all()
    assert (transformed["market_index"] > 0).all()


def test_find_data_file_finds_standard_and_variant_names(tmp_path):
    (tmp_path / "train-test.csv").write_text("col1,col2\n1,2\n")
    (tmp_path / "validation (1).csv").write_text("col1,col2\n3,4\n")

    assert find_data_file(tmp_path, "train_test").name == "train-test.csv"
    assert find_data_file(tmp_path, "validation").name == "validation (1).csv"

    with pytest.raises(FileNotFoundError):
        find_data_file(tmp_path, "non_existent_file")
