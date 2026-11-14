"""Unit tests for the building blocks in freight_lib.py (tiny synthetic data, no CSVs needed)."""
import numpy as np
import pandas as pd
import pytest

from freight_lib import (Blend, Cleaner, Hybrid, LookupBaseline, find,
                         flag_corrupted, make_features, score, TARGET)


def make_loads(n=600, seed=0):
    """Small fake dataset with the same columns as the real one."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=300, freq="D")
    df = pd.DataFrame({
        "date": rng.choice(dates, n),
        "distance": rng.integers(50, 2500, n).astype(float),
        "equipment": rng.choice(["Dry Van", "Flatbed", "Reefer"], n),
        "weight": rng.integers(10_000, 45_000, n).astype(float),
    })
    df["market_index"] = 1.0 + 0.1 * np.sin(df["date"].dt.dayofyear / 20)
    df[TARGET] = df["distance"] * 2.0 * df["market_index"] * rng.normal(1, 0.02, n)
    return df


# ---------- metrics ----------
def test_score_matches_hand_calculation():
    # actual 500/1000/2000, predicted 450/1100/1900 -> errors 50/100/100
    result = score([500, 1000, 2000], [450, 1100, 1900])
    assert result["MAE"] == pytest.approx(250 / 3)
    assert result["WAPE"] == pytest.approx(250 / 3500 * 100)
    assert result["MAPE"] == pytest.approx((10 + 10 + 5) / 3)


# ---------- cleaning ----------
def test_cleaner_fixes_negative_and_missing_weight():
    df = make_loads(50)
    df.loc[0, "weight"] = -20_000.0
    df.loc[1, "weight"] = np.nan
    cleaner = Cleaner().fit(df)
    out = cleaner.transform(df, df.groupby("date")["market_index"].median())
    assert (out["weight"] > 0).all()
    assert out["weight"].notna().all()
    assert out.loc[0, "weight"] == 20_000.0           # abs() applied


def test_cleaner_fills_missing_market_index_from_same_day():
    df = make_loads(50)
    by_day = df.groupby("date")["market_index"].median()
    day = df.loc[0, "date"]
    expected = by_day[day]
    df.loc[0, "market_index"] = np.nan
    out = Cleaner().fit(df).transform(df, by_day)
    assert out.loc[0, "market_index"] == pytest.approx(expected)


def test_flag_corrupted_catches_extreme_rates_only():
    df = make_loads()
    df.loc[0, TARGET] *= 4.0          # far too high
    df.loc[1, TARGET] *= 0.28         # far too low
    flags = flag_corrupted(df)
    assert flags.iloc[0] and flags.iloc[1]
    assert flags.sum() <= 4           # clean rows are (almost) never flagged


# ---------- features ----------
def test_make_features_columns_and_equipment_codes():
    X = make_features(make_loads(20))
    assert {"log_dist", "weight", "equip", "market", "t"} <= set(X.columns)
    assert set(X["equip"]) <= {0, 1, 2}


# ---------- models ----------
def test_hybrid_predicts_positive_finite_and_sensible_values():
    df = make_loads()
    model = Hybrid(["log_dist", "market", "t"]).fit(df)
    pred = model.predict(df)
    assert len(pred) == len(df)
    assert np.isfinite(pred).all() and (pred > 0).all()
    assert score(df[TARGET], pred)["WAPE"] < 10      # learns the simple pattern


def test_blend_with_equal_models_equals_single_model():
    df = make_loads()
    single = Hybrid(["log_dist", "market"]).fit(df).predict(df)
    blend = Blend([Hybrid(["log_dist", "market"]), Hybrid(["log_dist", "market"])], [0.7, 0.3]).fit(df)
    assert blend.predict(df) == pytest.approx(single)


def test_lookup_baseline_predicts_for_every_row():
    df = make_loads()
    pred = LookupBaseline().fit(df).predict(df)
    assert len(pred) == len(df) and np.isfinite(pred).all()


# ---------- file finder ----------
def test_find_accepts_hyphen_and_browser_copy_names(tmp_path):
    (tmp_path / "train-test.csv").write_text("a\n1\n")
    (tmp_path / "validation (1).csv").write_text("a\n1\n")
    assert find(tmp_path, "train_test").endswith("train-test.csv")
    assert find(tmp_path, "validation").endswith("validation (1).csv")


def test_find_prefers_exact_name_over_similar_file(tmp_path):
    (tmp_path / "validation.csv").write_text("a\n1\n")
    (tmp_path / "validation-predictions-template.csv").write_text("a\n1\n")
    assert find(tmp_path, "validation").endswith("validation.csv")


def test_find_raises_clear_error_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        find(tmp_path, "train_test")
