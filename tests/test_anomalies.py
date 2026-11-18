"""Unit tests for LabelAnomalyDetector and anomaly flagging."""
import numpy as np
import pandas as pd
import pytest

from src.anomalies import LabelAnomalyDetector, flag_corrupted_labels


def make_sample_data(n=200):
    rng = np.random.default_rng(42)
    distances = rng.uniform(100, 2000, n)
    equipments = rng.choice(["Dry Van", "Flatbed", "Reefer"], n)
    # Typical rate ~ $2.00 per mile
    rates = distances * 2.0 * rng.normal(1.0, 0.02, n)
    return pd.DataFrame({
        "distance": distances,
        "equipment": equipments,
        "posted_rate": rates,
    })


def test_flag_corrupted_detects_spikes_and_drops():
    df = make_sample_data(300)
    # Inject synthetic corrupted labels
    df.loc[5, "posted_rate"] *= 4.5    # 4.5x spike
    df.loc[12, "posted_rate"] *= 0.25  # severe drop

    flags = flag_corrupted_labels(df)
    assert flags.iloc[5] is True or flags.iloc[5] == 1
    assert flags.iloc[12] is True or flags.iloc[12] == 1
    # Clean records should overwhelmingly not be flagged
    assert flags.sum() <= 4


def test_anomaly_detector_class_fit_and_flag():
    train_df = make_sample_data(300)
    detector = LabelAnomalyDetector().fit(train_df)

    test_df = make_sample_data(20)
    test_df.loc[0, "posted_rate"] *= 5.0
    flags = detector.flag(test_df)
    assert flags.iloc[0] is True or flags.iloc[0] == 1
