"""Unit tests for HybridFreightModel and GeometricBlendModel."""
import numpy as np
import pandas as pd
import pytest

from src.models import GeometricBlendModel, HybridFreightModel
from src.metrics import compute_metrics


def make_synthetic_loads(n=400, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=180, freq="D")
    df = pd.DataFrame({
        "distance": rng.uniform(100, 2000, n),
        "weight": rng.uniform(15000, 45000, n),
        "equipment": rng.choice(["Dry Van", "Flatbed", "Reefer"], n),
        "date": rng.choice(dates, n),
        "market_index": rng.uniform(0.90, 1.10, n),
    })
    # Realistic base rate: ~$2.20/mile with scale economies and market multiplier
    rate = df["distance"] * 2.20 * df["market_index"] * rng.normal(1.0, 0.02, n)
    df["posted_rate"] = rate
    return df


def test_hybrid_freight_model_fit_and_predict():
    df = make_synthetic_loads(300)
    model = HybridFreightModel(include_time=True)
    model.fit(df)
    preds = model.predict(df)

    assert len(preds) == len(df)
    assert np.isfinite(preds).all()
    assert (preds > 0).all()

    metrics = compute_metrics(df["posted_rate"], preds)
    assert metrics["WAPE"] < 10.0


def test_geometric_blend_model_combines_predictions():
    df = make_synthetic_loads(300)
    m1 = HybridFreightModel(include_time=True)
    m2 = HybridFreightModel(include_time=False)
    blend = GeometricBlendModel(models=[m1, m2], weights=[0.70, 0.30])
    blend.fit(df)

    preds = blend.predict(df)
    assert len(preds) == len(df)
    assert (preds > 0).all()
