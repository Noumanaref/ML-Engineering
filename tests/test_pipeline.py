"""End-to-end smoke test: tiny fake data -> make_submission.py -> correct output files."""
import sys

import numpy as np
import pandas as pd

import make_submission
from freight_lib import TARGET


def _write_fake_data(folder, n_train=1200, n_val=40):
    rng = np.random.default_rng(1)

    def loads(n, start, days):
        dates = pd.date_range(start, periods=days, freq="D")
        df = pd.DataFrame({
            "pickup": "A", "delivery": "B",
            "distance": rng.integers(50, 2500, n).astype(float),
            "equipment": rng.choice(["Dry Van", "Flatbed", "Reefer"], n),
            "weight": rng.integers(10_000, 45_000, n).astype(float),
            "date": rng.choice(dates, n),
        })
        df["market_index"] = 1.0
        df["quote_signal"] = 2.0
        return df

    train = loads(n_train, "2025-01-01", 300)
    train[TARGET] = train["distance"] * 2.0 * rng.normal(1, 0.02, n_train)
    val = loads(n_val, "2025-11-01", 60)
    val.insert(0, "load_id", [f"TE-{i:06d}" for i in range(1, n_val + 1)])
    template = val[["load_id"]].assign(predicted_rate=np.nan)

    december = pd.DataFrame({
        "pickup": "Lexington", "delivery": "Fort Wayne", "distance": 360,
        "equipment": "Dry Van", "weight": 32000,
        "date": pd.date_range("2025-12-01", "2025-12-31").strftime("%Y-%m-%d"),
        "predicted_rate": np.nan})
    # the real validation file covers December days too, so the market lookup works
    val["date"] = pd.date_range("2025-11-20", periods=n_val, freq="D")
    dec_market = pd.DataFrame({"load_id": [f"X{i}" for i in range(31)], "pickup": "A", "delivery": "B",
                               "distance": 500.0, "equipment": "Dry Van", "weight": 30000.0,
                               "date": pd.date_range("2025-12-01", "2025-12-31"),
                               "market_index": 1.0, "quote_signal": 2.0})
    val = pd.concat([val, dec_market], ignore_index=True)
    template = pd.concat([template, pd.DataFrame({"load_id": dec_market["load_id"], "predicted_rate": np.nan})],
                         ignore_index=True)

    train.to_csv(folder / "train_test.csv", index=False)
    val.to_csv(folder / "validation.csv", index=False)
    template.to_csv(folder / "validation_predictions_template.csv", index=False)
    december.to_csv(folder / "december_chart_inputs.csv", index=False)
    return len(val)


def test_make_submission_writes_valid_files(tmp_path, monkeypatch):
    data_dir, out_dir = tmp_path / "data", tmp_path / "out"
    data_dir.mkdir()
    n_val = _write_fake_data(data_dir)

    monkeypatch.setattr(sys, "argv", ["make_submission.py", "--data-dir", str(data_dir),
                                      "--out-dir", str(out_dir), "--expected-rows", str(n_val)])
    make_submission.main()

    preds = pd.read_csv(out_dir / "validation_predictions.csv")
    assert list(preds.columns) == ["load_id", "predicted_rate"]
    assert len(preds) == n_val
    assert preds["load_id"].is_unique
    assert (preds["predicted_rate"] > 0).all()

    dec = pd.read_csv(out_dir / "december_chart_inputs.csv")
    assert list(dec.columns) == ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]
    assert len(dec) == 31 and (dec["predicted_rate"] > 0).all()
