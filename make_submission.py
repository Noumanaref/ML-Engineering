"""Train the final model on all clean training data and write the submission files.

Usage:
    python make_submission.py --data-dir data --out-dir .
Outputs (written to --out-dir):
    validation_predictions.csv      load_id,predicted_rate
    december_chart_inputs.csv       the December file with predicted_rate filled in
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from freight_lib import (find, Cleaner, flag_corrupted, Hybrid, Blend, TARGET)

WITH_TIME = ["log_dist", "market", "t"]
NO_TIME = ["log_dist", "market"]
BLEND_WEIGHTS = [0.7, 0.3]          # with-time, no-time (chosen in notebook 03b)
DEFAULT_EXPECTED_ROWS = 12_000      # size of the real validation set


def build_model():
    return Blend([Hybrid(WITH_TIME), Hybrid(NO_TIME)], BLEND_WEIGHTS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--expected-rows", type=int, default=DEFAULT_EXPECTED_ROWS,
                        help="Row count the validation set must have (default 12000).")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load
    train = pd.read_csv(find(args.data_dir, "train_test"), parse_dates=["date"])
    val = pd.read_csv(find(args.data_dir, "validation"), parse_dates=["date"])
    dec = pd.read_csv(find(args.data_dir, "december_chart_inputs"), parse_dates=["date"])
    template = pd.read_csv(find(args.data_dir, "validation_predictions_template"))

    # 2. Clean (fill values learned from train only; market_index looked up by day)
    market_by_day = pd.concat([train, val]).groupby("date")["market_index"].median()
    cleaner = Cleaner().fit(train)
    train_c = cleaner.transform(train, market_by_day)
    val_c = cleaner.transform(val, market_by_day)
    dec_c = cleaner.transform(dec.drop(columns="predicted_rate"), market_by_day)

    # 3. Drop corrupted labels, then train on every clean row (Jan-Oct)
    train_c["is_corrupted"] = flag_corrupted(train_c)
    clean = train_c[~train_c["is_corrupted"]]
    print(f"Training on {len(clean):,} clean rows ({train_c['is_corrupted'].sum()} dropped).")
    model = build_model().fit(clean)

    # 4. Predict validation, in template order
    val_pred = pd.DataFrame({"load_id": val["load_id"], "predicted_rate": model.predict(val_c).round(2)})
    submission = template[["load_id"]].merge(val_pred, on="load_id", how="left")
    assert len(submission) == args.expected_rows, f"expected {args.expected_rows} rows, got {len(submission)}"
    assert submission["predicted_rate"].notna().all(), "missing predictions"
    assert (submission["predicted_rate"] > 0).all(), "non-positive predictions"
    submission.to_csv(out_dir / "validation_predictions.csv", index=False)

    # 5. Predict December (same model)
    dec_out = dec.copy()
    dec_out["predicted_rate"] = model.predict(dec_c).round(2)
    dec_out["date"] = dec_out["date"].dt.strftime("%Y-%m-%d")
    dec_out.to_csv(out_dir / "december_chart_inputs.csv", index=False)

    print("Wrote validation_predictions.csv and december_chart_inputs.csv to", out_dir)


if __name__ == "__main__":
    main()
