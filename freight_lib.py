"""Shared code for the freight-rate project (cleaning, features, models, metrics)."""
import glob, os
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge

TARGET = "posted_rate"
EQUIPMENT = ["Dry Van", "Flatbed", "Reefer"]
START_DATE = pd.Timestamp("2025-01-01")
DIST_BINS = [0, 150, 300, 500, 800, 1200, 2000, 5000]
DEFAULT_GBM = dict(n_estimators=300, learning_rate=0.05, num_leaves=15,
                   min_child_samples=50, subsample=0.8, subsample_freq=1,
                   colsample_bytree=0.8, random_state=0, verbose=-1)
STATIC = ["log_dist", "weight", "equip"]

FOLDS = {"V1": ("2025-07-01", "2025-08-31"), "V2": ("2025-08-01", "2025-09-30"),
         "V3": ("2025-09-01", "2025-10-31"), "V4": ("2025-10-01", "2025-10-31")}


def find(data_dir, stem):
    """Find train_test.csv / train-test.csv / 'train-test (1).csv'."""
    data_dir = Path(data_dir)
    names = [stem, stem.replace("_", "-")]
    for n in names:
        if (data_dir / f"{n}.csv").is_file():
            return str(data_dir / f"{n}.csv")
    for n in names:
        hits = sorted(glob.glob(str(data_dir / f"{n} (*).csv")))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"Could not find {stem}.csv in {data_dir}.")


class Cleaner:
    """Fix weights and market_index. Fill values are learned from train only."""
    def fit(self, df):
        w = df["weight"].abs()
        self.weight_by_equipment = w.groupby(df["equipment"]).median().to_dict()
        self.weight_overall = w.median()
        return self

    def transform(self, df, market_by_day):
        out = df.copy()
        out["weight"] = out["weight"].abs()
        out["weight"] = (out["weight"]
                         .fillna(out["equipment"].map(self.weight_by_equipment))
                         .fillna(self.weight_overall))
        if "market_index" in out:
            out["market_index"] = out["market_index"].fillna(out["date"].map(market_by_day))
        else:
            out["market_index"] = out["date"].map(market_by_day)
        return out


def flag_corrupted(df, low=0.5, high=2.0):
    """True where $/mile is far from the typical value for its equipment x distance band."""
    bins = pd.cut(df["distance"], DIST_BINS)
    rpm = df[TARGET] / df["distance"]
    typical = rpm.groupby([df["equipment"], bins]).transform("median")
    ratio = rpm / typical
    return (ratio < low) | (ratio > high)


def make_features(df):
    X = pd.DataFrame(index=df.index)
    X["log_dist"] = np.log(df["distance"])
    X["weight"] = df["weight"]
    X["equip"] = pd.Categorical(df["equipment"], categories=EQUIPMENT).codes
    X["market"] = df["market_index"]
    X["t"] = (df["date"] - START_DATE).dt.days / 30.0
    if "quote_signal" in df:
        X["quote"] = df["quote_signal"]
    return X


def score(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    err = np.abs(y - p)
    return {"MAE": err.mean(), "RMSE": np.sqrt(((y - p) ** 2).mean()),
            "MAPE": (err / y).mean() * 100, "WAPE": err.sum() / y.sum() * 100}


class LookupBaseline:
    def fit(self, df):
        bins = pd.cut(df["distance"], DIST_BINS)
        self.table = (df[TARGET] / df["distance"]).groupby([df["equipment"], bins]).median()
        return self

    def predict(self, df):
        bins = pd.cut(df["distance"], DIST_BINS)
        keys = pd.MultiIndex.from_arrays([df["equipment"], bins])
        return self.table.reindex(keys).fillna(self.table.median()).values * df["distance"].values


class Hybrid:
    """Step 1: Ridge line on market (and optionally time). Step 2: GBM on the leftovers."""
    def __init__(self, linear_cols, gbm_cols=STATIC, alpha=1.0, gbm_params=None):
        self.linear_cols, self.gbm_cols, self.alpha = linear_cols, gbm_cols, alpha
        self.gbm_params = {**DEFAULT_GBM, **(gbm_params or {})}

    def _linear_X(self, df):
        X = make_features(df)
        out = X[self.linear_cols].copy()
        for code in (1, 2):
            out[f"equip_{code}"] = (X["equip"] == code).astype(int)
        return out

    def fit(self, df):
        y = np.log(df[TARGET])
        X = self._linear_X(df)
        self.mean, self.std = X.mean(), X.std() + 1e-9
        self.linear = Ridge(alpha=self.alpha).fit((X - self.mean) / self.std, y)
        residual = y - self.linear.predict((X - self.mean) / self.std)
        self.gbm = lgb.LGBMRegressor(**self.gbm_params).fit(make_features(df)[self.gbm_cols], residual)
        return self

    def predict_log(self, df):
        X = (self._linear_X(df) - self.mean) / self.std
        return self.linear.predict(X) + self.gbm.predict(make_features(df)[self.gbm_cols])

    def predict(self, df):
        return np.exp(self.predict_log(df))


class Blend:
    """Average of several models in log space (a geometric mean)."""
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights or [1 / len(models)] * len(models)

    def fit(self, df):
        for m in self.models:
            m.fit(df)
        return self

    def predict(self, df):
        logs = sum(w * m.predict_log(df) for w, m in zip(self.weights, self.models))
        return np.exp(logs)
