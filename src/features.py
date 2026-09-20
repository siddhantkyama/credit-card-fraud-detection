"""Feature engineering shared by training AND the API.

Why this file exists: if training and serving prepare data in different ways,
the model sees different inputs in production than it saw during training
("train/serve skew"). Keeping one function used by both removes that risk.
"""
import numpy as np
import pandas as pd

# The dataset's original columns, in the order the API and the CSV upload expect.
PCA_COLUMNS = [f"V{i}" for i in range(1, 29)]
RAW_COLUMNS = ["Time"] + PCA_COLUMNS + ["Amount"]

# What the model actually receives.
FEATURE_COLUMNS = PCA_COLUMNS + ["log_amount", "hour_sin", "hour_cos"]


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw transaction columns into model features.

    Changes compared with the original project:
      * `Time` is "seconds since the first transaction" (only ~48 hours of data).
        Used raw, the model can memorise *when* frauds happened, which does not
        generalise. We keep only time-of-day, encoded as sin/cos so 23:59 and
        00:01 are close together. (The dataset does not say what clock time the
        counter starts at, so treat this as an approximate time-of-day.)
      * `Amount` is heavily skewed, so we use log(1 + Amount).
    """
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")

    out = df[PCA_COLUMNS].astype(float).copy()
    out["log_amount"] = np.log1p(df["Amount"].astype(float).clip(lower=0))

    hour_of_day = (df["Time"].astype(float) % 86_400) / 3_600
    out["hour_sin"] = np.sin(2 * np.pi * hour_of_day / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour_of_day / 24)

    return out[FEATURE_COLUMNS]
