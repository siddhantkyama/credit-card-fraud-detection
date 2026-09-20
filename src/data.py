"""Loading the dataset and splitting it the way a real deployment would see it."""
from pathlib import Path

import pandas as pd


def load_data(path: str | Path) -> pd.DataFrame:
    """Read the CSV, drop duplicate rows and sort by time.

    The Kaggle file has ~1,000 exact duplicate rows. With a random split, the
    same transaction can land in both train and test, which inflates results.
    """
    df = pd.read_csv(path)
    if "Class" not in df.columns:
        raise ValueError("Expected a 'Class' column (0 = normal, 1 = fraud).")

    before = len(df)
    df = df.drop_duplicates()
    df = df.sort_values("Time", kind="mergesort").reset_index(drop=True)
    print(f"Loaded {before:,} rows, dropped {before - len(df):,} duplicates -> {len(df):,} rows")
    return df


def time_split(
    df: pd.DataFrame, train_frac: float = 0.70, valid_frac: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by time: oldest 70% -> train, next 15% -> validation, newest 15% -> test.

    In production a model is trained on the past and scores the future, so the
    test set must come *after* the training data. A random split hides problems
    such as fraud patterns changing over time.

      train      : fit models, cross-validation
      validation : choose the decision threshold
      test       : touched once, for the final honest numbers
    """
    n = len(df)
    i_train = int(n * train_frac)
    i_valid = int(n * (train_frac + valid_frac))
    return df.iloc[:i_train], df.iloc[i_train:i_valid], df.iloc[i_valid:]
