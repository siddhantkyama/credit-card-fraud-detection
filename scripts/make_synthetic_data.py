"""Create a small FAKE data set with the same columns as the Kaggle file.

Why: the real file is 150 MB and cannot live in Git, but automated tests (CI)
still need something to train on. This lets CI run the real pipeline end to end.
You can also use it to try the project before downloading the real data.

The numbers a model gets on this data mean nothing. Never report them.

    python scripts/make_synthetic_data.py --out data/synthetic.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_synthetic(n_rows: int = 20_000, fraud_rate: float = 0.01, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    is_fraud = (rng.random(n_rows) < fraud_rate).astype(int)

    # 28 anonymised components. Frauds are shifted on a few of them so the
    # models have something to learn, like the real data.
    features = rng.normal(size=(n_rows, 28))
    shift = np.zeros(28)
    shift[[0, 2, 3, 9, 13]] = [-2.0, -2.5, 2.0, -2.0, -3.0]  # V1, V3, V4, V10, V14
    features[is_fraud == 1] += shift

    df = pd.DataFrame(features, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", np.sort(rng.uniform(0, 172_792, n_rows)))  # about 48 hours
    df["Amount"] = np.round(rng.lognormal(mean=3.0, sigma=1.2, size=n_rows), 2)
    df["Class"] = is_fraud
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--out", default="data/synthetic.csv")
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--fraud-rate", type=float, default=0.01)
    args = parser.parse_args()

    data = make_synthetic(args.rows, args.fraud_rate)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.out, index=False)
    print(f"Wrote {len(data):,} rows ({int(data['Class'].sum())} frauds) to {args.out}")
