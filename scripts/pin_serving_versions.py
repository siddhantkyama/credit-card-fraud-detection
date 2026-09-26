"""Write requirements-serve.txt with the EXACT library versions installed right now.

Why: a trained model is a pickle file. It can only be loaded reliably with the
same library versions that trained it. The Docker image installs from this file,
so run this script in the same virtual environment you trained in, then commit
the file together with models/fraud_model.joblib.

    python scripts/pin_serving_versions.py
"""
import argparse
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

# Only what is needed to LOAD the model and SERVE it (no plotting, tuning or tests).
SERVING_PACKAGES = [
    "numpy", "pandas", "scikit-learn", "imbalanced-learn",
    "xgboost", "lightgbm", "joblib",
    "fastapi", "uvicorn", "python-multipart",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--out", default="requirements-serve.txt")
    args = parser.parse_args()

    lines = ["# Exact versions used to train the deployed model (made by scripts/pin_serving_versions.py)"]
    for name in SERVING_PACKAGES:
        try:
            lines.append(f"{name}=={version(name)}")
        except PackageNotFoundError:
            raise SystemExit(f"'{name}' is not installed here. Activate your training venv first.")

    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.out}:\n" + "\n".join(lines))
