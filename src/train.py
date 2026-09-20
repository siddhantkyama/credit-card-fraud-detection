"""Train, compare, (optionally) tune, evaluate and save the fraud model.

Run from the project root (the folder that contains `src/`):

    python -m src.train                 # full run
    python -m src.train --tune          # also tune XGBoost/LightGBM with Optuna
    python -m src.train --fast          # quick smoke test. NOT for reported numbers.

Steps
  1. Load data, drop duplicates, split by TIME (train / validation / test)
  2. Compare several models with 5-fold cross-validation on the training part
  3. (optional) Tune the best boosting model with Optuna
  4. Fit the winner on the training part, choose the decision threshold on the
     validation part by minimising business cost
  5. Report honest numbers on the test part, save model, plots and metrics
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate

from src.data import load_data, time_split
from src.evaluate import (
    choose_threshold, full_report, plot_confusion, plot_cost_curve,
    plot_importance, plot_model_comparison, plot_pr_curve,
)
from src.features import FEATURE_COLUMNS, RAW_COLUMNS, make_features
from src.models import BOOSTERS, COMPARISON_ONLY, SEED, build_models

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data", default=str(ROOT / "data" / "creditcard.csv"))
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--tune", action="store_true", help="tune XGBoost/LightGBM with Optuna")
    p.add_argument("--n-trials", type=int, default=25, help="Optuna trials (with --tune)")
    p.add_argument("--review-cost", type=float, default=5.0,
                   help="cost of an analyst reviewing one alert, in the dataset's currency")
    p.add_argument("--fast", action="store_true",
                   help="use a small subset for a quick check (numbers are NOT reportable)")
    return p.parse_args()


def run_cv(name, model, X, y, folds):
    """Stratified K-fold CV scored by PR-AUC (average precision) and ROC-AUC."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    start = time.time()
    scores = cross_validate(
        model, X, y, cv=cv, n_jobs=1, error_score="raise",
        scoring={"pr_auc": "average_precision", "roc_auc": "roc_auc"},
    )
    return {
        "model": name,
        "pr_auc_mean": float(scores["test_pr_auc"].mean()),
        "pr_auc_std": float(scores["test_pr_auc"].std()),
        "roc_auc_mean": float(scores["test_roc_auc"].mean()),
        "seconds": round(time.time() - start, 1),
    }


def tune_booster(name, X, y, pos_weight, n_trials):
    """Search hyper-parameters with Optuna, maximising 3-fold CV PR-AUC."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

    def objective(trial):
        common = dict(
            n_estimators=trial.suggest_int("n_estimators", 150, 500),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.5, 10.0, log=True),
            scale_pos_weight=trial.suggest_float("scale_pos_weight", 1.0, 100.0, log=True),
        )
        if name == "XGBoost":
            common["max_depth"] = trial.suggest_int("max_depth", 3, 8)
            common["min_child_weight"] = trial.suggest_int("min_child_weight", 1, 10)
        else:  # LightGBM
            common["num_leaves"] = trial.suggest_int("num_leaves", 15, 127)
            common["min_child_samples"] = trial.suggest_int("min_child_samples", 5, 100)
        model = BOOSTERS[name](pos_weight, **common)
        return cross_val_score(model, X, y, cv=cv, scoring="average_precision").mean()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value


def save_samples(test, model, path, per_class=5):
    """Save a few held-out transactions so the web app can offer 'Load an example'."""
    rng = np.random.default_rng(SEED)
    samples = []
    for label in (0, 1):
        pool = test[test["Class"] == label]
        pick = pool.iloc[rng.choice(len(pool), size=min(per_class, len(pool)), replace=False)]
        for _, row in pick.iterrows():
            samples.append({"label": int(label), "values": {c: float(row[c]) for c in RAW_COLUMNS}})
    path.write_text(json.dumps(samples, indent=2))


def markdown_table(results, selected):
    lines = ["| Model | CV PR-AUC (mean ± sd) | CV ROC-AUC | Seconds |", "|---|---|---|---|"]
    for r in sorted(results, key=lambda r: -r["pr_auc_mean"]):
        mark = " **(selected)**" if r["model"] == selected else ""
        lines.append(f"| {r['model']}{mark} | {r['pr_auc_mean']:.4f} ± {r['pr_auc_std']:.4f} "
                     f"| {r['roc_auc_mean']:.4f} | {r['seconds']} |")
    return "\n".join(lines)


def main():
    args = parse_args()
    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)
    t_start = time.time()

    if args.fast:
        print("\n*** FAST MODE: small subset, for checking the code only. ***\n")

    # 1. Data ---------------------------------------------------------------
    df = load_data(args.data)
    train, valid, test = time_split(df)
    for name, part in (("train", train), ("validation", valid), ("test", test)):
        print(f"  {name:<10} {len(part):>8,} rows, {int(part['Class'].sum()):>4} frauds")

    X_train, y_train = make_features(train), train["Class"].to_numpy()
    X_valid, y_valid = make_features(valid), valid["Class"].to_numpy()
    X_test, y_test = make_features(test), test["Class"].to_numpy()

    folds = args.cv_folds
    if args.fast:  # keep all frauds, 15% of normals, 3 folds
        rng = np.random.default_rng(SEED)
        keep = (y_train == 1) | (rng.random(len(y_train)) < 0.15)
        X_train, y_train = X_train[keep], y_train[keep]
        folds = 3

    pos_weight = float(np.sqrt((y_train == 0).sum() / (y_train == 1).sum()))
    print(f"\nscale_pos_weight (sqrt of class ratio): {pos_weight:.1f}")

    # 2. Compare models -----------------------------------------------------
    print(f"\nComparing models with {folds}-fold stratified CV on the training part...")
    models = build_models(pos_weight)
    results = []
    for name, model in models.items():
        r = run_cv(name, model, X_train, y_train, folds)
        results.append(r)
        print(f"  {name:<34} PR-AUC {r['pr_auc_mean']:.4f} ± {r['pr_auc_std']:.4f}   "
              f"ROC-AUC {r['roc_auc_mean']:.4f}   ({r['seconds']}s)")

    # 3. Optional tuning ------------------------------------------------------
    if args.tune:
        boosters = [r for r in results if r["model"] in BOOSTERS]
        base = max(boosters, key=lambda r: r["pr_auc_mean"])["model"]
        n_trials = 3 if args.fast else args.n_trials
        print(f"\nTuning {base} with Optuna ({n_trials} trials)...")
        best_params, best_value = tune_booster(base, X_train, y_train, pos_weight, n_trials)
        print(f"  best 3-fold PR-AUC during search: {best_value:.4f}")
        tuned_name = f"{base} (tuned)"
        tuned = BOOSTERS[base](pos_weight, **best_params)
        models[tuned_name] = tuned
        r = run_cv(tuned_name, tuned, X_train, y_train, folds)
        results.append(r)
        print(f"  {tuned_name:<34} PR-AUC {r['pr_auc_mean']:.4f} ± {r['pr_auc_std']:.4f}")

    # 4. Pick the winner, fit it, choose threshold ----------------------------
    eligible = [r for r in results if r["model"] not in COMPARISON_ONLY]
    selected = max(eligible, key=lambda r: r["pr_auc_mean"])["model"]
    print(f"\nSelected model (best CV PR-AUC): {selected}")
    model = models[selected]
    model.fit(X_train, y_train)

    p_valid = model.predict_proba(X_valid)[:, 1]
    threshold, grid, costs = choose_threshold(
        y_valid, p_valid, valid["Amount"].to_numpy(), args.review_cost)
    print(f"Threshold chosen on validation set (stable low-cost region): {threshold:.3f}")

    # 5. Honest evaluation on the untouched test part -------------------------
    p_test = model.predict_proba(X_test)[:, 1]
    report = full_report(y_test, p_test, test["Amount"].to_numpy(), threshold, args.review_cost)
    t, c = report["at_chosen_threshold"], report["cost_chosen_threshold"]
    lo, hi = report["pr_auc_ci95"]
    print("\nTEST SET RESULTS (data the model and threshold never saw)")
    print(f"  PR-AUC      {report['pr_auc']:.4f}  (95% CI {lo:.3f} to {hi:.3f})")
    print(f"  ROC-AUC     {report['roc_auc']:.4f}")
    print(f"  Precision   {t['precision']:.3f}   Recall {t['recall']:.3f}   F1 {t['f1']:.3f}")
    print(f"  Frauds caught {t['tp']} of {t['tp'] + t['fn']}, false alarms {t['fp']}")
    print(f"  Fraud loss with no model {c['no_model_loss']:,.0f}, with model {c['total_cost']:,.0f} "
          f"(saved {c['saved_pct']:.1f}%)")

    # Save everything ---------------------------------------------------------
    plot_model_comparison(results, selected, OUTPUTS_DIR / "model_comparison.png")
    plot_pr_curve(y_test, p_test, threshold, OUTPUTS_DIR / "pr_curve.png")
    plot_cost_curve(grid, costs, threshold, OUTPUTS_DIR / "cost_curve.png")
    plot_confusion(t, OUTPUTS_DIR / "confusion_matrix.png")

    joblib.dump({
        "model": model,
        "model_name": selected,
        "threshold": threshold,
        "review_cost": args.review_cost,
        "raw_columns": RAW_COLUMNS,
        "feature_columns": FEATURE_COLUMNS,
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, MODELS_DIR / "fraud_model.joblib")

    save_samples(test, model, MODELS_DIR / "samples.json")

    metrics = {
        "mode": "fast" if args.fast else "full",
        "model_name": selected,
        "threshold": threshold,
        "review_cost": args.review_cost,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "splits": {n: {"rows": len(p), "frauds": int(p["Class"].sum())}
                   for n, p in (("train", train), ("validation", valid), ("test", test))},
        "cv_results": sorted(results, key=lambda r: -r["pr_auc_mean"]),
        "test": report,
        "plots": {"comparison": "model_comparison.png", "pr_curve": "pr_curve.png",
                  "cost_curve": "cost_curve.png", "confusion": "confusion_matrix.png",
                  "importance": None},
    }
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    (OUTPUTS_DIR / "results.md").write_text(
        "## Model comparison (5-fold CV on training data)\n\n" + markdown_table(results, selected)
        + f"\n\n## Test set ({report['n_rows']:,} rows, {report['n_frauds']} frauds)\n\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| PR-AUC | {report['pr_auc']:.3f} (95% CI {lo:.3f} to {hi:.3f}) |\n"
        f"| Precision / Recall | {t['precision']:.3f} / {t['recall']:.3f} |\n"
        f"| Frauds caught | {t['tp']} of {t['tp'] + t['fn']} |\n"
        f"| False alarms | {t['fp']} |\n"
        f"| Fraud loss avoided | {c['saved_pct']:.1f}% |\n")

    print("\nSaved model, metrics, plots and results.md.")

    # Optional and last: a feature-importance chart. If this step ever fails (for
    # example, low memory), everything above is already safely saved.
    try:
        frauds = X_test[y_test == 1]
        normals = X_test[y_test == 0].sample(n=min(800, int((y_test == 0).sum())), random_state=SEED)
        kind = plot_importance(model, pd.concat([frauds, normals]), OUTPUTS_DIR / "feature_importance.png")
        if kind != "none":
            metrics["plots"]["importance"] = "feature_importance.png"
            (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    except Exception as exc:
        print(f"  (feature importance chart skipped: {exc})")

    print(f"Total time {time.time() - t_start:.0f}s")
    print("Next: uvicorn api.main:app --reload   then open http://127.0.0.1:8000")


if __name__ == "__main__":
    main()
