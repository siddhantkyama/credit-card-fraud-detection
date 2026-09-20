"""Evaluation for a rare-event problem: PR-AUC, cost-based threshold, CIs and plots.

Accuracy is useless here (predicting "normal" for everything scores 99.8%).
What matters: how many frauds we catch (recall), how many alerts are wrong
(precision), and what it costs the business.
"""
import matplotlib

matplotlib.use("Agg")  # draw to files; no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, confusion_matrix, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)

INK = "#14201B"
RED = "#B42318"
GREEN = "#146C4B"
GREY = "#9AA8A0"


# --------------------------------------------------------------------------- #
# Business cost
# --------------------------------------------------------------------------- #
def business_cost(y, p, amount, threshold: float, review_cost: float) -> dict:
    """What a threshold costs.

    * A fraud we miss costs the transaction amount.
    * Every alert (right or wrong) costs `review_cost` for an analyst to check.
    * "No model" means we miss every fraud, so we lose all fraud amounts.
    """
    y, p, amount = np.asarray(y), np.asarray(p), np.asarray(amount)
    alert = p >= threshold
    missed_loss = float(amount[(y == 1) & ~alert].sum())
    review_total = float(review_cost * alert.sum())
    total = missed_loss + review_total
    no_model = float(amount[y == 1].sum())
    return {
        "total_cost": total,
        "missed_fraud_loss": missed_loss,
        "review_cost_total": review_total,
        "no_model_loss": no_model,
        "saved": no_model - total,
        "saved_pct": (no_model - total) / no_model * 100 if no_model else 0.0,
    }


def choose_threshold(y, p, amount, review_cost: float):
    """Scan thresholds on the VALIDATION set and pick a cheap, stable one.

    Costs are averaged over a +/-0.05 window of thresholds. The validation set has
    only ~55-70 frauds, so one transaction can create a sudden cliff in the raw
    cost curve; averaging keeps us away from such cliff edges.
    """
    grid = np.round(np.linspace(0.005, 0.995, 199), 3)
    costs = np.array([business_cost(y, p, amount, t, review_cost)["total_cost"] for t in grid])
    smooth = pd.Series(costs).rolling(21, center=True, min_periods=1).mean().to_numpy()
    # Take the MIDDLE of the near-optimal plateau (within 2% of the lowest cost)
    # rather than its edge, so a small shift in the data cannot push us off a cliff.
    plateau = np.where(smooth <= smooth.min() * 1.02)[0]
    best = int(plateau[np.argmin(np.abs(plateau - np.median(plateau)))])
    return float(grid[best]), grid, costs


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def threshold_metrics(y, p, threshold: float) -> dict:
    pred = (np.asarray(p) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = precision_score(y, pred, zero_division=0)
    recall = recall_score(y, pred, zero_division=0)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "threshold": float(threshold), "precision": float(precision),
        "recall": float(recall), "f1": float(f1),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def bootstrap_ci(y, p, metric, n_boot: int = 500, seed: int = 42):
    """95% confidence interval by resampling. With ~70 frauds the test numbers
    are noisy, and reporting the range is more honest than one number."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() == 0:
            continue
        values.append(metric(y[idx], p[idx]))
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def full_report(y, p, amount, threshold: float, review_cost: float) -> dict:
    """Everything we report on the test set."""
    at_thr = threshold_metrics(y, p, threshold)
    at_half = threshold_metrics(y, p, 0.5)
    pr_auc = float(average_precision_score(y, p))
    return {
        "pr_auc": pr_auc,
        "pr_auc_ci95": bootstrap_ci(y, p, average_precision_score),
        "roc_auc": float(roc_auc_score(y, p)),
        "at_chosen_threshold": at_thr,
        "recall_ci95": bootstrap_ci(
            y, p, lambda yt, pt: recall_score(yt, pt >= threshold, zero_division=0)),
        "at_threshold_0.5": at_half,
        "cost_chosen_threshold": business_cost(y, p, amount, threshold, review_cost),
        "cost_threshold_0.5": business_cost(y, p, amount, 0.5, review_cost),
        "n_rows": int(len(y)),
        "n_frauds": int(np.sum(y)),
    }


# --------------------------------------------------------------------------- #
# Plots (saved as PNG files for the README and the frontend's Model report tab)
# --------------------------------------------------------------------------- #
def _finish(path):
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_model_comparison(results: list[dict], selected: str, path):
    rows = sorted(results, key=lambda r: r["pr_auc_mean"])
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(rows) + 1.2))
    colors = [INK if r["model"] == selected else GREY for r in rows]
    ax.barh([r["model"] for r in rows], [r["pr_auc_mean"] for r in rows],
            xerr=[r["pr_auc_std"] for r in rows], color=colors, capsize=3)
    ax.set_xlabel("Cross-validated PR-AUC (higher is better)")
    ax.set_title("Model comparison (training data, 5-fold CV)")
    ax.spines[["top", "right"]].set_visible(False)
    _finish(path)


def plot_pr_curve(y, p, threshold, path):
    precision, recall, _ = precision_recall_curve(y, p)
    point = threshold_metrics(y, p, threshold)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(recall, precision, color=INK, lw=2)
    ax.scatter([point["recall"]], [point["precision"]], color=RED, zorder=3,
               label=f"Chosen threshold ({threshold:.2f})")
    ax.axhline(np.mean(y), color=GREY, ls="--", lw=1, label="Random guessing")
    ax.set_xlabel("Recall (share of frauds caught)")
    ax.set_ylabel("Precision (share of alerts that are fraud)")
    ax.set_title(f"Precision-recall curve, test set (PR-AUC {average_precision_score(y, p):.3f})")
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    _finish(path)


def plot_cost_curve(grid, costs, best_threshold, path):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(grid, costs, color=INK, lw=2)
    ax.axvline(best_threshold, color=RED, ls="--", label=f"Chosen threshold ({best_threshold:.2f})")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Total cost on validation set")
    ax.set_title("Cost by threshold (missed fraud + review effort)")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    _finish(path)


def plot_confusion(metrics: dict, path):
    cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm, cmap="Greens")
    labels = [["Normal, approved", "Normal, wrongly flagged"], ["Fraud, missed", "Fraud, caught"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n{labels[i][j]}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else INK, fontsize=9)
    ax.set_xticks([0, 1], ["Predicted normal", "Predicted fraud"])
    ax.set_yticks([0, 1], ["Actually normal", "Actually fraud"])
    ax.set_title(f"Confusion matrix, test set (threshold {metrics['threshold']:.2f})")
    _finish(path)


def plot_importance(model, X_sample: pd.DataFrame, path) -> str:
    """SHAP summary if possible, else plain feature importances. Returns what was drawn.

    Note: V1-V28 are anonymised PCA components, so this shows WHICH inputs matter
    but cannot say what they mean. That limit is worth stating in interviews.
    """
    estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    try:
        import shap

        values = shap.TreeExplainer(estimator).shap_values(X_sample)
        if isinstance(values, list):
            values = values[-1]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, -1]
        shap.summary_plot(values, X_sample, show=False, max_display=15)
        _finish(path)
        return "shap"
    except Exception as exc:  # SHAP missing or model not tree-based
        print(f"  (SHAP skipped: {exc})")

    if hasattr(estimator, "feature_importances_"):
        imp = pd.Series(estimator.feature_importances_, index=X_sample.columns).nlargest(15)
        fig, ax = plt.subplots(figsize=(6, 5))
        imp[::-1].plot.barh(ax=ax, color=INK)
        ax.set_title("Top 15 feature importances")
        ax.spines[["top", "right"]].set_visible(False)
        _finish(path)
        return "importances"
    return "none"
