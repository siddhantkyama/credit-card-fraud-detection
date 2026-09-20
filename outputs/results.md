## Model comparison (5-fold CV on training data)

| Model | CV PR-AUC (mean ± sd) | CV ROC-AUC | Seconds |
|---|---|---|---|
| XGBoost **(selected)** | 0.8701 ± 0.0357 | 0.9798 | 16.5 |
| XGBoost (tuned) | 0.8682 ± 0.0377 | 0.9794 | 25.2 |
| LightGBM + SMOTE | 0.8588 ± 0.0349 | 0.9786 | 16.7 |
| Random Forest + SMOTE | 0.8539 ± 0.0351 | 0.9750 | 109.2 |
| Random Forest | 0.8469 ± 0.0383 | 0.9601 | 139.1 |
| Logistic Regression | 0.7308 ± 0.0528 | 0.9786 | 3.9 |
| LightGBM | 0.6346 ± 0.0697 | 0.9161 | 9.8 |
| Isolation Forest (unsupervised) | 0.2363 ± 0.0284 | 0.9533 | 6.6 |

## Test set (42,559 rows, 52 frauds)

| Metric | Value |
|---|---|
| PR-AUC | 0.765 (95% CI 0.644 to 0.869) |
| Precision / Recall | 0.750 / 0.750 |
| Frauds caught | 39 of 52 |
| False alarms | 13 |
| Fraud loss avoided | 57.3% |
