# modelTraining.py — holdout + cross-validation

**Generated (UTC):** 2026-05-05T09:43:49.494757+00:00

## Stratified cross-validation

Same hyperparameters as `build_models()`, **5-fold** stratified CV on the **full** `ml_X` / `target_binary` data. Metrics are **mean ± std** over folds.

| model | accuracy_mean | accuracy_std | precision_mean | precision_std | recall_mean | recall_std | roc_auc_mean | roc_auc_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.8547 | 0.0485 | 0.8744 | 0.0572 | 0.7979 | 0.0727 | 0.9186 | 0.0208 |
| Random Forest | 0.8317 | 0.0114 | 0.8321 | 0.0442 | 0.7984 | 0.0295 | 0.9151 | 0.0235 |
| SVM | 0.8480 | 0.0414 | 0.8517 | 0.0577 | 0.8127 | 0.0534 | 0.9028 | 0.0291 |
| Decision Tree | 0.7985 | 0.0467 | 0.8036 | 0.0706 | 0.7481 | 0.0453 | 0.8160 | 0.0375 |

## Holdout test split (20%, stratified)

Single split metrics from the same run (models fitted on the 80% train portion):

| model | accuracy | precision | recall | roc_auc | f1 |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.8852 | 0.8387 | 0.9286 | 0.9665 | 0.8814 |
| SVM | 0.8852 | 0.8387 | 0.9286 | 0.9643 | 0.8814 |
| Random Forest | 0.8689 | 0.8125 | 0.9286 | 0.9443 | 0.8667 |
| Decision Tree | 0.7377 | 0.6667 | 0.8571 | 0.8544 | 0.7500 |
