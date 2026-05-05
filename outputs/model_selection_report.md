# Model selection and tuning

**Generated (UTC):** 2026-05-05T09:44:17.006884+00:00

## Data

- Features: `data/ml_X.csv` (303 rows × 28 columns)
- Target: `target_binary` from `data/ml_y.csv` (binary: disease vs no disease)
- Split for tuning: 80% train / 20% stratified holdout (`random_state=42`)

## Methodology

1. **Cross-validation (defaults)** — Each baseline model is evaluated with
   **StratifiedKFold(5)** on the **full** dataset. Reported metrics:
   **accuracy**, **precision** (binary), **recall** (binary), **ROC-AUC**
   (probability or decision function). Values are **mean ± std** across folds.
2. **Hyperparameter tuning** — **RandomizedSearchCV** (`n_iter` capped by grid size,
   max 30) on the **training split only**, same folds, primary metric
   **ROC-AUC**. Best params are refit on all training data, then evaluated once on
   the **held-out test set** (accuracy, precision, recall, ROC-AUC).

## 1. Cross-validation — baseline hyperparameters

| model | accuracy_mean | accuracy_std | precision_mean | precision_std | recall_mean | recall_std | roc_auc_mean | roc_auc_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.8547 | 0.0485 | 0.8744 | 0.0572 | 0.7979 | 0.0727 | 0.9186 | 0.0208 |
| Decision Tree | 0.7985 | 0.0467 | 0.8036 | 0.0706 | 0.7481 | 0.0453 | 0.8160 | 0.0375 |
| Random Forest | 0.8317 | 0.0114 | 0.8321 | 0.0442 | 0.7984 | 0.0295 | 0.9151 | 0.0235 |
| SVM | 0.8480 | 0.0414 | 0.8517 | 0.0577 | 0.8127 | 0.0534 | 0.9028 | 0.0291 |

## 2. Tuning — search spaces (summary)

| Model | Search dimensions |
|-------|-------------------|
| Logistic Regression | `C` (log-spaced 1e-2 … 1e2) |
| Decision Tree | `max_depth`, `min_samples_leaf`, `min_samples_split` |
| Random Forest | `n_estimators`, `max_depth`, `min_samples_leaf` |
| SVM (RBF) | `C`, `gamma` |

## 3. Best parameters (train CV, optimize ROC-AUC)

| model | best_cv_roc_auc | best_params |
| --- | --- | --- |
| Logistic Regression | 0.9038 | {'C': 0.2848035868435802} |
| Decision Tree | 0.8430 | {'min_samples_split': 8, 'min_samples_leaf': 12, 'max_depth': 15} |
| Random Forest | 0.9023 | {'n_estimators': 150, 'min_samples_leaf': 6, 'max_depth': None} |
| SVM | 0.9009 | {'gamma': 0.001, 'C': 16.0} |

## 4. Holdout test performance (after tuning)

| model | accuracy | precision | recall | roc_auc |
| --- | --- | --- | --- | --- |
| Logistic Regression | 0.8852 | 0.8387 | 0.9286 | 0.9675 |
| Decision Tree | 0.8361 | 0.8214 | 0.8214 | 0.8588 |
| Random Forest | 0.8525 | 0.8065 | 0.8929 | 0.9481 |
| SVM | 0.8525 | 0.8276 | 0.8571 | 0.9643 |


*Holdout metrics use a single stratified 20% split; for low sample sizes, confidence intervals are wide. Prefer CV means for model comparison.*
