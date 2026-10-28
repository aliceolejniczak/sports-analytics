# Modeling choices

- Performance target: a 0-100 Hooper-style index combining readiness, sleep score, fatigue, soreness and stress.
- Performance model: Random Forest regression, selected with time-series cross-validation.
- Injury target: whether an injury occurs during the next 3 days.
- Injury model: XGBoost classification, with class weighting because injury is less frequent.
- Predictors are shifted by one day to avoid using same-day measurements that directly define the performance index.
- Regression metrics: MAE, RMSE and R2. Classification metrics: balanced accuracy, precision, recall, F1, ROC-AUC, PR-AUC and confusion matrix.
