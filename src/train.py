import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, accuracy_score,
    balanced_accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

DATA_FILE = "data/processed/model_dataset.csv"
REPORT_DIR = "reports/modeling"

def prepare_data(df):
    """Prépare un scénario prospectif : les prédicteurs sont décalés d'un jour."""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['participant_id', 'date'])
    
    cols_to_drop = [
        'participant_id', 'date', 'performance_index', 'is_injured',
        'injury_next_3_days', 'gdoc_meals', 'detected_food',
        'calorie_estimation_confidence'
    ]
    features = [c for c in df.columns if c not in cols_to_drop]
    numeric_features = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    lagged_features = df.groupby('participant_id')[numeric_features].shift(1)
    lagged_features.columns = [f'{column}_previous_day' for column in numeric_features]

    df = pd.concat([df, lagged_features], axis=1).dropna(subset=lagged_features.columns)
    df = df.sort_values(by=['date', 'participant_id'])
    features = list(lagged_features.columns)
    
    X = df[features]
    y_reg = df['performance_index']        
    y_clf = df['injury_next_3_days']       
    
    split_idx = int(len(df) * 0.8)
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_reg_train, y_reg_test = y_reg.iloc[:split_idx], y_reg.iloc[split_idx:]
    y_clf_train, y_clf_test = y_clf.iloc[:split_idx], y_clf.iloc[split_idx:]
    
    return X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test, features

def train_performance_model(X_train, X_test, y_train, y_test):
    """Modèle 1 optimisé : Prédire l'Index de Performance"""
    print("\n--- MODÈLE 1 : PERFORMANCE INDEX (Optimisation Random Forest) ---")

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5]
    }

    tscv = TimeSeriesSplit(n_splits=3)
    
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=tscv, scoring='neg_root_mean_squared_error', n_jobs=-1)
    
    print("Recherche des meilleurs hyperparamètres en cours...")
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Meilleurs paramètres trouvés : {grid_search.best_params_}")
    
    predictions = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    mae = mean_absolute_error(y_test, predictions)
    baseline_prediction = np.repeat(y_train.mean(), len(y_test))
    print(f"RMSE : {rmse:.2f} points (sur 100)")
    print(f"MAE : {mae:.2f} points")
    print(f"R² : {r2:.2f}")
    return {
        'model': 'RandomForestRegressor',
        'best_params': str(grid_search.best_params_),
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'baseline_rmse': np.sqrt(mean_squared_error(y_test, baseline_prediction)),
        'baseline_mae': mean_absolute_error(y_test, baseline_prediction)
    }

def train_injury_model(X_train, X_test, y_train, y_test):
    """Modèle 2 optimisé : Prédire le risque de blessure"""
    print("\n--- MODÈLE 2 : INJURY RISK (Optimisation XGBoost) ---")
    
    neg_class = (y_train == 0).sum()
    pos_class = (y_train == 1).sum()
    scale_pos_weight = neg_class / pos_class if pos_class > 0 else 1
    
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5]
    }
    
    tscv = TimeSeriesSplit(n_splits=3)
    xgb = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42)
    
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=tscv, scoring='f1', n_jobs=-1)
    
    print("Recherche des meilleurs hyperparamètres en cours...")
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Meilleurs paramètres trouvés : {grid_search.best_params_}")
    
    predictions = best_model.predict(X_test)
    probabilities = best_model.predict_proba(X_test)[:, 1]
    majority_class = int(y_train.mode().iloc[0])
    baseline_predictions = np.repeat(majority_class, len(y_test))
    
    print("Matrice de confusion :")
    matrix = confusion_matrix(y_test, predictions)
    print(matrix)
    metrics = {
        'model': 'XGBClassifier',
        'best_params': str(grid_search.best_params_),
        'accuracy': accuracy_score(y_test, predictions),
        'balanced_accuracy': balanced_accuracy_score(y_test, predictions),
        'precision': precision_score(y_test, predictions, zero_division=0),
        'recall': recall_score(y_test, predictions, zero_division=0),
        'f1': f1_score(y_test, predictions, zero_division=0),
        'roc_auc': roc_auc_score(y_test, probabilities),
        'pr_auc': average_precision_score(y_test, probabilities),
        'baseline_balanced_accuracy': balanced_accuracy_score(y_test, baseline_predictions),
        'baseline_f1': f1_score(y_test, baseline_predictions, zero_division=0),
        'tn': matrix[0, 0], 'fp': matrix[0, 1],
        'fn': matrix[1, 0], 'tp': matrix[1, 1]
    }
    for metric, value in metrics.items():
        if metric not in ['model', 'best_params']:
            print(f"{metric} : {value:.3f}" if isinstance(value, float) else f"{metric} : {value}")
    return metrics

def main():
    df = pd.read_csv(DATA_FILE)
    
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test, features = prepare_data(df)
    print(f"Entraînement sur {len(X_train)} jours (passé), Test sur {len(X_test)} jours (futur).")
    
    os.makedirs(REPORT_DIR, exist_ok=True)
    performance_metrics = train_performance_model(X_train, X_test, y_reg_train, y_reg_test)
    injury_metrics = train_injury_model(X_train, X_test, y_clf_train, y_clf_test)
    pd.DataFrame([performance_metrics]).to_csv(
        os.path.join(REPORT_DIR, 'performance_metrics.csv'), index=False
    )
    pd.DataFrame([injury_metrics]).to_csv(
        os.path.join(REPORT_DIR, 'injury_metrics.csv'), index=False
    )
    with open(os.path.join(REPORT_DIR, 'modeling_choices.md'), 'w') as report_file:
        report_file.write("""# Modeling choices\n\n""")
        report_file.write("- Performance target: a 0-100 Hooper-style index combining readiness, sleep score, fatigue, soreness and stress.\n")
        report_file.write("- Performance model: Random Forest regression, selected with time-series cross-validation.\n")
        report_file.write("- Injury target: whether an injury occurs during the next 3 days.\n")
        report_file.write("- Injury model: XGBoost classification, with class weighting because injury is less frequent.\n")
        report_file.write("- Predictors are shifted by one day to avoid using same-day measurements that directly define the performance index.\n")
        report_file.write("- Regression metrics: MAE, RMSE and R2. Classification metrics: balanced accuracy, precision, recall, F1, ROC-AUC, PR-AUC and confusion matrix.\n")

if __name__ == "__main__":
    main()