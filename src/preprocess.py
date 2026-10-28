import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

INPUT_FILE = "data/processed/raw_merged_dataset.csv"
OUTPUT_FILE = "data/processed/test_dataset.csv"
MODEL_OUTPUT_FILE = "data/processed/model_dataset.csv"
PREPROCESSING_REPORT_DIR = "reports/preprocessing"

MODEL_COLUMNS = [
    'participant_id', 'date', 'is_injured',
    'fatigue', 'mood', 'readiness', 'sleep_duration_h', 'sleep_quality',
    'soreness', 'stress', 'perceived_exertion', 'duration_min',
    'training_load', 'pmsys_activity_count', 'total_steps', 'total_calories',
    'overall_score', 'deep_sleep_in_minutes', 'resting_heart_rate',
    'restlessness', 'fitbit_sleep_minutesAsleep', 'fitbit_sleep_efficiency',
    'fitbit_exercise_calories', 'fitbit_exercise_duration',
    'fitbit_exercise_steps', 'gdoc_weight', 'gdoc_glasses_of_fluid',
    'gdoc_meal_count', 'gdoc_alcohol_consumed', 'calories_consumed',
    'protein_g', 'carbohydrates_g', 'fat_g', 'food_image_count',
    'food_model_confidence', 'performance_index', 'injury_next_3_days'
]

DOMAIN_BOUNDS = {
    'fatigue': (0, 10), 'mood': (0, 10), 'readiness': (0, 10),
    'sleep_quality': (0, 10), 'soreness': (0, 10), 'stress': (0, 10),
    'perceived_exertion': (0, 10), 'gdoc_glasses_of_fluid': (0, 30),
    'gdoc_alcohol_consumed': (0, 1), 'is_injured': (0, 1),
    'injury_next_3_days': (0, 1), 'food_model_confidence': (0, 1),
}

ID_AND_TARGET_COLUMNS = {
    'participant_id', 'sleep_log_entry_id', 'fitbit_exercise_logId',
    'fitbit_exercise_activityTypeId', 'fitbit_sleep_logId',
    'is_injured', 'performance_index', 'injury_next_3_days'
}

def manage_types_and_units(df):
    """Convertit les dates, identifiants et durées vers des types/unités explicites."""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['participant_id'] = pd.to_numeric(df['participant_id'], errors='coerce').astype('Int64')

    numeric_candidates = [column for column in df.columns if column not in ['date', 'gdoc_meals',
                      'detected_food', 'calorie_estimation_confidence']]
    for column in numeric_candidates:
        df[column] = pd.to_numeric(df[column], errors='coerce')

    millisecond_columns = [
        'fitbit_sleep_duration', 'fitbit_exercise_duration',
        'fitbit_exercise_activeDuration', 'fitbit_exercise_originalDuration'
    ]
    for column in millisecond_columns:
        if column in df.columns:
            df[column] = df[column] / 60000

    if 'fitbit_sleep_duration' in df.columns:
        df['fitbit_sleep_duration'] = df['fitbit_sleep_duration'] / 60
    return df

def detect_and_process_outliers(df):
    """Applique des bornes métier puis un plafonnement IQR sans supprimer de lignes."""
    df = df.copy()
    report = []
    numeric_columns = df.select_dtypes(include='number').columns

    for column in numeric_columns:
        before = df[column].notna()
        lower_bound, upper_bound = DOMAIN_BOUNDS.get(column, (None, None))
        domain_count = 0
        if lower_bound is not None:
            invalid = before & ((df[column] < lower_bound) | (df[column] > upper_bound))
            domain_count = int(invalid.sum())
            df.loc[invalid, column] = pd.NA

        iqr_count = 0
        if column not in ID_AND_TARGET_COLUMNS and df[column].nunique(dropna=True) >= 10:
            q1, q3 = df[column].quantile([0.25, 0.75])
            iqr = q3 - q1
            if pd.notna(iqr) and iqr > 0:
                lower_iqr, upper_iqr = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                outliers = df[column].notna() & ((df[column] < lower_iqr) | (df[column] > upper_iqr))
                iqr_count = int(outliers.sum())
                df[column] = df[column].clip(lower_iqr, upper_iqr)

        report.append({
            'column': column,
            'domain_invalid_values': domain_count,
            'iqr_values_capped': iqr_count,
            'total_handled': domain_count + iqr_count
        })
    return df, pd.DataFrame(report)

def clean_and_impute(df):
    """
    Nettoyage basé sur la logique métier des séries temporelles sportives.
    """
    df = manage_types_and_units(df)
    df = df.sort_values(by=['participant_id', 'date'])
    df, _ = detect_and_process_outliers(df)
    
    df['perceived_exertion'] = df['perceived_exertion'].fillna(0)
    df['duration_min'] = df['duration_min'].fillna(0)
    
    df['training_load'] = df['perceived_exertion'] * df['duration_min']
    
    time_series_cols = [
        'fatigue', 'stress', 'soreness', 'mood', 'readiness', 
        'sleep_duration_h', 'sleep_quality', 'resting_heart_rate',
        'overall_score', 'composition_score', 'revitalization_score',
        'duration_score', 'deep_sleep_in_minutes', 'restlessness'
    ]
    
    for col in time_series_cols:
        if col in df.columns:
            df[col] = df.groupby('participant_id')[col].ffill().bfill()
            df[col] = df[col].fillna(df[col].median())

    for col in [
        'total_steps', 'total_calories', 'calories_consumed',
        'gdoc_weight', 'gdoc_glasses_of_fluid', 'gdoc_alcohol_consumed'
    ]:
        if col in df.columns:
            df[col] = df.groupby('participant_id')[col].transform(lambda x: x.fillna(x.median()))
            df[col] = df[col].fillna(df[col].median())

    if 'gdoc_meals' in df.columns:
        df['gdoc_meals'] = df['gdoc_meals'].fillna('Unknown')
    if 'gdoc_meal_count' in df.columns:
        df['gdoc_meal_count'] = df['gdoc_meal_count'].fillna(0)
    if 'calorie_estimation_confidence' in df.columns:
        df['calorie_estimation_confidence'] = df['calorie_estimation_confidence'].fillna('unknown')
    if 'detected_food' in df.columns:
        df['detected_food'] = df['detected_food'].fillna('Unknown')

    numeric_columns = df.select_dtypes(include=['number']).columns
    for col in numeric_columns:
        df[col] = df.groupby('participant_id')[col].transform(lambda x: x.fillna(x.median()))
        df[col] = df[col].fillna(df[col].median()).fillna(0)
            
    return df

def engineer_features_and_targets(df):
    """
    Création des variables cibles (Performance Index et Injury Risk window).
    """
    df = df.copy()
    
    raw_perf = (
        (df['readiness'] * 10) + 
        df['overall_score'] - 
        (df['fatigue'] * 10) - 
        (df['soreness'] * 10) - 
        (df['stress'] * 5)
    )
    
    scaler = MinMaxScaler(feature_range=(0, 100))
    df['performance_index'] = scaler.fit_transform(raw_perf.values.reshape(-1, 1)).round(2)
 
    df['injury_next_3_days'] = (
        df.groupby('participant_id')['is_injured']
        .transform(lambda x: x.rolling(window=3, min_periods=1).max().shift(-3))
        .fillna(0)
        .astype(int)
    )
    
    return df

def build_model_dataset(df):
    """Sélectionne les variables métier utiles à l'entraînement."""
    available_columns = [column for column in MODEL_COLUMNS if column in df.columns]
    return df[available_columns].copy()

def main():
    print("Chargement du dataset brut agrégé...")
    df = pd.read_csv(INPUT_FILE)
    
    os.makedirs(PREPROCESSING_REPORT_DIR, exist_ok=True)
    print("Gestion des types, unités, valeurs aberrantes et valeurs manquantes...")
    typed_df = manage_types_and_units(df)
    outlier_df, outlier_report = detect_and_process_outliers(typed_df)
    outlier_report.to_csv(os.path.join(PREPROCESSING_REPORT_DIR, 'outlier_summary.csv'), index=False)
    df_cleaned = clean_and_impute(df)
    
    print("Création des variables cibles (Performance Index & Injury Risk)...")
    df_final = engineer_features_and_targets(df_cleaned)

    df_final.to_csv(OUTPUT_FILE, index=False)
    model_df = build_model_dataset(df_final)
    model_df.to_csv(MODEL_OUTPUT_FILE, index=False)
    missing_report = pd.DataFrame({
        'missing_before': df.isna().sum(),
        'missing_after': df_final.isna().sum()
    })
    missing_report['filled_values'] = (
        missing_report['missing_before'] - missing_report['missing_after']
    )
    missing_report.to_csv(os.path.join(PREPROCESSING_REPORT_DIR, 'missing_summary.csv'))
    with open(os.path.join(PREPROCESSING_REPORT_DIR, 'preprocessing_choices.md'), 'w') as report_file:
        report_file.write("""# Data preprocessing\n\n""")
        report_file.write("## Type and unit management\n")
        report_file.write("- Dates are parsed with `errors='coerce'` and participant identifiers are numeric.\n")
        report_file.write("- Numeric fields are converted with `pd.to_numeric`.\n")
        report_file.write("- Fitbit sleep duration is converted from milliseconds to hours; exercise durations to minutes.\n\n")
        report_file.write("## Missing values: alternatives and choice\n")
        report_file.write("1. Listwise deletion is simple but would discard many athlete-days.\n")
        report_file.write("2. Global mean imputation is fast but ignores participant differences.\n")
        report_file.write("3. Participant median is robust and is used for activity/nutrition variables.\n")
        report_file.write("4. Forward/backward filling respects the longitudinal structure and is used for wellness/sleep series.\n")
        report_file.write("5. KNN or MICE would be more advanced, but are harder to justify with only three participants and can leak temporal information.\n\n")
        report_file.write("## Outliers: alternatives and choice\n")
        report_file.write("- Rows are not deleted because an extreme training day may be a real event.\n")
        report_file.write("- Domain bounds convert impossible values to missing values.\n")
        report_file.write("- Remaining numeric extremes are capped with the 1.5-IQR rule, then imputed.\n")
        report_file.write("- See `outlier_summary.csv` and `missing_summary.csv` for audit counts.\n")
    print(f"✅ Fichier nettoyé sauvegardé sous : {OUTPUT_FILE}")
    print(f"✅ Dataset modèle sauvegardé sous : {MODEL_OUTPUT_FILE}")
    print(f"Dimensions dataset complet : {df_final.shape}")
    print(f"Dimensions dataset modèle : {model_df.shape}")
    print(f"Bilan valeurs manquantes restantes : {df_final.isnull().sum().sum()}")
    print("\nAperçu des cibles créées :")
    print(df_final[['participant_id', 'date', 'performance_index', 'is_injured', 'injury_next_3_days']].head())

if __name__ == "__main__":
    main()