import os

import matplotlib.pyplot as plt
import pandas as pd

INPUT_FILE = "data/processed/test_dataset.csv"
OUTPUT_DIR = "reports/eda"

MAIN_VARIABLES = [
    'fatigue', 'stress', 'readiness', 'sleep_duration_h',
    'total_steps', 'total_calories', 'training_load',
    'calories_consumed', 'performance_index', 'is_injured',
    'injury_next_3_days'
]


def save_structure_report(df):
    structure = pd.DataFrame({
        'metric': [
            'rows', 'columns', 'participants', 'days_per_participant',
            'missing_cells', 'numeric_columns', 'text_columns'
        ],
        'value': [
            len(df), len(df.columns), df['participant_id'].nunique(),
            df.groupby('participant_id')['date'].nunique().to_dict(),
            int(df.isna().sum().sum()),
            int(len(df.select_dtypes(include='number').columns)),
            int(len(df.select_dtypes(exclude='number').columns))
        ]
    })
    structure.to_csv(os.path.join(OUTPUT_DIR, 'structure_summary.csv'), index=False)

    available = [column for column in MAIN_VARIABLES if column in df.columns]
    df[available].describe().T.to_csv(os.path.join(OUTPUT_DIR, 'main_variables_summary.csv'))
    df.isna().sum().sort_values(ascending=False).to_csv(
        os.path.join(OUTPUT_DIR, 'missing_values_summary.csv'), header=['missing_count']
    )


def save_distribution_plot(df):
    variables = [column for column in MAIN_VARIABLES if column in df.columns]
    figure, axes = plt.subplots(3, 4, figsize=(16, 11))
    axes = axes.ravel()

    for axis, variable in zip(axes, variables):
        if variable in ['is_injured', 'injury_next_3_days']:
            counts = df[variable].value_counts().sort_index()
            axis.bar(counts.index.astype(str), counts.values)
            axis.set_xlabel('value')
        else:
            axis.hist(df[variable], bins=20, edgecolor='black')
        axis.set_title(variable)
        axis.set_ylabel('count')

    for axis in axes[len(variables):]:
        axis.axis('off')
    figure.suptitle('Distribution des variables principales')
    figure.tight_layout()
    figure.savefig(os.path.join(OUTPUT_DIR, 'main_variables_distributions.png'), dpi=150)
    plt.close(figure)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(INPUT_FILE)
    df['date'] = pd.to_datetime(df['date'])
    save_structure_report(df)
    save_distribution_plot(df)
    print(f"Rapports EDA créés dans : {OUTPUT_DIR}")
    print(f"Dimensions : {df.shape}")
    print(f"Valeurs manquantes : {int(df.isna().sum().sum())}")


if __name__ == '__main__':
    main()