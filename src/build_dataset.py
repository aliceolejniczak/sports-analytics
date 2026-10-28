import os
import json
import pandas as pd
from functools import reduce
from image_processing import process_all_images

RAW_DATA_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

def load_daily_csv(filepath, participant_id, date_col='effective_time_frame', dayfirst=False):
    if not os.path.exists(filepath):
        return pd.DataFrame()
    
    df = pd.read_csv(filepath)
    if date_col not in df.columns:
        date_cols = [c for c in df.columns if 'time' in c.lower() or 'date' in c.lower()]
        if date_cols:
            date_col = date_cols[0]
        else:
            return pd.DataFrame()
            
    df['date'] = pd.to_datetime(
        df[date_col], format='mixed', dayfirst=dayfirst, errors='coerce'
    ).dt.date
    df['participant_id'] = participant_id
    
    numeric_cols = df.select_dtypes(include=['number']).columns
    df_daily = df.groupby(['participant_id', 'date'])[numeric_cols].mean().reset_index()
    return df_daily

def load_googledocs_folder(participant_id, p_path):
    """Parcourt tous les CSV du dossier googledocs et les fusionne."""
    googledocs_path = os.path.join(p_path, 'googledocs')
    if not os.path.exists(googledocs_path):
        return pd.DataFrame()
        
    gdoc_dfs = []
    for file in os.listdir(googledocs_path):
        if file.endswith('.csv'):
            filepath = os.path.join(googledocs_path, file)
            df_temp = load_daily_csv(filepath, participant_id, dayfirst=True)
            raw_df = pd.read_csv(filepath)
            raw_df['date'] = pd.to_datetime(
                raw_df['date'], format='mixed', dayfirst=True, errors='coerce'
            ).dt.date

            if 'meals' in raw_df.columns:
                meals_daily = (
                    raw_df.dropna(subset=['date'])
                    .groupby(['date'])['meals']
                    .agg(lambda values: ', '.join(sorted(set(', '.join(values).split(', ')))))
                    .reset_index(name='gdoc_meals')
                )
                meals_daily['gdoc_meal_count'] = meals_daily['gdoc_meals'].str.count(',') + 1
                meals_daily['participant_id'] = participant_id
                df_temp = pd.merge(
                    df_temp, meals_daily, on=['participant_id', 'date'], how='outer'
                )

            if 'alcohol_consumed' in raw_df.columns:
                alcohol_daily = (
                    raw_df.dropna(subset=['date'])
                    .assign(
                        gdoc_alcohol_consumed=lambda frame: frame['alcohol_consumed']
                        .astype(str).str.strip().str.lower().eq('yes').astype(int)
                    )
                    .groupby('date')['gdoc_alcohol_consumed']
                    .max()
                    .reset_index()
                )
                alcohol_daily['participant_id'] = participant_id
                df_temp = pd.merge(
                    df_temp, alcohol_daily, on=['participant_id', 'date'], how='outer'
                )

            if not df_temp.empty:
                rename_dict = {
                    c: f"gdoc_{c}"
                    for c in df_temp.columns
                    if c not in ['participant_id', 'date'] and not c.startswith('gdoc_')
                }
                df_temp = df_temp.rename(columns=rename_dict)
                gdoc_dfs.append(df_temp)
                
    if not gdoc_dfs:
        return pd.DataFrame()
        
    return reduce(lambda left, right: pd.merge(left, right, on=['participant_id', 'date'], how='outer'), gdoc_dfs)

def aggregate_minute_json(filepath, participant_id, value_col, agg_func='sum'):
    if not os.path.exists(filepath):
        return pd.DataFrame()
        
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    if not data:
        return pd.DataFrame()
        
    df = pd.DataFrame(data)
    if 'dateTime' in df.columns and 'value' in df.columns:
        df['date'] = pd.to_datetime(
            df['dateTime'], format='mixed', dayfirst=False, errors='coerce'
        ).dt.date
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        if agg_func == 'sum':
            df_daily = df.groupby('date')['value'].sum().reset_index()
        else:
            df_daily = df.groupby('date')['value'].mean().reset_index()
            
        df_daily = df_daily.rename(columns={'value': value_col})
        df_daily['participant_id'] = participant_id
        
        return df_daily[['participant_id', 'date', value_col]]
    
    return pd.DataFrame()

def aggregate_fitbit_json(filepath, participant_id, prefix, date_columns, sum_columns=None):
    """Agrège les champs numériques de tout fichier Fitbit au niveau journalier."""
    if not os.path.exists(filepath):
        return pd.DataFrame()

    with open(filepath, 'r') as file:
        data = json.load(file)
    if not isinstance(data, list) or not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    date_column = next((column for column in date_columns if column in df.columns), None)
    if date_column is None:
        return pd.DataFrame()

    df['date'] = pd.to_datetime(df[date_column], format='mixed', errors='coerce').dt.date
    for column in df.columns:
        if column not in [date_column, 'date']:
            converted = pd.to_numeric(df[column], errors='coerce')
            if converted.notna().any():
                df[column] = converted

    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    if not numeric_columns:
        return pd.DataFrame()

    sum_columns = set(sum_columns or [])
    daily = {}
    for column in numeric_columns:
        aggregation = 'sum' if column in sum_columns else 'mean'
        daily[f'{prefix}_{column}'] = df.groupby('date')[column].agg(aggregation)

    result = pd.DataFrame(daily).reset_index()
    result['participant_id'] = participant_id
    return result[['participant_id', 'date'] + list(daily)]

def main():
    print("Début de l'agrégation des données...")
    all_participants_data = []
    participant_folders = [f for f in os.listdir(RAW_DATA_DIR) if f.startswith('p') and os.path.isdir(os.path.join(RAW_DATA_DIR, f))]
    
    for p_folder in participant_folders:
        p_path = os.path.join(RAW_DATA_DIR, p_folder)
        participant_id = p_folder.replace('p', '').lstrip('0')
        
        dataframes_to_merge = []
        
        # 1. PMsys
        pmsys_path = os.path.join(p_path, 'pmsys')
        df_wellness = load_daily_csv(os.path.join(pmsys_path, 'wellness.csv'), participant_id)
        df_srpe = load_daily_csv(os.path.join(pmsys_path, 'srpe.csv'), participant_id)
        
        injury_file = os.path.join(pmsys_path, 'injury.csv')
        if os.path.exists(injury_file):
            df_injury = pd.read_csv(injury_file)
            date_col = [c for c in df_injury.columns if 'time' in c.lower() or 'date' in c.lower()][0]
            df_injury['date'] = pd.to_datetime(
                df_injury[date_col], format='mixed', dayfirst=False, errors='coerce'
            ).dt.date
            df_injury['participant_id'] = participant_id
            df_injury['is_injured'] = 1 
            df_injury_daily = df_injury.groupby(['participant_id', 'date'])['is_injured'].max().reset_index()
            dataframes_to_merge.append(df_injury_daily)
            
        if not df_wellness.empty: dataframes_to_merge.append(df_wellness)
        if not df_srpe.empty: dataframes_to_merge.append(df_srpe)

        srpe_path = os.path.join(pmsys_path, 'srpe.csv')
        if os.path.exists(srpe_path):
            srpe_raw = pd.read_csv(srpe_path)
            srpe_raw['date'] = pd.to_datetime(
                srpe_raw['end_date_time'], format='mixed', errors='coerce'
            ).dt.date
            srpe_activity = srpe_raw.groupby('date').agg(
                pmsys_activity_count=('activity_names', 'count')
            ).reset_index()
            srpe_activity['participant_id'] = participant_id
            dataframes_to_merge.append(srpe_activity[['participant_id', 'date', 'pmsys_activity_count']])

        # 2. GOOGLEDOCS 
        df_gdocs = load_googledocs_folder(participant_id, p_path)
        if not df_gdocs.empty: dataframes_to_merge.append(df_gdocs)
            
        # 3. FitBit
        fitbit_path = os.path.join(p_path, 'fitbit')
        df_steps = aggregate_minute_json(os.path.join(fitbit_path, 'steps.json'), participant_id, 'total_steps', 'sum')
        df_calories = aggregate_minute_json(os.path.join(fitbit_path, 'calories.json'), participant_id, 'total_calories', 'sum')
        df_hr = aggregate_minute_json(os.path.join(fitbit_path, 'heart_rate.json'), participant_id, 'avg_heart_rate', 'mean')
        df_sleep = load_daily_csv(os.path.join(fitbit_path, 'sleep_score.csv'), participant_id, 'timestamp')

        fitbit_files = [
            ('distance.json', 'fitbit_distance', ['dateTime'], ['value']),
            ('resting_heart_rate.json', 'fitbit_resting_hr', ['dateTime'], []),
            ('time_in_heart_rate_zones.json', 'fitbit_hr_zone', ['dateTime'], ['value']),
            ('exercise.json', 'fitbit_exercise', ['startTime', 'originalStartTime'],
             ['calories', 'duration', 'activeDuration', 'steps', 'elevationGain']),
            ('sleep.json', 'fitbit_sleep', ['dateOfSleep', 'startTime'],
             ['duration', 'minutesAsleep', 'minutesAwake', 'timeInBed'])
        ]
        extra_fitbit = [
            aggregate_fitbit_json(os.path.join(fitbit_path, filename), participant_id,
                                  prefix, date_columns, sum_columns)
            for filename, prefix, date_columns, sum_columns in fitbit_files
        ]

        for dataframe in [df_steps, df_calories, df_hr, df_sleep] + extra_fitbit:
            if not dataframe.empty:
                dataframes_to_merge.append(dataframe)
            
        if dataframes_to_merge:
            df_participant = reduce(lambda left, right: pd.merge(left, right, on=['participant_id', 'date'], how='outer'), dataframes_to_merge)
            all_participants_data.append(df_participant)
            
    if all_participants_data:
        final_df = pd.concat(all_participants_data, ignore_index=True)
        final_df['participant_id'] = final_df['participant_id'].astype(str).str.lstrip('0')
        final_df['is_injured'] = final_df['is_injured'].fillna(0)
        final_df = final_df.sort_values(by=['participant_id', 'date'])

        print("Extraction des calories depuis les photos de repas...")
        df_food = process_all_images(RAW_DATA_DIR)
        
        if not df_food.empty:
            df_food['participant_id'] = df_food['participant_id'].astype(str).str.lstrip('0')
            df_food['date'] = pd.to_datetime(df_food['date'], errors='coerce').dt.date
            final_df = pd.merge(final_df, df_food, on=['participant_id', 'date'], how='left')
        
        output_path = os.path.join(PROCESSED_DIR, "raw_merged_dataset.csv")
        final_df.to_csv(output_path, index=False)
        print(f"✅ Agrégation terminée ! Fichier sauvegardé dans : {output_path}")
        print(f"Shape du dataset : {final_df.shape}")

if __name__ == "__main__":
    main()