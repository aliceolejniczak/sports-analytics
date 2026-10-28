# Data preprocessing

## Type and unit management
- Dates are parsed with `errors='coerce'` and participant identifiers are numeric.
- Numeric fields are converted with `pd.to_numeric`.
- Fitbit sleep duration is converted from milliseconds to hours; exercise durations to minutes.

## Missing values: alternatives and choice
1. Listwise deletion is simple but would discard many athlete-days.
2. Global mean imputation is fast but ignores participant differences.
3. Participant median is robust and is used for activity/nutrition variables.
4. Forward/backward filling respects the longitudinal structure and is used for wellness/sleep series.
5. KNN or MICE would be more advanced, but are harder to justify with only three participants and can leak temporal information.

## Outliers: alternatives and choice
- Rows are not deleted because an extreme training day may be a real event.
- Domain bounds convert impossible values to missing values.
- Remaining numeric extremes are capped with the 1.5-IQR rule, then imputed.
- See `outlier_summary.csv` and `missing_summary.csv` for audit counts.
