# feature_engineering.py
# This script handles Step 4: Feature Engineering for the Ho Chi Minh City temperature forecasting project.
# We assume the data has been preprocessed using the functions from data_preprocessing.py.
# The goal is to transform the raw/preprocessed data into a feature matrix X and target Y suitable for machine learning.
# Since this is time series forecasting, we will:
# 1. Drop unnecessary columns early to optimize memory and computation.
# 2. Extract datetime components for seasonality.
# 3. Create lag features (past values) to capture temporal dependencies.
# 4. Create rolling window features (e.g., moving averages) to capture trends and volatility.
# 5. Optionally, create interactive features (e.g., products of key variables).
# 6. Define Y as the temperatures for the next 5 days (multi-step forecasting targets).
# 7. Use Random Forest to select features with importance >= 0.0065 for a single-step prediction.
# 8. Export X and Y to outputs/X_features.csv and outputs/Y_target.csv.
# Note: Optimized to avoid PerformanceWarning and DTypePromotionError, with early dropping of unnecessary columns.

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import os  # For creating output directory

# Import functions from data_preprocessing.py (assuming it's in the same directory)
from data_preprocessing import get_data, basic_transform

def drop_unnecessary_columns(df):
    """
    Drop columns that are irrelevant or redundant based on data understanding.
    - Low-correlation features: moonphase, windgust, precipcover, precipprob.
    - Non-numeric or redundant: sunrise, sunset (since datetime features are extracted).
    """
    low_corr_cols = ['moonphase', 'windgust', 'precipcover', 'precipprob']
    non_numeric_cols = ['sunrise', 'sunset']
    drop_cols = low_corr_cols + non_numeric_cols
    df_copy = df.copy()
    df_copy.drop(columns=[col for col in drop_cols if col in df_copy.columns], inplace=True)
    return df_copy

def extract_datetime_features(df):
    """
    Extract components from datetime for seasonality.
    - Year, month, day, day of week, quarter.
    This helps capture annual, monthly, and weekly patterns in temperature.
    """
    df_copy = df.copy()
    df_copy['year'] = df_copy['datetime'].dt.year
    df_copy['month'] = df_copy['datetime'].dt.month
    df_copy['day'] = df_copy['datetime'].dt.day
    df_copy['dayofweek'] = df_copy['datetime'].dt.dayofweek
    df_copy['quarter'] = df_copy['datetime'].dt.quarter
    df_copy['is_weekend'] = df_copy['dayofweek'].isin([5, 6]).astype(int)
    return df_copy

def create_lag_features(df, columns, lags=[1, 2, 3, 7, 14, 30]):
    """
    Create lag features for specified columns.
    - Lags: Past values shifted by 1,2,3,7,14,30 days to capture short-term and longer-term dependencies.
    - Uses pd.concat to avoid PerformanceWarning.
    """
    df_copy = df.copy()
    new_columns = {}
    for col in columns:
        for lag in lags:
            new_columns[f'{col}_lag_{lag}'] = df_copy[col].shift(lag)
    lag_df = pd.DataFrame(new_columns, index=df_copy.index)
    return pd.concat([df_copy, lag_df], axis=1)

def create_rolling_features(df, columns, windows=[3, 7, 14, 30]):
    """
    Create rolling window features.
    - Mean and std over windows of 3,7,14,30 days, shifted by 1 to avoid data leakage.
    - Uses pd.concat to avoid PerformanceWarning.
    """
    df_copy = df.copy()
    new_columns = {}
    for col in columns:
        for window in windows:
            new_columns[f'{col}_roll_mean_{window}'] = df_copy[col].rolling(window=window).mean().shift(1)
            new_columns[f'{col}_roll_std_{window}'] = df_copy[col].rolling(window=window).std().shift(1)
    rolling_df = pd.DataFrame(new_columns, index=df_copy.index)
    return pd.concat([df_copy, rolling_df], axis=1)

def create_interactive_features(df):
    """
    Optional: Create interaction features.
    - Based on insights from data understanding, e.g., temp * humidity, solarradiation / (cloudcover + 1).
    - Uses copy to avoid fragmentation.
    """
    df_copy = df.copy()
    df_copy['temp_humidity_interact'] = df_copy['temp'] * df_copy['humidity']
    df_copy['effective_solar'] = df_copy['solarradiation'] / (df_copy['cloudcover'] + 1)
    df_copy['dew_wind_interact'] = df_copy['dew'] * df_copy['windspeedmean']
    return df_copy

def create_targets(df, horizons=5):
    """
    Create multi-step targets for next 5 days.
    - Y will be a DataFrame with columns ['temp_target_1', 'temp_target_2', ..., 'temp_target_5'].
    """
    targets = pd.DataFrame(index=df.index)
    for h in range(1, horizons + 1):
        targets[f'temp_target_{h}'] = df['temp'].shift(-h)
    return targets

def select_features_with_rf(X, y, importance_threshold=0.0065):
    """
    Use Random Forest to select features with importance >= threshold.
    - Train RF on X and y (single target, e.g., 1-day ahead).
    - Returns selected X with features meeting the threshold.
    """
    valid_idx = y.notna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("Top Feature Importances:\n", importances.head(20))
    print(f"\nNumber of features with importance >= {importance_threshold}: {sum(importances >= importance_threshold)}")
    
    top_features = importances[importances >= importance_threshold].index
    X_selected = X[top_features]
    
    return X_selected, importances

def main():
    # Step 1: Load and preprocess data
    path = r"https://raw.githubusercontent.com/7hufofbun/DSEB--Machine-Learning_Group5/refs/heads/main/data/weather_hcm_daily.csv"
    data = get_data(path)
    data = basic_transform(data)
    
    # Step 2: Drop unnecessary columns for optimization
    data = drop_unnecessary_columns(data)
    
    # Step 3: Extract datetime features (before setting index)
    data = extract_datetime_features(data)
    
    # Step 4: Set index to datetime for time series operations
    data.set_index('datetime', inplace=True)
    data.sort_index(inplace=True)
    
    # Step 5: Define key columns for lags and rollings
    key_columns = ['temp', 'tempmax', 'tempmin', 'feelslike', 'humidity', 'dew', 'solarradiation', 'cloudcover', 'windspeedmean']
    encoded_cols = [col for col in data.columns if col.startswith(('icon_', 'conditions_'))]
    key_columns.extend(encoded_cols)
    
    # Step 6: Create lag features
    data = create_lag_features(data, key_columns)
    
    # Step 7: Create rolling features
    data = create_rolling_features(data, key_columns)
    
    # Step 8: Create interactive features
    data = create_interactive_features(data)
    
    # Step 9: Create targets Y (next 5 days)
    Y = create_targets(data)
    
    # Step 10: Define X, avoiding leakage
    drop_cols = ['temp', 'tempmax', 'tempmin', 'feelslike']
    X = data.drop(columns=drop_cols, errors='ignore')
    
    # Align X and Y
    combined = pd.concat([X, Y], axis=1).dropna()
    X = combined[X.columns]
    Y = combined[Y.columns]
    
    # Step 11: Feature selection with Random Forest
    y_single = Y['temp_target_1']
    X_selected, importances = select_features_with_rf(X, y_single, importance_threshold=0.0065)
    
    # Step 12: Export X and Y to outputs directory
    os.makedirs('outputs', exist_ok=True)
    X_selected.to_csv('outputs/X_features.csv', index=True)
    Y.to_csv('outputs/Y_target.csv', index=True)
    
    print("Feature engineering complete. X and Y exported to outputs/X_features.csv and outputs/Y_target.csv.")

if __name__ == "__main__":
    main()