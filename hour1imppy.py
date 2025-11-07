import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import warnings
import pickle
import json
warnings.filterwarnings('ignore')

# Try to import Optuna and ONNX libraries
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not available. Using default hyperparameters.")

try:
    import onnx
    import onnxmltools
    from onnxmltools.convert.common.data_types import FloatTensorType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: ONNX libraries not available. Install with: pip install onnx onnxmltools")

# ============================================================================
# CONFIGURATION
# ============================================================================
TUNE_HYPERPARAMETERS = True
N_TRIALS = 50
MODEL_NAME = "d1_temperature_model"
SAVE_DIR = "./models/d1/"

print("="*80)
print("SPECIALIZED D+1 (NEXT DAY) TEMPERATURE PREDICTION")
print("="*80)
print("Focus: Predicting tomorrow's temperature (24-hour forecast)")
print("Strategy: Ultra-short-term patterns, recent momentum, diurnal cycles")
print("="*80)

# Create save directory
import os
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================================
# 1. LOAD DATA & ANALYZE
# ============================================================================
print("\n" + "="*80)
print("LOADING AND ANALYZING DATA")
print("="*80)

hourly = pd.read_csv('https://raw.githubusercontent.com/7hufofbun/DSEB--Machine-Learning_Group5/main/data/weather_hcm_hourly.csv')
daily = pd.read_csv('https://raw.githubusercontent.com/7hufofbun/DSEB--Machine-Learning_Group5/main/data/weather_hcm_daily.csv')

hourly['datetime'] = pd.to_datetime(hourly['datetime'])
daily['datetime'] = pd.to_datetime(daily['datetime'])

print(f"Hourly: {hourly.shape}")
print(f"Daily: {daily.shape}")

# Analyze daily temperature calculation
hourly['date'] = hourly['datetime'].dt.date
sample_dates = hourly['date'].unique()[:10]

comparison_results = []
for date in sample_dates:
    day_data = hourly[hourly['date'] == date]
    if len(day_data) >= 20:
        hourly_mean = day_data['temp'].mean()
        max_min_mean = (day_data['temp'].max() + day_data['temp'].min()) / 2
        daily_date = pd.to_datetime(date)
        daily_temp = daily[daily['datetime'].dt.date == date]['temp'].values
        
        if len(daily_temp) > 0:
            daily_temp = daily_temp[0]
            comparison_results.append({
                'diff_hourly': abs(daily_temp - hourly_mean),
                'diff_maxmin': abs(daily_temp - max_min_mean)
            })

comp_df = pd.DataFrame(comparison_results)
DAILY_CALC_METHOD = "hourly_mean" if comp_df['diff_hourly'].mean() < comp_df['diff_maxmin'].mean() else "max_min"
print(f"Daily calculation method: {DAILY_CALC_METHOD.upper()}")

# ============================================================================
# 2. PREPARE DAILY TARGET WITH D+1 SPECIFIC FEATURES
# ============================================================================
print("\n" + "="*80)
print("PREPARING DAILY TARGET WITH D+1 SPECIFIC FEATURES")
print("="*80)

daily_target = daily[['datetime', 'temp']].copy()
daily_target.columns = ['date', 'daily_temp']
daily_target['date'] = pd.to_datetime(daily_target['date'])
daily_target = daily_target.sort_values('date').reset_index(drop=True)

# D+1 specific features
daily_target['temp_yesterday'] = daily_target['daily_temp'].shift(1)
daily_target['temp_diff_1d'] = daily_target['daily_temp'].diff(1)

for window in [2, 3]:
    daily_target[f'temp_std_{window}d'] = daily_target['daily_temp'].rolling(
        window=window, min_periods=1).std()
    daily_target[f'temp_range_{window}d'] = (
        daily_target['daily_temp'].rolling(window=window, min_periods=1).max() -
        daily_target['daily_temp'].rolling(window=window, min_periods=1).min())
    daily_target[f'temp_mean_{window}d'] = daily_target['daily_temp'].rolling(
        window=window, min_periods=1).mean()

daily_target['temp_persistence'] = daily_target['daily_temp'].shift(1)
daily_target['day_of_week'] = daily_target['date'].dt.dayofweek
daily_target['is_weekend'] = (daily_target['day_of_week'] >= 5).astype(int)
daily_target['date'] = daily_target['date'].dt.date

print(f"Daily target shape: {daily_target.shape}")

# ============================================================================
# 3. SPLIT DATA
# ============================================================================
print("\n" + "="*80)
print("SPLITTING DATA")
print("="*80)

text_columns = ['name', 'address', 'resolvedAddress', 'conditions', 'source',
                'latitude', 'longitude', 'preciptype', 'severerisk']
hourly = hourly.drop(columns=[col for col in text_columns if col in hourly.columns])

hourly = hourly.sort_values('datetime')
n = len(hourly)
train_size = int(n * 0.7)
val_size = int(n * 0.15)

hourly_train = hourly.iloc[:train_size].copy()
hourly_val = hourly.iloc[train_size:train_size + val_size].copy()
hourly_test = hourly.iloc[train_size + val_size:].copy()

print(f"Train: {len(hourly_train)} hours ({len(hourly_train)//24} days)")
print(f"Val:   {len(hourly_val)} hours ({len(hourly_val)//24} days)")
print(f"Test:  {len(hourly_test)} hours ({len(hourly_test)//24} days)")

# ============================================================================
# 4. FEATURE ENGINEERING FOR D+1
# ============================================================================
def engineer_features_for_d1(df):
    """Specialized feature engineering for 24-hour (D+1) forecasting"""
    df_eng = df.copy()
    df_eng['datetime'] = pd.to_datetime(df_eng['datetime'])
    df_eng = df_eng.set_index('datetime').sort_index()
    
    if 'datetime' in df_eng.columns:
        df_eng = df_eng.drop(columns=['datetime'])
    
    # Temporal features
    df_eng['hour'] = df_eng.index.hour
    df_eng['day_of_week'] = df_eng.index.dayofweek
    df_eng['month'] = df_eng.index.month
    df_eng['day_of_year'] = df_eng.index.dayofyear
    df_eng['is_weekend'] = (df_eng.index.dayofweek >= 5).astype(int)
    df_eng['week_of_year'] = df_eng.index.isocalendar().week
    df_eng['quarter'] = df_eng.index.quarter
    
    # Time of day categories
    df_eng['is_morning'] = ((df_eng.index.hour >= 6) & (df_eng.index.hour < 12)).astype(int)
    df_eng['is_afternoon'] = ((df_eng.index.hour >= 12) & (df_eng.index.hour < 18)).astype(int)
    df_eng['is_evening'] = ((df_eng.index.hour >= 18) & (df_eng.index.hour < 22)).astype(int)
    df_eng['is_night'] = ((df_eng.index.hour >= 22) | (df_eng.index.hour < 6)).astype(int)
    
    # Cyclical encoding
    df_eng['hour_sin'] = np.sin(2 * np.pi * df_eng.index.hour / 24)
    df_eng['hour_cos'] = np.cos(2 * np.pi * df_eng.index.hour / 24)
    df_eng['month_sin'] = np.sin(2 * np.pi * df_eng.index.month / 12)
    df_eng['month_cos'] = np.cos(2 * np.pi * df_eng.index.month / 12)
    df_eng['day_sin'] = np.sin(2 * np.pi * df_eng.index.dayofweek / 7)
    df_eng['day_cos'] = np.cos(2 * np.pi * df_eng.index.dayofweek / 7)
    
    # Temperature features
    if 'temp' in df_eng.columns:
        # Ultra-short lags
        for lag in [1, 2, 3, 4, 6, 8, 12, 18, 24]:
            df_eng[f'temp_lag_{lag}h'] = df_eng['temp'].shift(lag)
        
        # Short rolling windows
        for window in [3, 6, 12, 18, 24]:
            df_eng[f'temp_mean_{window}h'] = df_eng['temp'].rolling(window=window, min_periods=1).mean()
            df_eng[f'temp_std_{window}h'] = df_eng['temp'].rolling(window=window, min_periods=1).std()
            df_eng[f'temp_min_{window}h'] = df_eng['temp'].rolling(window=window, min_periods=1).min()
            df_eng[f'temp_max_{window}h'] = df_eng['temp'].rolling(window=window, min_periods=1).max()
            df_eng[f'temp_range_{window}h'] = df_eng[f'temp_max_{window}h'] - df_eng[f'temp_min_{window}h']
        
        # Temperature changes
        for lag in [1, 2, 3, 6, 12]:
            df_eng[f'temp_change_{lag}h'] = df_eng['temp'] - df_eng['temp'].shift(lag)
        
        # Temperature velocity
        df_eng['temp_velocity_3h'] = df_eng['temp_change_3h'] / 3
        df_eng['temp_velocity_6h'] = df_eng['temp_change_6h'] / 6
        
        # Trend direction
        df_eng['temp_trending_up'] = (df_eng['temp_change_6h'] > 0).astype(int)
        df_eng['temp_trending_down'] = (df_eng['temp_change_6h'] < 0).astype(int)
        
        # Position in diurnal cycle
        df_eng['hours_since_midnight'] = df_eng.index.hour
        df_eng['hours_to_noon'] = abs(df_eng.index.hour - 12)
        
        # Temperature vs recent average
        df_eng['temp_vs_6h_avg'] = df_eng['temp'] - df_eng['temp_mean_6h']
        df_eng['temp_vs_12h_avg'] = df_eng['temp'] - df_eng['temp_mean_12h']
        df_eng['temp_vs_24h_avg'] = df_eng['temp'] - df_eng['temp_mean_24h']
        
        # Temperature percentile
        for window in [12, 24]:
            df_eng[f'temp_percentile_{window}h'] = df_eng['temp'].rolling(
                window=window, min_periods=1
            ).apply(lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-6) if len(x) > 1 else 0.5, raw=False)
    
    # Weather condition features
    if 'humidity' in df_eng.columns:
        for window in [6, 12, 24]:
            df_eng[f'humidity_mean_{window}h'] = df_eng['humidity'].rolling(window=window, min_periods=1).mean()
            df_eng[f'humidity_std_{window}h'] = df_eng['humidity'].rolling(window=window, min_periods=1).std()
        df_eng['humidity_change_6h'] = df_eng['humidity'] - df_eng['humidity'].shift(6)
        df_eng['humidity_change_12h'] = df_eng['humidity'] - df_eng['humidity'].shift(12)
    
    if 'pressure' in df_eng.columns:
        for window in [6, 12, 24]:
            df_eng[f'pressure_mean_{window}h'] = df_eng['pressure'].rolling(window=window, min_periods=1).mean()
        df_eng['pressure_change_6h'] = df_eng['pressure'] - df_eng['pressure'].shift(6)
        df_eng['pressure_change_12h'] = df_eng['pressure'] - df_eng['pressure'].shift(12)
        df_eng['pressure_change_24h'] = df_eng['pressure'] - df_eng['pressure'].shift(24)
        df_eng['pressure_rapid_drop'] = (df_eng['pressure_change_6h'] < -2).astype(int)
        df_eng['pressure_rapid_rise'] = (df_eng['pressure_change_6h'] > 2).astype(int)
    
    if 'windspeed' in df_eng.columns:
        for window in [6, 12, 24]:
            df_eng[f'windspeed_mean_{window}h'] = df_eng['windspeed'].rolling(window=window, min_periods=1).mean()
            df_eng[f'windspeed_max_{window}h'] = df_eng['windspeed'].rolling(window=window, min_periods=1).max()
        df_eng['windspeed_change_6h'] = df_eng['windspeed'] - df_eng['windspeed'].shift(6)
    
    if 'cloudcover' in df_eng.columns:
        for window in [6, 12, 24]:
            df_eng[f'cloudcover_mean_{window}h'] = df_eng['cloudcover'].rolling(window=window, min_periods=1).mean()
        df_eng['cloudcover_change_6h'] = df_eng['cloudcover'] - df_eng['cloudcover'].shift(6)
        df_eng['cloudcover_change_12h'] = df_eng['cloudcover'] - df_eng['cloudcover'].shift(12)
        df_eng['sky_clearing'] = (df_eng['cloudcover_change_6h'] < -10).astype(int)
        df_eng['sky_clouding'] = (df_eng['cloudcover_change_6h'] > 10).astype(int)
    
    # Interaction features
    if 'humidity' in df_eng.columns and 'temp' in df_eng.columns:
        df_eng['apparent_temp'] = df_eng['temp'] + 0.5 * (df_eng['humidity'] / 100 - 0.1) * (df_eng['temp'] - 14)
        df_eng['apparent_temp_lag_6h'] = df_eng['apparent_temp'].shift(6)
        df_eng['apparent_temp_change_6h'] = df_eng['apparent_temp'] - df_eng['apparent_temp_lag_6h']
    
    if 'humidity' in df_eng.columns and 'windspeed' in df_eng.columns:
        df_eng['humidity_windspeed'] = df_eng['humidity'] * df_eng['windspeed']
    
    if 'pressure' in df_eng.columns and 'humidity' in df_eng.columns:
        df_eng['pressure_humidity'] = df_eng['pressure'] * df_eng['humidity']
    
    if 'windspeed' in df_eng.columns and 'temp' in df_eng.columns:
        df_eng['wind_chill_effect'] = df_eng['windspeed'] * (20 - df_eng['temp'])
    
    # Squared terms
    if 'humidity' in df_eng.columns:
        df_eng['humidity_squared'] = df_eng['humidity'] ** 2
    if 'windspeed' in df_eng.columns:
        df_eng['windspeed_squared'] = df_eng['windspeed'] ** 2
    
    # Categorical encoding
    if 'icon' in df_eng.columns:
        if df_eng['icon'].dtype == 'object':
            icon_dummies = pd.get_dummies(df_eng['icon'], prefix='icon', drop_first=False)
            df_eng = pd.concat([df_eng, icon_dummies], axis=1)
        df_eng = df_eng.drop('icon', axis=1)
    
    # Clean up
    non_numeric = df_eng.select_dtypes(include=['object']).columns.tolist()
    if non_numeric:
        df_eng = df_eng.drop(columns=non_numeric)
    
    # Handle NaN
    if len(df_eng) <= 48:
        df_eng = df_eng.ffill().bfill().fillna(0)
    else:
        df_eng = df_eng.dropna()
    
    return df_eng

print("\n" + "="*80)
print("FEATURE ENGINEERING FOR D+1")
print("="*80)

hourly_train_eng = engineer_features_for_d1(hourly_train)
hourly_val_eng = engineer_features_for_d1(hourly_val)
hourly_test_eng = engineer_features_for_d1(hourly_test)

print(f"Train: {hourly_train_eng.shape}")
print(f"Val:   {hourly_val_eng.shape}")
print(f"Test:  {hourly_test_eng.shape}")

# ============================================================================
# 5. AGGREGATE TO DAILY
# ============================================================================
def aggregate_hourly_to_daily(df_hourly):
    """Aggregate with emphasis on most recent hours"""
    df = df_hourly.copy()
    df['date'] = df.index.date
    
    agg_functions = {}
    for col in df.columns:
        if col == 'date':
            continue
        elif 'sin' in col or 'cos' in col:
            agg_functions[col] = 'mean'
        elif any(stat in col for stat in ['mean', 'std', 'min', 'max', 'range', 'change', 'velocity', 'percentile']):
            agg_functions[col] = 'mean'
        elif col in ['is_weekend', 'day_of_week', 'month', 'day_of_year', 'week_of_year', 'quarter',
                     'is_morning', 'is_afternoon', 'is_evening', 'is_night',
                     'temp_trending_up', 'temp_trending_down', 'pressure_rapid_drop', 
                     'pressure_rapid_rise', 'sky_clearing', 'sky_clouding']:
            agg_functions[col] = 'max'
        else:
            agg_functions[col] = ['mean', 'min', 'max', 'std']
    
    daily_df = df.groupby('date').agg(agg_functions)
    
    if isinstance(daily_df.columns, pd.MultiIndex):
        daily_df.columns = ['_'.join(col).strip('_') for col in daily_df.columns.values]
    
    return daily_df

print("\n" + "="*80)
print("AGGREGATING TO DAILY LEVEL")
print("="*80)

daily_train_features = aggregate_hourly_to_daily(hourly_train_eng)
daily_val_features = aggregate_hourly_to_daily(hourly_val_eng)
daily_test_features = aggregate_hourly_to_daily(hourly_test_eng)

print(f"Train: {daily_train_features.shape}")
print(f"Val:   {daily_val_features.shape}")
print(f"Test:  {daily_test_features.shape}")

# ============================================================================
# 6. PREPARE D+1 TRAINING DATA
# ============================================================================
print("\n" + "="*80)
print("PREPARING D+1 PREDICTION DATA")
print("="*80)

# Shift features by 1 day
train_features = daily_train_features.copy()
val_features = daily_val_features.copy()
test_features = daily_test_features.copy()

train_features.index = pd.to_datetime(train_features.index) + pd.Timedelta(days=1)
val_features.index = pd.to_datetime(val_features.index) + pd.Timedelta(days=1)
test_features.index = pd.to_datetime(test_features.index) + pd.Timedelta(days=1)

train_features.index = train_features.index.date
val_features.index = val_features.index.date
test_features.index = test_features.index.date

# Align columns
all_columns = sorted(list(set(train_features.columns) | set(val_features.columns) | set(test_features.columns)))

for col in all_columns:
    if col not in train_features.columns:
        train_features[col] = 0
    if col not in val_features.columns:
        val_features[col] = 0
    if col not in test_features.columns:
        test_features[col] = 0

train_features = train_features[all_columns].reset_index().rename(columns={'index': 'date'})
val_features = val_features[all_columns].reset_index().rename(columns={'index': 'date'})
test_features = test_features[all_columns].reset_index().rename(columns={'index': 'date'})

# Merge with target
train_data = train_features.merge(daily_target, on='date', how='inner')
val_data = val_features.merge(daily_target, on='date', how='inner')
test_data = test_features.merge(daily_target, on='date', how='inner')

print(f"Train: {train_data.shape}")
print(f"Val:   {val_data.shape}")
print(f"Test:  {test_data.shape}")

train_data = train_data.set_index('date')
val_data = val_data.set_index('date')
test_data = test_data.set_index('date')

# Prepare X and y
TARGET_COL = 'daily_temp'
exclude_cols = [TARGET_COL] + [col for col in train_data.columns if 'temp_diff' in col or ('temp_std' in col and 'd' in col) or ('temp_range' in col and 'd' in col) or ('temp_mean' in col and 'd' in col) or col in ['temp_yesterday', 'temp_persistence', 'day_of_week', 'is_weekend']]

X_train = train_data.drop(columns=exclude_cols, errors='ignore')
y_train = train_data[TARGET_COL]
X_val = val_data.drop(columns=exclude_cols, errors='ignore')
y_val = val_data[TARGET_COL]
X_test = test_data.drop(columns=exclude_cols, errors='ignore')
y_test = test_data[TARGET_COL]

print(f"\nFeature matrix: {X_train.shape[1]} features")

# Scale
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# 7. TRAIN D+1 MODEL WITH OPTIMIZATION
# ============================================================================
print("\n" + "="*80)
print("TRAINING D+1 MODEL")
print("="*80)

if TUNE_HYPERPARAMETERS and OPTUNA_AVAILABLE:
    print(f"Using Optuna optimization with {N_TRIALS} trials...")
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 0.3),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 0.5),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 2.0),
            'objective': 'reg:squarederror',
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
        
        model = XGBRegressor(**params)
        model.fit(X_train_scaled, y_train, 
                 eval_set=[(X_val_scaled, y_val)],
                 verbose=False)
        
        y_pred = model.predict(X_val_scaled)
        mae = mean_absolute_error(y_val, y_pred)
        
        return mae
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    
    best_params = study.best_params
    best_params.update({
        'objective': 'reg:squarederror',
        'random_state': 42,
        'n_jobs': -1
    })
    
    print(f"\nBest validation MAE: {study.best_value:.4f}°C")
    
    model = XGBRegressor(**best_params)
    model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)
else:
    print("Training with default parameters optimized for D+1...")
    model = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=250,
        learning_rate=0.08,
        max_depth=6,
        min_child_weight=1,
        subsample=0.9,
        colsample_bytree=0.9,
        gamma=0.05,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)

# ============================================================================
# 8. EVALUATE MODEL
# ============================================================================
print("\n" + "="*80)
print("D+1 MODEL PERFORMANCE")
print("="*80)

y_train_pred = model.predict(X_train_scaled)
y_val_pred = model.predict(X_val_scaled)
y_test_pred = model.predict(X_test_scaled)

results = []
for split_name, y_true, y_pred in [
    ('Train', y_train, y_train_pred),
    ('Val', y_val, y_val_pred),
    ('Test', y_test, y_test_pred)
]:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)
    
    results.append({
        'Split': split_name,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R²': r2,
        'N': len(y_true)
    })

results_df = pd.DataFrame(results)
print("\n" + results_df.to_string(index=False))

val_metrics = results_df[results_df['Split'] == 'Val'].iloc[0]

# ============================================================================
# 9. SAVE MODEL ARTIFACTS
# ============================================================================
print("\n" + "="*80)
print("SAVING MODEL ARTIFACTS")
print("="*80)

# Save XGBoost model
model_path = os.path.join(SAVE_DIR, f"{MODEL_NAME}.json")
model.save_model(model_path)
print(f"✓ XGBoost model saved: {model_path}")

# Save scaler
scaler_path = os.path.join(SAVE_DIR, f"{MODEL_NAME}_scaler.pkl")
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"✓ Scaler saved: {scaler_path}")

# Save feature names
feature_names_path = os.path.join(SAVE_DIR, f"{MODEL_NAME}_features.json")
with open(feature_names_path, 'w') as f:
    json.dump(list(X_train.columns), f)
print(f"✓ Feature names saved: {feature_names_path}")

# Save metadata
metadata = {
    'model_name': MODEL_NAME,
    'forecast_horizon': 'D+1',
    'n_features': X_train.shape[1],
    'training_samples': len(X_train),
    'validation_mae': float(val_metrics['MAE']),
    'validation_rmse': float(val_metrics['RMSE']),
    'validation_r2': float(val_metrics['R²']),
    'validation_mape': float(val_metrics['MAPE']),
    'daily_calc_method': DAILY_CALC_METHOD
}
metadata_path = os.path.join(SAVE_DIR, f"{MODEL_NAME}_metadata.json")
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"✓ Metadata saved: {metadata_path}")

# ============================================================================
# 10. EXPORT TO ONNX
# ============================================================================
if ONNX_AVAILABLE:
    print("\n" + "="*80)
    print("EXPORTING TO ONNX FORMAT")
    print("="*80)
    
    try:
        # Define input type
        initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
        
        # Convert to ONNX
        onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_type)
        
        # Save ONNX model
        onnx_path = os.path.join(SAVE_DIR, f"{MODEL_NAME}.onnx")
        onnx.save_model(onnx_model, onnx_path)
        print(f"✓ ONNX model saved: {onnx_path}")
        
        # Verify ONNX model
        onnx_model_check = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model_check)
        print("✓ ONNX model verification passed")
        
        # Test ONNX inference
        try:
            import onnxruntime as rt
            sess = rt.InferenceSession(onnx_path)
            input_name = sess.get_inputs()[0].name
            output_name = sess.get_outputs()[0].name
            
            # Test with first sample
            test_sample = X_val_scaled[:1].astype(np.float32)
            onnx_pred = sess.run([output_name], {input_name: test_sample})[0]
            xgb_pred = model.predict(test_sample)
            
            # Compare predictions
            diff = abs(onnx_pred[0] - xgb_pred[0])
            print(f"✓ ONNX inference test passed (diff: {diff:.6f})")
            
        except ImportError:
            print("  Note: Install onnxruntime to test inference: pip install onnxruntime")
    
    except Exception as e:
        print(f"✗ ONNX export failed: {str(e)}")
        print("  This is non-critical. XGBoost model is still saved.")
else:
    print("\n✗ ONNX export skipped (libraries not available)")
    print("  Install with: pip install onnx onnxmltools onnxruntime")

print("\n" + "="*80)
print("MODEL TRAINING COMPLETE")
print("="*80)
print(f"\nModel files saved in: {SAVE_DIR}")
print(f"  - XGBoost model: {MODEL_NAME}.json")
print(f"  - Scaler: {MODEL_NAME}_scaler.pkl")
print(f"  - Feature names: {MODEL_NAME}_features.json")
print(f"  - Metadata: {MODEL_NAME}_metadata.json")
if ONNX_AVAILABLE:
    print(f"  - ONNX model: {MODEL_NAME}.onnx")
print(f"\nValidation MAE: {val_metrics['MAE']:.4f}°C")
print(f"Validation R²: {val_metrics['R²']:.4f}")
print("="*80)