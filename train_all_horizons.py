"""
Unified Temperature Forecasting Model Training
Trains all 5 forecast horizons (D+1 through D+5) in one script
With ONNX export and complete model persistence
"""
import sys
import io

# Force UTF-8 encoding for stdout and stderr on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import warnings
import pickle
import json
import os
import time
from datetime import datetime
warnings.filterwarnings('ignore')

# Try to import optional libraries
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
    print("Warning: ONNX libraries not available.")

try:
    import onnxruntime as rt
    ONNX_RUNTIME_AVAILABLE = True
except ImportError:
    ONNX_RUNTIME_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================
TUNE_HYPERPARAMETERS = True  # Set to False for faster training
N_TRIALS = 50  # Number of Optuna trials per model
HORIZONS_TO_TRAIN = ['d1', 'd2', 'd3', 'd4', 'd5']  # Which models to train
BASE_SAVE_DIR = "./models/"

# Model configurations for each horizon
HORIZON_CONFIGS = {
    'd1': {
        'name': 'D+1 (Tomorrow)',
        'shift_days': 1,
        'model_name': 'd1_temperature_model',
        'save_dir': 'd1/',
        'default_params': {
            'n_estimators': 250,
            'learning_rate': 0.08,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'gamma': 0.05,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
        },
        'expected_mae': '0.4-0.6°C',
        'lags': [1, 2, 3, 4, 6, 8, 12, 18, 24],
        'windows': [3, 6, 12, 18, 24]
    },
    'd2': {
        'name': 'D+2 (Day After Tomorrow)',
        'shift_days': 2,
        'model_name': 'd2_temperature_model',
        'save_dir': 'd2/',
        'default_params': {
            'n_estimators': 300,
            'learning_rate': 0.08,
            'max_depth': 7,
            'min_child_weight': 2,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'gamma': 0.1,
            'reg_alpha': 0.2,
            'reg_lambda': 1.5,
        },
        'expected_mae': '0.6-0.8°C',
        'lags': [1, 2, 3, 6, 12, 24, 36, 48],
        'windows': [12, 24, 36, 48, 72]
    },
    'd3': {
        'name': 'D+3 (3 Days Ahead)',
        'shift_days': 3,
        'model_name': 'd3_temperature_model',
        'save_dir': 'd3/',
        'default_params': {
            'n_estimators': 300,
            'learning_rate': 0.07,
            'max_depth': 7,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.2,
            'reg_alpha': 0.3,
            'reg_lambda': 2.0,
        },
        'expected_mae': '0.8-1.1°C',
        'lags': [12, 24, 36, 48, 60, 72, 168],
        'windows': [24, 36, 48, 72, 96, 120, 168]
    },
    'd4': {
        'name': 'D+4 (4 Days Ahead)',
        'shift_days': 4,
        'model_name': 'd4_temperature_model',
        'save_dir': 'd4/',
        'default_params': {
            'n_estimators': 400,
            'learning_rate': 0.05,
            'max_depth': 8,
            'min_child_weight': 4,
            'subsample': 0.75,
            'colsample_bytree': 0.75,
            'gamma': 0.3,
            'reg_alpha': 0.5,
            'reg_lambda': 2.5,
        },
        'expected_mae': '1.0-1.3°C',
        'lags': [48, 72, 96, 120, 144, 168, 336],
        'windows': [48, 72, 96, 120, 168, 240, 336]
    },
    'd5': {
        'name': 'D+5 (5 Days Ahead)',
        'shift_days': 5,
        'model_name': 'd5_temperature_model',
        'save_dir': 'd5/',
        'default_params': {
            'n_estimators': 500,
            'learning_rate': 0.03,
            'max_depth': 7,
            'min_child_weight': 7,
            'subsample': 0.65,
            'colsample_bytree': 0.65,
            'gamma': 0.5,
            'reg_alpha': 1.0,
            'reg_lambda': 3.0,
        },
        'expected_mae': '1.2-1.5°C',
        'lags': [120, 168, 240, 336, 504, 720],
        'windows': [168, 336, 504, 720]
    }
}

print("="*80)
print("UNIFIED TEMPERATURE FORECASTING MODEL TRAINING")
print("="*80)
print(f"Training horizons: {', '.join(HORIZONS_TO_TRAIN)}")
print(f"Hyperparameter tuning: {'Enabled' if TUNE_HYPERPARAMETERS else 'Disabled'}")
print(f"ONNX export: {'Enabled' if ONNX_AVAILABLE else 'Disabled'}")
print("="*80)

# ============================================================================
# DATA LOADING
# ============================================================================
def load_data():
    """Load and prepare data"""
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    hourly = pd.read_csv('https://raw.githubusercontent.com/7hufofbun/DSEB--Machine-Learning_Group5/main/data/weather_hcm_hourly.csv')
    daily = pd.read_csv('https://raw.githubusercontent.com/7hufofbun/DSEB--Machine-Learning_Group5/main/data/weather_hcm_daily.csv')
    
    hourly['datetime'] = pd.to_datetime(hourly['datetime'])
    daily['datetime'] = pd.to_datetime(daily['datetime'])
    
    print(f"Hourly data: {hourly.shape}")
    print(f"Daily data: {daily.shape}")
    print(f"Date range: {hourly['datetime'].min()} to {hourly['datetime'].max()}")
    
    # Analyze daily calculation method
    hourly['date'] = hourly['datetime'].dt.date
    sample_dates = hourly['date'].unique()[:10]
    
    comparison_results = []
    for date in sample_dates:
        day_data = hourly[hourly['date'] == date]
        if len(day_data) >= 20:
            hourly_mean = day_data['temp'].mean()
            max_min_mean = (day_data['temp'].max() + day_data['temp'].min()) / 2
            daily_temp = daily[daily['datetime'].dt.date == date]['temp'].values
            
            if len(daily_temp) > 0:
                comparison_results.append({
                    'diff_hourly': abs(daily_temp[0] - hourly_mean),
                    'diff_maxmin': abs(daily_temp[0] - max_min_mean)
                })
    
    comp_df = pd.DataFrame(comparison_results)
    daily_calc_method = "hourly_mean" if comp_df['diff_hourly'].mean() < comp_df['diff_maxmin'].mean() else "max_min"
    print(f"Daily calculation method: {daily_calc_method.upper()}")
    
    return hourly, daily, daily_calc_method

# ============================================================================
# DATA SPLITTING
# ============================================================================
def split_data(hourly, train_ratio=0.7, val_ratio=0.15):
    """Split data into train/val/test"""
    text_columns = ['name', 'address', 'resolvedAddress', 'conditions', 'source',
                    'latitude', 'longitude', 'preciptype', 'severerisk']
    hourly = hourly.drop(columns=[col for col in text_columns if col in hourly.columns])
    
    hourly = hourly.sort_values('datetime')
    n = len(hourly)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    hourly_train = hourly.iloc[:train_size].copy()
    hourly_val = hourly.iloc[train_size:train_size + val_size].copy()
    hourly_test = hourly.iloc[train_size + val_size:].copy()
    
    print(f"\nData split:")
    print(f"  Train: {len(hourly_train)//24} days ({len(hourly_train)} hours)")
    print(f"  Val:   {len(hourly_val)//24} days ({len(hourly_val)} hours)")
    print(f"  Test:  {len(hourly_test)//24} days ({len(hourly_test)} hours)")
    
    return hourly_train, hourly_val, hourly_test

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
def engineer_features(df, config):
    """Universal feature engineering based on horizon configuration"""
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
    df_eng['day_of_month'] = df_eng.index.day
    
    # Weekly cycle indicators (important for D+3 and beyond)
    df_eng['is_weekday'] = (df_eng.index.dayofweek < 5).astype(int)
    for i in range(7):
        df_eng[f'is_day_{i}'] = (df_eng.index.dayofweek == i).astype(int)
    
    # Cyclical encoding
    df_eng['hour_sin'] = np.sin(2 * np.pi * df_eng.index.hour / 24)
    df_eng['hour_cos'] = np.cos(2 * np.pi * df_eng.index.hour / 24)
    df_eng['day_of_week_sin'] = np.sin(2 * np.pi * df_eng.index.dayofweek / 7)
    df_eng['day_of_week_cos'] = np.cos(2 * np.pi * df_eng.index.dayofweek / 7)
    df_eng['month_sin'] = np.sin(2 * np.pi * df_eng.index.month / 12)
    df_eng['month_cos'] = np.cos(2 * np.pi * df_eng.index.month / 12)
    df_eng['day_of_year_sin'] = np.sin(2 * np.pi * df_eng.index.dayofyear / 365)
    df_eng['day_of_year_cos'] = np.cos(2 * np.pi * df_eng.index.dayofyear / 365)
    
    # Temperature features based on horizon
    if 'temp' in df_eng.columns:
        # Lags specific to horizon
        for lag in config['lags']:
            df_eng[f'temp_lag_{lag}h'] = df_eng['temp'].shift(lag)
        
        # Rolling windows specific to horizon
        for window in config['windows']:
            df_eng[f'temp_mean_{window}h'] = df_eng['temp'].rolling(window=window, min_periods=1).mean()
            df_eng[f'temp_std_{window}h'] = df_eng['temp'].rolling(window=window, min_periods=1).std()
            df_eng[f'temp_min_{window}h'] = df_eng['temp'].rolling(window=window, min_periods=1).min()
            df_eng[f'temp_max_{window}h'] = df_eng['temp'].rolling(window=window, min_periods=1).max()
            df_eng[f'temp_range_{window}h'] = df_eng[f'temp_max_{window}h'] - df_eng[f'temp_min_{window}h']
        
        # Temperature changes
        for lag in config['lags'][:5]:  # Use first 5 lags for changes
            df_eng[f'temp_change_{lag}h'] = df_eng['temp'] - df_eng['temp'].shift(lag)
        
        # Temperature momentum (rate of change)
        if 24 in config['lags']:
            df_eng['temp_momentum_24h'] = df_eng['temp'] - df_eng['temp'].shift(24)
        if 72 in config['lags']:
            df_eng['temp_momentum_72h'] = (df_eng['temp'] - df_eng['temp'].shift(72)) / 3
        if 168 in config['lags']:
            df_eng['temp_momentum_168h'] = (df_eng['temp'] - df_eng['temp'].shift(168)) / 7
        
        # Temperature vs recent averages
        for window in config['windows'][:3]:  # First 3 windows
            df_eng[f'temp_vs_{window}h_avg'] = df_eng['temp'] - df_eng[f'temp_mean_{window}h']
        
        # Temperature stability (coefficient of variation)
        for window in config['windows'][-2:]:  # Last 2 windows
            mean = df_eng[f'temp_mean_{window}h']
            std = df_eng[f'temp_std_{window}h']
            df_eng[f'temp_cv_{window}h'] = std / (mean + 1e-6)
    
    # Weather condition features
    if 'humidity' in df_eng.columns:
        for window in config['windows'][:3]:
            df_eng[f'humidity_mean_{window}h'] = df_eng['humidity'].rolling(window=window, min_periods=1).mean()
            df_eng[f'humidity_std_{window}h'] = df_eng['humidity'].rolling(window=window, min_periods=1).std()
    
    if 'pressure' in df_eng.columns:
        for window in config['windows'][:3]:
            df_eng[f'pressure_mean_{window}h'] = df_eng['pressure'].rolling(window=window, min_periods=1).mean()
        for lag in config['lags'][:3]:
            if lag >= 6:
                df_eng[f'pressure_change_{lag}h'] = df_eng['pressure'] - df_eng['pressure'].shift(lag)
    
    if 'windspeed' in df_eng.columns:
        for window in config['windows'][:3]:
            df_eng[f'windspeed_mean_{window}h'] = df_eng['windspeed'].rolling(window=window, min_periods=1).mean()
            df_eng[f'windspeed_max_{window}h'] = df_eng['windspeed'].rolling(window=window, min_periods=1).max()
    
    if 'cloudcover' in df_eng.columns:
        for window in config['windows'][:3]:
            df_eng[f'cloudcover_mean_{window}h'] = df_eng['cloudcover'].rolling(window=window, min_periods=1).mean()
    
    # Interaction features
    if 'humidity' in df_eng.columns and 'temp' in df_eng.columns:
        df_eng['apparent_temp'] = df_eng['temp'] + 0.5 * (df_eng['humidity'] / 100 - 0.1) * (df_eng['temp'] - 14)
    
    if 'humidity' in df_eng.columns and 'windspeed' in df_eng.columns:
        df_eng['humidity_windspeed'] = df_eng['humidity'] * df_eng['windspeed']
    
    if 'pressure' in df_eng.columns and 'humidity' in df_eng.columns:
        df_eng['pressure_humidity'] = df_eng['pressure'] * df_eng['humidity']
    
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
    max_lag = max(config['lags']) if config['lags'] else 24
    if len(df_eng) <= max_lag * 2:
        df_eng = df_eng.ffill().bfill().fillna(0)
    else:
        df_eng = df_eng.dropna()
    
    return df_eng

# ============================================================================
# AGGREGATION
# ============================================================================
def aggregate_hourly_to_daily(df_hourly):
    """Aggregate hourly data to daily"""
    df = df_hourly.copy()
    df['date'] = df.index.date
    
    agg_functions = {}
    for col in df.columns:
        if col == 'date':
            continue
        elif 'sin' in col or 'cos' in col:
            agg_functions[col] = 'mean'
        elif any(stat in col for stat in ['mean', 'std', 'min', 'max', 'range', 'change', 'momentum', 'vs', 'cv']):
            agg_functions[col] = 'mean'
        elif col in (['is_weekend', 'is_weekday'] + [f'is_day_{i}' for i in range(7)] + 
                     ['day_of_week', 'month', 'day_of_year', 'week_of_year', 'quarter', 'day_of_month']):
            agg_functions[col] = 'first'
        else:
            agg_functions[col] = ['mean', 'min', 'max', 'std']
    
    daily_df = df.groupby('date').agg(agg_functions)
    
    if isinstance(daily_df.columns, pd.MultiIndex):
        daily_df.columns = ['_'.join(col).strip('_') for col in daily_df.columns.values]
    
    return daily_df

# ============================================================================
# PREPARE DAILY TARGET
# ============================================================================
def prepare_daily_target(daily):
    """Prepare daily target with basic features"""
    daily_target = daily[['datetime', 'temp']].copy()
    daily_target.columns = ['date', 'daily_temp']
    daily_target['date'] = pd.to_datetime(daily_target['date'])
    daily_target = daily_target.sort_values('date').reset_index(drop=True)
    daily_target['date'] = daily_target['date'].dt.date
    return daily_target

# ============================================================================
# HYPERPARAMETER OPTIMIZATION
# ============================================================================
def optimize_hyperparameters(X_train, y_train, X_val, y_val, n_trials, horizon_name):
    """Optimize hyperparameters using Optuna"""
    print(f"  Optimizing hyperparameters ({n_trials} trials)...")
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 600),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 0.8),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 2.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 5.0),
            'objective': 'reg:squarederror',
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
        
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        y_pred = model.predict(X_val)
        
        return mean_absolute_error(y_val, y_pred)
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    best_params = study.best_params
    best_params.update({
        'objective': 'reg:squarederror',
        'random_state': 42,
        'n_jobs': -1
    })
    
    print(f"  Best validation MAE: {study.best_value:.4f}°C")
    
    return best_params

# ============================================================================
# MODEL EVALUATION
# ============================================================================
def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test):
    """Evaluate model performance"""
    results = []
    
    for split_name, X, y_true in [
        ('Train', X_train, y_train),
        ('Val', X_val, y_val),
        ('Test', X_test, y_test)
    ]:
        y_pred = model.predict(X)
        
        results.append({
            'Split': split_name,
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100,
            'R2': r2_score(y_true, y_pred),
            'N': len(y_true)
        })
    
    return pd.DataFrame(results)

# ============================================================================
# MODEL SAVING
# ============================================================================
def save_model_artifacts(model, scaler, feature_names, metadata, save_dir, model_name):
    """Save all model artifacts including ONNX"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save XGBoost model
    model_path = os.path.join(save_dir, f"{model_name}.json")
    model.save_model(model_path)
    
    # Save scaler
    scaler_path = os.path.join(save_dir, f"{model_name}_scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save feature names
    features_path = os.path.join(save_dir, f"{model_name}_features.json")
    with open(features_path, 'w') as f:
        json.dump(feature_names, f)
    
    # Save metadata
    metadata_path = os.path.join(save_dir, f"{model_name}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    paths = {
        'xgboost': model_path,
        'scaler': scaler_path,
        'features': features_path,
        'metadata': metadata_path
    }
    
    # Export to ONNX
    if ONNX_AVAILABLE:
        try:
            initial_type = [('float_input', FloatTensorType([None, len(feature_names)]))]
            onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_type)
            onnx_path = os.path.join(save_dir, f"{model_name}.onnx")
            onnx.save_model(onnx_model, onnx_path)
            paths['onnx'] = onnx_path
            
            # Verify
            onnx_model_check = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model_check)
        except Exception as e:
            print(f"    Warning: ONNX export failed - {str(e)}")
    
    return paths

# ============================================================================
# TRAIN SINGLE HORIZON
# ============================================================================
def train_horizon(horizon_id, hourly_train, hourly_val, hourly_test, daily_target, daily_calc_method):
    """Train model for a specific horizon"""
    config = HORIZON_CONFIGS[horizon_id]
    
    print("\n" + "="*80)
    print(f"TRAINING {config['name'].upper()}")
    print("="*80)
    print(f"Shift: {config['shift_days']} days")
    print(f"Expected MAE: {config['expected_mae']}")
    print(f"Lags: {config['lags']}")
    print(f"Windows: {config['windows']}")
    
    start_time = time.time()
    
    # Feature engineering
    print("\n  Feature engineering...")
    hourly_train_eng = engineer_features(hourly_train, config)
    hourly_val_eng = engineer_features(hourly_val, config)
    hourly_test_eng = engineer_features(hourly_test, config)
    
    print(f"  Engineered features: {hourly_train_eng.shape[1]}")
    
    # Aggregate to daily
    print("  Aggregating to daily...")
    daily_train_features = aggregate_hourly_to_daily(hourly_train_eng)
    daily_val_features = aggregate_hourly_to_daily(hourly_val_eng)
    daily_test_features = aggregate_hourly_to_daily(hourly_test_eng)
    
    # Shift features by forecast horizon
    train_features = daily_train_features.copy()
    val_features = daily_val_features.copy()
    test_features = daily_test_features.copy()
    
    shift_days = config['shift_days']
    train_features.index = pd.to_datetime(train_features.index) + pd.Timedelta(days=shift_days)
    val_features.index = pd.to_datetime(val_features.index) + pd.Timedelta(days=shift_days)
    test_features.index = pd.to_datetime(test_features.index) + pd.Timedelta(days=shift_days)
    
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
    
    train_data = train_data.set_index('date')
    val_data = val_data.set_index('date')
    test_data = test_data.set_index('date')
    
    # Prepare X and y
    TARGET_COL = 'daily_temp'
    X_train = train_data.drop(columns=[TARGET_COL], errors='ignore')
    y_train = train_data[TARGET_COL]
    X_val = val_data.drop(columns=[TARGET_COL], errors='ignore')
    y_val = val_data[TARGET_COL]
    X_test = test_data.drop(columns=[TARGET_COL], errors='ignore')
    y_test = test_data[TARGET_COL]
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Features: {X_train.shape[1]}")
    
    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("\n  Training model...")
    if TUNE_HYPERPARAMETERS and OPTUNA_AVAILABLE:
        params = optimize_hyperparameters(X_train_scaled, y_train, X_val_scaled, y_val, N_TRIALS, config['name'])
    else:
        params = config['default_params'].copy()
        params.update({
            'objective': 'reg:squarederror',
            'random_state': 42,
            'n_jobs': -1
        })
    
    model = XGBRegressor(**params)
    model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)
    
    # Evaluate
    print("\n  Evaluating...")
    results_df = evaluate_model(model, X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test)
    print("\n" + results_df.to_string(index=False))
    
    val_metrics = results_df[results_df['Split'] == 'Val'].iloc[0]
    
    # Save model
    save_dir = os.path.join(BASE_SAVE_DIR, config['save_dir'])
    print(f"\n  Saving to {save_dir}...")
    
    metadata = {
        'model_name': config['model_name'],
        'forecast_horizon': horizon_id.upper(),
        'horizon_name': config['name'],
        'shift_days': config['shift_days'],
        'n_features': X_train.shape[1],
        'training_samples': len(X_train),
        'validation_mae': float(val_metrics['MAE']),
        'validation_rmse': float(val_metrics['RMSE']),
        'validation_r2': float(val_metrics['R2']),
        'validation_mape': float(val_metrics['MAPE']),
        'daily_calc_method': daily_calc_method,
        'hyperparameters': params,
        'lags_used': config['lags'],
        'windows_used': config['windows'],
        'training_time_seconds': time.time() - start_time,
        'trained_at': datetime.now().isoformat()
    }
    
    paths = save_model_artifacts(model, scaler, list(X_train.columns), metadata, save_dir, config['model_name'])
    
    elapsed = time.time() - start_time
    print(f"  ✓ Training complete in {elapsed:.1f} seconds")
    print(f"  ✓ Validation MAE: {val_metrics['MAE']:.4f}°C")
    print(f"  ✓ Validation R²: {val_metrics['R2']:.4f}")
    
    return {
        'horizon': horizon_id,
        'config': config,
        'model': model,
        'scaler': scaler,
        'results': results_df,
        'metrics': val_metrics,
        'metadata': metadata,
        'paths': paths,
        'training_time': elapsed
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Main execution function"""
    overall_start = time.time()
    
    # Load data
    hourly, daily, daily_calc_method = load_data()
    
    # Split data
    hourly_train, hourly_val, hourly_test = split_data(hourly)
    
    # Prepare daily target
    daily_target = prepare_daily_target(daily)
    
    # Train all horizons
    print("\n" + "="*80)
    print("TRAINING ALL HORIZONS")
    print("="*80)
    
    all_results = {}
    
    for horizon_id in HORIZONS_TO_TRAIN:
        try:
            result = train_horizon(
                horizon_id, 
                hourly_train, 
                hourly_val, 
                hourly_test, 
                daily_target, 
                daily_calc_method
            )
            all_results[horizon_id] = result
        except Exception as e:
            print(f"\n  ✗ Error training {horizon_id}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    
    summary_data = []
    for horizon_id, result in all_results.items():
        metrics = result['metrics']
        summary_data.append({
            'Horizon': result['config']['name'],
            'MAE': f"{metrics['MAE']:.4f}",
            'RMSE': f"{metrics['RMSE']:.4f}",
            'MAPE': f"{metrics['MAPE']:.2f}%",
            'R²': f"{metrics['R2']:.4f}",
            'Time': f"{result['training_time']:.1f}s",
            'Samples': result['metadata']['training_samples'],
            'Features': result['metadata']['n_features']
        })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        print("\n" + summary_df.to_string(index=False))
    
    total_time = time.time() - overall_start
    
    print("\n" + "="*80)
    print("ALL MODELS TRAINED SUCCESSFULLY!")
    print("="*80)
    print(f"Total training time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Models saved in: {BASE_SAVE_DIR}")
    print(f"Horizons trained: {len(all_results)}/{len(HORIZONS_TO_TRAIN)}")
    
    # Save training summary
    summary_path = os.path.join(BASE_SAVE_DIR, 'training_summary.json')
    summary_info = {
        'trained_at': datetime.now().isoformat(),
        'total_time_seconds': total_time,
        'horizons_trained': list(all_results.keys()),
        'tuning_enabled': TUNE_HYPERPARAMETERS,
        'n_trials': N_TRIALS if TUNE_HYPERPARAMETERS else 0,
        'models': {
            h: {
                'mae': float(r['metrics']['MAE']),
                'rmse': float(r['metrics']['RMSE']),
                'r2': float(r['metrics']['R2']),
                'training_time': r['training_time']
            } for h, r in all_results.items()
        }
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary_info, f, indent=2)
    
    print(f"\nTraining summary saved: {summary_path}")
    print("="*80)
    
    return all_results

if __name__ == "__main__":
    print(f"\nStarting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    results = main()
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")