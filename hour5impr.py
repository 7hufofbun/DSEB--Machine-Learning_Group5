import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import warnings
import pickle
import json
import os
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

try:
    import onnxruntime as rt
    ONNX_RUNTIME_AVAILABLE = True
except ImportError:
    ONNX_RUNTIME_AVAILABLE = False

# ============================================================================
# CONFIGURATION - SPECIALIZED FOR D+5
# ============================================================================
TUNE_HYPERPARAMETERS = True
N_TRIALS = 60
MODEL_NAME = "d5_temperature_model"
SAVE_DIR = "./models/d5/"

print("="*80)
print("SPECIALIZED D+5 (5 DAYS AHEAD) TEMPERATURE PREDICTION")
print("="*80)
print("Focus: 120-hour extended forecast (limit of statistical skill)")
print("Strategy: Climatology, monthly patterns, regime persistence")
print("="*80)

os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n" + "="*80)
print("LOADING DATA")
print("="*80)

hourly = pd.read_csv('https://raw.githubusercontent.com/7hufofbun/DSEB--Machine-Learning_Group5/main/data/weather_hcm_hourly.csv')
daily = pd.read_csv('https://raw.githubusercontent.com/7hufofbun/DSEB--Machine-Learning_Group5/main/data/weather_hcm_daily.csv')

hourly['datetime'] = pd.to_datetime(hourly['datetime'])
daily['datetime'] = pd.to_datetime(daily['datetime'])

print(f"Hourly: {hourly.shape}")
print(f"Daily: {daily.shape}")

# Analyze daily calculation
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
DAILY_CALC_METHOD = "hourly_mean" if comp_df['diff_hourly'].mean() < comp_df['diff_maxmin'].mean() else "max_min"
print(f"Daily calculation: {DAILY_CALC_METHOD.upper()}")

# ============================================================================
# 2. PREPARE DAILY TARGET WITH D+5 CLIMATOLOGICAL FEATURES
# ============================================================================
print("\n" + "="*80)
print("PREPARING CLIMATOLOGICAL FEATURES")
print("="*80)

daily_target = daily[['datetime', 'temp']].copy()
daily_target.columns = ['date', 'daily_temp']
daily_target['date'] = pd.to_datetime(daily_target['date'])
daily_target = daily_target.sort_values('date').reset_index(drop=True)

# Very long-term patterns
for lag in [5, 7, 14, 21, 30]:
    daily_target[f'temp_{lag}d_ago'] = daily_target['daily_temp'].shift(lag)
    daily_target[f'temp_diff_{lag}d'] = daily_target['daily_temp'].diff(lag)

# Extended windows
for window in [7, 10, 14, 21, 30]:
    daily_target[f'temp_std_{window}d'] = daily_target['daily_temp'].rolling(window=window, min_periods=1).std()
    daily_target[f'temp_range_{window}d'] = (
        daily_target['daily_temp'].rolling(window=window, min_periods=1).max() -
        daily_target['daily_temp'].rolling(window=window, min_periods=1).min())
    daily_target[f'temp_mean_{window}d'] = daily_target['daily_temp'].rolling(window=window, min_periods=1).mean()

# Calendar features
daily_target['day_of_week'] = daily_target['date'].dt.dayofweek
daily_target['day_of_month'] = daily_target['date'].dt.day
daily_target['month'] = daily_target['date'].dt.month
daily_target['is_weekend'] = (daily_target['day_of_week'] >= 5).astype(int)
for i in range(7):
    daily_target[f'is_day_{i}'] = (daily_target['day_of_week'] == i).astype(int)

daily_target['date'] = daily_target['date'].dt.date
print(f"Daily target: {daily_target.shape}")

# ============================================================================
# 3. SPLIT DATA
# ============================================================================
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

print(f"\nTrain: {len(hourly_train)//24} days | Val: {len(hourly_val)//24} days | Test: {len(hourly_test)//24} days")

# ============================================================================
# 4. FEATURE ENGINEERING FOR D+5
# ============================================================================
def engineer_features_for_d5(df):
    """Climatological feature engineering for D+5"""
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
    df_eng['week_of_year'] = df_eng.index.isocalendar().week
    df_eng['day_of_month'] = df_eng.index.day
    
    # Weekly cycle
    df_eng['is_weekend'] = (df_eng.index.dayofweek >= 5).astype(int)
    for i in range(7):
        df_eng[f'is_day_{i}'] = (df_eng.index.dayofweek == i).astype(int)
    
    # Strong cyclical encoding
    df_eng['hour_sin'] = np.sin(2 * np.pi * df_eng.index.hour / 24)
    df_eng['hour_cos'] = np.cos(2 * np.pi * df_eng.index.hour / 24)
    df_eng['day_of_week_sin'] = np.sin(2 * np.pi * df_eng.index.dayofweek / 7)
    df_eng['day_of_week_cos'] = np.cos(2 * np.pi * df_eng.index.dayofweek / 7)
    df_eng['month_sin'] = np.sin(2 * np.pi * df_eng.index.month / 12)
    df_eng['month_cos'] = np.cos(2 * np.pi * df_eng.index.month / 12)
    df_eng['day_of_year_sin'] = np.sin(2 * np.pi * df_eng.index.dayofyear / 365)
    df_eng['day_of_year_cos'] = np.cos(2 * np.pi * df_eng.index.dayofyear / 365)
    
    # Temperature features - very extended lags
    if 'temp' in df_eng.columns:
        # Lags up to 30 days
        for lag in [120, 168, 240, 336, 504, 720]:  # 5d, 7d, 10d, 14d, 21d, 30d
            df_eng[f'temp_lag_{lag}h'] = df_eng['temp'].shift(lag)
        
        # Very long rolling windows
        for window in [168, 336, 504, 720]:  # 7d, 14d, 21d, 30d
            df_eng[f'temp_mean_{window}h'] = df_eng['temp'].rolling(window=window, min_periods=1).mean()
            df_eng[f'temp_std_{window}h'] = df_eng['temp'].rolling(window=window, min_periods=1).std()
            df_eng[f'temp_min_{window}h'] = df_eng['temp'].rolling(window=window, min_periods=1).min()
            df_eng[f'temp_max_{window}h'] = df_eng['temp'].rolling(window=window, min_periods=1).max()
            df_eng[f'temp_range_{window}h'] = df_eng[f'temp_max_{window}h'] - df_eng[f'temp_min_{window}h']
        
        # Long-term trends
        for lag in [168, 336, 504, 720]:
            df_eng[f'temp_trend_{lag}h'] = df_eng['temp'] - df_eng['temp'].shift(lag)
        
        # Multi-week momentum
        df_eng['temp_momentum_7d'] = (df_eng['temp'] - df_eng['temp'].shift(168)) / 7
        df_eng['temp_momentum_14d'] = (df_eng['temp'] - df_eng['temp'].shift(336)) / 14
        df_eng['temp_momentum_30d'] = (df_eng['temp'] - df_eng['temp'].shift(720)) / 30
        
        # Weekly comparisons
        df_eng['temp_vs_1w'] = df_eng['temp'] - df_eng['temp_lag_168h']
        df_eng['temp_vs_2w'] = df_eng['temp'] - df_eng['temp_lag_336h']
        
        # Anomaly detection
        for window in [336, 720]:
            mean = df_eng[f'temp_mean_{window}h']
            std = df_eng[f'temp_std_{window}h']
            df_eng[f'temp_zscore_{window}h'] = (df_eng['temp'] - mean) / (std + 1e-6)
    
    # Extended weather patterns
    if 'pressure' in df_eng.columns:
        for window in [168, 336, 720]:
            df_eng[f'pressure_mean_{window}h'] = df_eng['pressure'].rolling(window=window, min_periods=1).mean()
            df_eng[f'pressure_std_{window}h'] = df_eng['pressure'].rolling(window=window, min_periods=1).std()
    
    if 'humidity' in df_eng.columns:
        for window in [168, 336]:
            df_eng[f'humidity_mean_{window}h'] = df_eng['humidity'].rolling(window=window, min_periods=1).mean()
    
    if 'windspeed' in df_eng.columns:
        for window in [168, 336]:
            df_eng[f'windspeed_mean_{window}h'] = df_eng['windspeed'].rolling(window=window, min_periods=1).mean()
    
    # Categorical encoding
    if 'icon' in df_eng.columns and df_eng['icon'].dtype == 'object':
        icon_dummies = pd.get_dummies(df_eng['icon'], prefix='icon', drop_first=False)
        df_eng = pd.concat([df_eng, icon_dummies], axis=1)
        df_eng = df_eng.drop('icon', axis=1)
    
    # Clean up
    non_numeric = df_eng.select_dtypes(include=['object']).columns.tolist()
    if non_numeric:
        df_eng = df_eng.drop(columns=non_numeric)
    
    # Handle NaN
    if len(df_eng) <= 720:
        df_eng = df_eng.ffill().bfill().fillna(0)
    else:
        df_eng = df_eng.dropna()
    
    return df_eng

print("\n" + "="*80)
print("FEATURE ENGINEERING")
print("="*80)

hourly_train_eng = engineer_features_for_d5(hourly_train)
hourly_val_eng = engineer_features_for_d5(hourly_val)
hourly_test_eng = engineer_features_for_d5(hourly_test)

print(f"Train: {hourly_train_eng.shape}")
print(f"Val:   {hourly_val_eng.shape}")
print(f"Test:  {hourly_test_eng.shape}")

# ============================================================================
# 5. AGGREGATE TO DAILY
# ============================================================================
def aggregate_hourly_to_daily(df_hourly):
    df = df_hourly.copy()
    df['date'] = df.index.date
    
    agg_functions = {}
    for col in df.columns:
        if col == 'date':
            continue
        elif 'sin' in col or 'cos' in col:
            agg_functions[col] = 'mean'
        elif any(stat in col for stat in ['mean', 'std', 'min', 'max', 'range', 'trend', 'momentum', 'vs', 'zscore']):
            agg_functions[col] = 'mean'
        elif col in (['is_weekend'] + [f'is_day_{i}' for i in range(7)] + 
                     ['day_of_week', 'month', 'day_of_year', 'week_of_year', 'day_of_month']):
            agg_functions[col] = 'first'
        else:
            agg_functions[col] = ['mean', 'min', 'max', 'std']
    
    daily_df = df.groupby('date').agg(agg_functions)
    
    if isinstance(daily_df.columns, pd.MultiIndex):
        daily_df.columns = ['_'.join(col).strip('_') for col in daily_df.columns.values]
    
    return daily_df

daily_train_features = aggregate_hourly_to_daily(hourly_train_eng)
daily_val_features = aggregate_hourly_to_daily(hourly_val_eng)
daily_test_features = aggregate_hourly_to_daily(hourly_test_eng)

print(f"\nDaily features - Train: {daily_train_features.shape}")

# ============================================================================
# 6. PREPARE D+5 TRAINING DATA
# ============================================================================
print("\n" + "="*80)
print("PREPARING D+5 DATA")
print("="*80)

# Shift by 5 days
train_features = daily_train_features.copy()
val_features = daily_val_features.copy()
test_features = daily_test_features.copy()

train_features.index = pd.to_datetime(train_features.index) + pd.Timedelta(days=5)
val_features.index = pd.to_datetime(val_features.index) + pd.Timedelta(days=5)
test_features.index = pd.to_datetime(test_features.index) + pd.Timedelta(days=5)

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

print(f"After merge - Train: {train_data.shape} | Val: {val_data.shape}")

train_data = train_data.set_index('date')
val_data = val_data.set_index('date')
test_data = test_data.set_index('date')

# Prepare X and y
TARGET_COL = 'daily_temp'
exclude_cols = [TARGET_COL] + [col for col in train_data.columns if 
    any(x in col for x in ['temp_diff', 'temp_std', 'temp_range', 'temp_mean']) and 'd' in col or
    col in ['temp_5d_ago', 'temp_7d_ago', 'temp_14d_ago', 'temp_21d_ago', 'temp_30d_ago',
            'day_of_week', 'day_of_month', 'month', 'is_weekend'] + [f'is_day_{i}' for i in range(7)]]

X_train = train_data.drop(columns=exclude_cols, errors='ignore')
y_train = train_data[TARGET_COL]
X_val = val_data.drop(columns=exclude_cols, errors='ignore')
y_val = val_data[TARGET_COL]
X_test = test_data.drop(columns=exclude_cols, errors='ignore')
y_test = test_data[TARGET_COL]

print(f"Features: {X_train.shape[1]}")

# Scale
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# 7. TRAIN D+5 MODEL
# ============================================================================
print("\n" + "="*80)
print("TRAINING D+5 MODEL")
print("="*80)

if TUNE_HYPERPARAMETERS and OPTUNA_AVAILABLE:
    print(f"Optuna optimization ({N_TRIALS} trials)...")
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 700),
            'learning_rate': trial.suggest_float('learning_rate', 0.003, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 5, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 5, 15),
            'subsample': trial.suggest_float('subsample', 0.5, 0.8),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),
            'gamma': trial.suggest_float('gamma', 0.3, 0.8),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.5, 2.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 2.0, 5.0),
            'objective': 'reg:squarederror',
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
        
        model = XGBRegressor(**params)
        model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)
        y_pred = model.predict(X_val_scaled)
        return mean_absolute_error(y_val, y_pred)
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    
    best_params = study.best_params
    best_params.update({'objective': 'reg:squarederror', 'random_state': 42, 'n_jobs': -1})
    
    print(f"Best MAE: {study.best_value:.4f} C")
    
    model = XGBRegressor(**best_params)
    model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)
else:
    model = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=500,
        learning_rate=0.03,
        max_depth=7,
        min_child_weight=7,
        subsample=0.65,
        colsample_bytree=0.65,
        gamma=0.5,
        reg_alpha=1.0,
        reg_lambda=3.0,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)

# ============================================================================
# 8. EVALUATE
# ============================================================================
print("\n" + "="*80)
print("D+5 PERFORMANCE")
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
    results.append({
        'Split': split_name,
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100,
        'R2': r2_score(y_true, y_pred),
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

model_path = os.path.join(SAVE_DIR, f"{MODEL_NAME}.json")
model.save_model(model_path)
print(f"+ XGBoost model saved: {model_path}")

scaler_path = os.path.join(SAVE_DIR, f"{MODEL_NAME}_scaler.pkl")
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"+ Scaler saved: {scaler_path}")

feature_names_path = os.path.join(SAVE_DIR, f"{MODEL_NAME}_features.json")
with open(feature_names_path, 'w') as f:
    json.dump(list(X_train.columns), f)
print(f"+ Feature names saved: {feature_names_path}")

metadata = {
    'model_name': MODEL_NAME,
    'forecast_horizon': 'D+5',
    'n_features': X_train.shape[1],
    'training_samples': len(X_train),
    'validation_mae': float(val_metrics['MAE']),
    'validation_rmse': float(val_metrics['RMSE']),
    'validation_r2': float(val_metrics['R2']),
    'validation_mape': float(val_metrics['MAPE']),
    'daily_calc_method': DAILY_CALC_METHOD
}
metadata_path = os.path.join(SAVE_DIR, f"{MODEL_NAME}_metadata.json")
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"+ Metadata saved: {metadata_path}")

# ============================================================================
# 10. EXPORT TO ONNX
# ============================================================================
if ONNX_AVAILABLE:
    print("\n" + "="*80)
    print("EXPORTING TO ONNX FORMAT")
    print("="*80)
    
    try:
        initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
        onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_type)
        onnx_path = os.path.join(SAVE_DIR, f"{MODEL_NAME}.onnx")
        onnx.save_model(onnx_model, onnx_path)
        print(f"+ ONNX model saved: {onnx_path}")
        
        onnx_model_check = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model_check)
        print("+ ONNX model verification passed")
        
        if ONNX_RUNTIME_AVAILABLE:
            sess = rt.InferenceSession(onnx_path)
            input_name = sess.get_inputs()[0].name
            output_name = sess.get_outputs()[0].name
            test_sample = X_val_scaled[:1].astype(np.float32)
            onnx_pred = sess.run([output_name], {input_name: test_sample})[0]
            xgb_pred = model.predict(test_sample)
            diff = abs(onnx_pred[0] - xgb_pred[0])
            print(f"+ ONNX inference test passed (diff: {diff:.6f})")
    
    except Exception as e:
        print(f"x ONNX export failed: {str(e)}")
else:
    print("\nx ONNX export skipped (libraries not available)")

print("\n" + "="*80)
print("MODEL TRAINING COMPLETE - D+5")
print("="*80)
print(f"Validation MAE: {val_metrics['MAE']:.4f} C")
print(f"Validation R2: {val_metrics['R2']:.4f}")
print(f"Model saved in: {SAVE_DIR}")
print("="*80)