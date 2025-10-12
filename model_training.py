# model_training.py
# This script handles Step 5: Model Training and Hyper-parameter Tuning for the Ho Chi Minh City temperature forecasting project.
# We use the features and targets generated from feature_engineering.py (X_features.csv and Y_target.csv).
# Data splitting: Chronological split to avoid data leakage (70% train, 15% val, 15% test).
# Models: RandomForestRegressor, XGBoost (both via MultiOutputRegressor for multi-step forecasting), and LSTM for hourly data.
# Hyper-parameter tuning: Optuna.
# Metrics: RMSE, MAPE, R2.
# Monitoring: ClearML (initialize task, log metrics).
# Pipeline: Used for scaling (though tree models don't strictly need it, good practice).
# For LSTM on hourly: Similar processing, but with hourly data.

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from optuna import create_study
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from clearml import Task  # For monitoring
import os

# Load X and Y
# Read features and targets; explicitly parse the index column as datetime to avoid
# pandas falling back to slower dateutil parsing per-element.
X = pd.read_csv(r'outputs\X_features.csv', index_col=0, parse_dates=[0])
Y = pd.read_csv(r'outputs\Y_target.csv', index_col=0, parse_dates=[0])

# Ensure chronological order
X = X.sort_index()
Y = Y.sort_index()

# Split: train 70%, val 15%, test 15% - chronological to prevent leakage
n = len(X)
train_size = int(0.7 * n)
val_size = int(0.15 * n)

X_train = X.iloc[:train_size]
Y_train = Y.iloc[:train_size]
X_val = X.iloc[train_size:train_size + val_size]
Y_val = Y.iloc[train_size:train_size + val_size]
X_test = X.iloc[train_size + val_size:]
Y_test = Y.iloc[train_size + val_size:]

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Pipeline for RF
pipeline_rf = Pipeline([
    ('scaler', RobustScaler()),
    ('model', MultiOutputRegressor(RandomForestRegressor(random_state=42)))
])

# Pipeline for XGBoost
pipeline_xgb = Pipeline([
    ('scaler', RobustScaler()),
    ('model', MultiOutputRegressor(XGBRegressor(random_state=42)))
])


def train_rf_model(X_train, Y_train, X_val, Y_val, **rf_kwargs):
    """Train a multi-output RandomForest and evaluate on validation set.
    X_train/X_val: 2D arrays (n_samples, n_features)
    Y_train/Y_val: 2D arrays (n_samples, n_targets)
    Returns: fitted model, val_rmse
    """
    model = MultiOutputRegressor(RandomForestRegressor(random_state=42, **rf_kwargs))
    model.fit(X_train, Y_train)
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(Y_val, preds))
    print(f"RF Validation RMSE: {rmse:.4f}")
    return model, rmse


def train_xgb_model(X_train, Y_train, X_val, Y_val, **xgb_kwargs):
    """Train a multi-output XGBoost and evaluate on validation set.
    Returns: fitted model, val_rmse
    """
    base = XGBRegressor(random_state=42, verbosity=0, n_jobs=-1, **xgb_kwargs)
    model = MultiOutputRegressor(base)
    model.fit(X_train, Y_train)
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(Y_val, preds))
    print(f"XGB Validation RMSE: {rmse:.4f}")
    return model, rmse

from data_preprocessing import (get_data, basic_transform, clean_data) 
from feature_engineering import (drop_unnecessary_columns, extract_datetime_features,
                                 create_lag_features, create_rolling_features,
                                 create_interactive_features, create_targets) 
hourly_path = r"D:\DSEB\SEM5\ML\data\weather_hcm_hourly.csv"
hourly_data = get_data(hourly_path)

# Lightweight local cleaning (avoid expensive time-based interpolation)
# Use this instead of calling `clean_data()` to prevent long-running
# pandas `interpolate(method='time')` on large datasets. This keeps the
# cleaning conservative but fast.
hourly_data = hourly_data.copy()
# Parse datetime if present
if 'datetime' in hourly_data.columns:
    hourly_data['datetime'] = pd.to_datetime(hourly_data['datetime'], errors='coerce')

# Drop known non-essential columns if present
hourly_data.drop(columns=['description', 'sunrise', 'sunset'], errors='ignore', inplace=True)

# Drop columns with a single unique value (uninformative)
try:
    uni_value = hourly_data.nunique()
    cols = uni_value[uni_value == 1].index.tolist()
    if cols:
        hourly_data.drop(columns=cols, errors='ignore', inplace=True)
except Exception:
    pass

# Interpolate numeric columns (fast, non-time method). If datetime index
# exists we still set it for alignment but avoid method='time'.
try:
    num_cols = hourly_data.select_dtypes(include=['float64', 'int64']).columns
    if 'datetime' in hourly_data.columns:
        hourly_data = hourly_data.set_index('datetime')
        if len(num_cols) > 0:
            hourly_data[num_cols] = hourly_data[num_cols].interpolate(limit_direction='both')
        hourly_data = hourly_data.reset_index()
    else:
        if len(num_cols) > 0:
            hourly_data[num_cols] = hourly_data[num_cols].interpolate(limit_direction='both')
except Exception:
    pass

# Forward/backward fill categorical/object columns
try:
    cat_cols = hourly_data.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        hourly_data[cat_cols] = hourly_data[cat_cols].ffill().bfill()
except Exception:
    pass

# Drop duplicates conservatively
try:
    hourly_data = hourly_data.reset_index(drop=True)
    hourly_data.drop_duplicates(inplace=True)
except Exception:
    pass

# Continue with feature transforms
hourly_data = basic_transform(hourly_data)
hourly_data = drop_unnecessary_columns(hourly_data)
hourly_data = extract_datetime_features(hourly_data)
hourly_data.set_index('datetime', inplace=True)
hourly_data.sort_index(inplace=True)

# Key columns for hourly (adjusted, e.g., no tempmax/min)
key_columns_hourly = ['temp', 'feelslike', 'humidity', 'dew', 'solarradiation', 'cloudcover', 'windspeed']
encoded_cols_hourly = [col for col in hourly_data.columns if col.startswith(('icon_', 'conditions_'))]
key_columns_hourly.extend(encoded_cols_hourly)

# Lags and windows for hourly: short-term (1-3h), medium (6-12h), daily (24h)
def create_lag_features_fast(df, columns, lags=[1,2,3,6,12,24]):
    # inplace lag creation to avoid full DataFrame copies
    for col in columns:
        if col not in df.columns:
            continue
        for lag in lags:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    return df

def create_rolling_features_fast(df, columns, windows=[3,6,12,24]):
    # inplace rolling features; mean and std shifted by 1 to avoid leakage
    for col in columns:
        if col not in df.columns:
            continue
        s = df[col]
        for window in windows:
            df[f'{col}_roll_mean_{window}'] = s.rolling(window=window).mean().shift(1)
            df[f'{col}_roll_std_{window}'] = s.rolling(window=window).std().shift(1)
    return df

# Use faster in-place builders to avoid expensive copies in feature_engineering
hourly_data = create_lag_features_fast(hourly_data, key_columns_hourly, lags=[1, 2, 3, 6, 12, 24])
hourly_data = create_rolling_features_fast(hourly_data, key_columns_hourly, windows=[3, 6, 12, 24])

# Ensure expected columns used by feature functions exist. The shared
# `create_interactive_features` expects `windspeedmean` and `dew`. Some
# hourly datasets use `windspeed` instead of `windspeedmean` or may be
# missing Dew; create safe fallbacks to avoid KeyError while keeping
# semantics reasonable.
if 'windspeedmean' not in hourly_data.columns:
    if 'windspeed' in hourly_data.columns:
        hourly_data['windspeedmean'] = hourly_data['windspeed']
    else:
        # create the column with NaNs so downstream code works without KeyError
        hourly_data['windspeedmean'] = np.nan

if 'dew' not in hourly_data.columns:
    # try common alternative names, otherwise fill with NaN
    if 'dewpoint' in hourly_data.columns:
        hourly_data['dew'] = hourly_data['dewpoint']
    else:
        hourly_data['dew'] = np.nan

hourly_data = create_interactive_features(hourly_data)

# Targets: next 1 hour (single-value target per sequence)
Y_hourly = create_targets(hourly_data, horizons=1)
drop_cols_hourly = ['temp', 'feelslike']  # Drop current temp-related
X_hourly = hourly_data.drop(columns=drop_cols_hourly, errors='ignore')

# Drop NaNs from shifts
combined_hourly = pd.concat([X_hourly, Y_hourly], axis=1).dropna()
X_hourly = combined_hourly[X_hourly.columns]
Y_hourly = combined_hourly[Y_hourly.columns]


def create_sliding_windows(X_df, Y_df, seq_len):
        """
        Create sliding windows from X and align Y to the window end.
        X_df: DataFrame of shape (T, n_features)
        Y_df: DataFrame of shape (T, horizon)
        Returns:
            X_seq: numpy array (N, seq_len, n_features)
            Y_seq: numpy array (N, horizon)
        where N = T - seq_len + 1
        """
        X_arr = X_df.values
        Y_arr = Y_df.values
        T = X_arr.shape[0]
        if T < seq_len:
                return np.empty((0, seq_len, X_arr.shape[1])), np.empty((0, Y_arr.shape[1]))
        N = T - seq_len + 1
        X_seq = np.stack([X_arr[i:i+seq_len] for i in range(N)], axis=0)
        # Align target to the last time step of each window
        Y_seq = np.stack([Y_arr[i+seq_len-1] for i in range(N)], axis=0)
        return X_seq, Y_seq


# --- Create sliding windows (sequence length) for LSTM input ---
seq_len = 12  # change this to the desired sequence length
X_seq, Y_seq = create_sliding_windows(X_hourly, Y_hourly, seq_len=seq_len)
print(f"Sliding windows created: X_seq shape={X_seq.shape}, Y_seq shape={Y_seq.shape}, seq_len={seq_len}")
if X_seq.size == 0:
    raise SystemExit("No sliding windows created: reduce seq_len or verify hourly data length after preprocessing.")

# Chronological split on sequences
n_h = len(X_seq)
train_size_h = int(0.7 * n_h)
val_size_h = int(0.15 * n_h)

X_train_h = X_seq[:train_size_h]
Y_train_h = Y_seq[:train_size_h]
X_val_h = X_seq[train_size_h:train_size_h + val_size_h]
Y_val_h = Y_seq[train_size_h:train_size_h + val_size_h]
X_test_h = X_seq[train_size_h + val_size_h:]
Y_test_h = Y_seq[train_size_h + val_size_h:]

# Scaling in pipeline-like manner
# For sequences we fit scaler on the reshaped 2D array (samples*time, features)
nsamples, seq_len_actual, nfeatures = X_train_h.shape

def manual_robust_scale_sequences(X_train, X_val=None, X_test=None, max_samples=50000, random_state=42):
    """Compute per-feature median and IQR on a sample and scale arrays accordingly.
    Returns scaled arrays and (median, iqr) to inverse-transform if needed.
    """
    np.random.seed(random_state)
    train_2d = X_train.reshape(-1, nfeatures)
    n_total = train_2d.shape[0]
    sample_n = min(n_total, max_samples)
    if sample_n < n_total:
        idx = np.random.choice(n_total, sample_n, replace=False)
        sample = train_2d[idx]
    else:
        sample = train_2d

    med = np.median(sample, axis=0)
    q75 = np.percentile(sample, 75, axis=0)
    q25 = np.percentile(sample, 25, axis=0)
    iqr = q75 - q25
    # avoid division by zero
    iqr[iqr == 0] = 1.0

    def scale_array(arr):
        if arr.size == 0:
            return arr
        arr2 = arr.reshape(-1, nfeatures)
        arr2 = (arr2 - med) / iqr
        return arr2.reshape(arr.shape)

    X_train_s = scale_array(X_train)
    X_val_s = scale_array(X_val) if X_val is not None and X_val.size else X_val
    X_test_s = scale_array(X_test) if X_test is not None and X_test.size else X_test
    return X_train_s, X_val_s, X_test_s, med, iqr


# Scale sequences using manual robust scaler (sample-based)
X_train_h, X_val_h, X_test_h, x_med, x_iqr = manual_robust_scale_sequences(X_train_h, X_val_h, X_test_h)

# Scale Y per-output (no sequence dimension)
scaler_y_h = StandardScaler().fit(Y_train_h)
Y_train_h = scaler_y_h.transform(Y_train_h)
if len(Y_val_h) > 0:
    Y_val_h = scaler_y_h.transform(Y_val_h)
if len(Y_test_h) > 0:
    Y_test_h = scaler_y_h.transform(Y_test_h)

# PyTorch Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        # X: (N, seq_len, features), y: (N, horizon)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TimeSeriesDataset(X_train_h, Y_train_h)
val_dataset = TimeSeriesDataset(X_val_h, Y_val_h)
test_dataset = TimeSeriesDataset(X_test_h, Y_test_h)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)  # No shuffle for time series
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Pipeline for LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x expected shape: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
def train_lstm(model, train_loader, val_loader, criterion, optimizer, n_epochs=50):
    for epoch in range(n_epochs):
        model.train()
        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                outputs = model(X_batch)
                val_loss += criterion(outputs, Y_batch).item()
        print(f'Epoch {epoch+1}, Val Loss: {val_loss/len(val_loader)}')
    return model