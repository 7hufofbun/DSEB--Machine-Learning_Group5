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

#--------------HOURLY-------------------#
from data_preprocessing import (
    get_data,
    basic_transform,
    clean_data,
    extract_datetime_features,
    create_lag_features,
    create_rolling_features,
    create_interactive_features,
)

hourly_path = r"data\weather_hcm_hourly.csv"
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
# drop_unnecessary_columns is part of feature_engineering; avoid using it
# per user's request and rely on data_preprocessing transformations only
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
# Consolidate frame to reduce fragmentation after many column inserts
hourly_data = hourly_data.copy()

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

# Use create_interactive_features from data_preprocessing
hourly_data = create_interactive_features(hourly_data)

# Targets: next 1 hour (single-value target)
def create_targets(df, horizon=1, target_col='temp'):
    targets = pd.DataFrame(index=df.index)
    for h in range(1, horizon + 1):
        targets[f'{target_col}_target_{h}'] = df[target_col].shift(-h)
    return targets

Y_hourly = create_targets(hourly_data, horizon=1, target_col='temp')
drop_cols_hourly = ['temp', 'feelslike']  # Drop current temp-related
X_hourly = hourly_data.drop(columns=drop_cols_hourly, errors='ignore')

# Drop NaNs from shifts
combined_hourly = pd.concat([X_hourly, Y_hourly], axis=1).dropna()
X_hourly = combined_hourly[X_hourly.columns]
Y_hourly = combined_hourly[Y_hourly.columns]

# Chronological split for tabular hourly features (no sliding windows)
n_h = len(X_hourly)
train_size_h = int(0.7 * n_h)
val_size_h = int(0.15 * n_h)

X_train_h = X_hourly.iloc[:train_size_h].values
Y_train_h = Y_hourly.iloc[:train_size_h].values
X_val_h = X_hourly.iloc[train_size_h:train_size_h + val_size_h].values
Y_val_h = Y_hourly.iloc[train_size_h:train_size_h + val_size_h].values
X_test_h = X_hourly.iloc[train_size_h + val_size_h:].values
Y_test_h = Y_hourly.iloc[train_size_h + val_size_h:].values

print(f"Hourly tabular splits — Train: {X_train_h.shape}, Val: {X_val_h.shape}, Test: {X_test_h.shape}")

# Scaling: use RobustScaler for X and StandardScaler for Y
scaler_x_h = RobustScaler().fit(X_train_h)
X_train_h = scaler_x_h.transform(X_train_h)
X_val_h = scaler_x_h.transform(X_val_h) if len(X_val_h) > 0 else X_val_h
X_test_h = scaler_x_h.transform(X_test_h) if len(X_test_h) > 0 else X_test_h

scaler_y_h = StandardScaler().fit(Y_train_h)
Y_train_h = scaler_y_h.transform(Y_train_h)
Y_val_h = scaler_y_h.transform(Y_val_h) if len(Y_val_h) > 0 else Y_val_h
Y_test_h = scaler_y_h.transform(Y_test_h) if len(Y_test_h) > 0 else Y_test_h

# PyTorch Dataset for 2D tabular data (flattened inputs)
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        # X: (N, features), y: (N, 1)
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

def compute_metrics(y_true, y_pred):
    # y_true/pred: arrays shape (n_samples, n_targets)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    try:
        mape = mean_absolute_percentage_error(y_true, y_pred)
    except Exception:
        mape = np.nan
    try:
        r2 = r2_score(y_true, y_pred)
    except Exception:
        r2 = np.nan
    return {'rmse': rmse, 'mape': mape, 'r2': r2}


def save_model(obj, path):
    import joblib
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)


if __name__ == '__main__':
    print('\n--- Training tabular hourly models (no sliding windows) ---')

    # Train RF pipeline (scaler + multioutput RF)
    print('Training RandomForest pipeline...')
    # Temporarily reduce complexity for quick runs
    pipeline_rf.named_steps['model'] = MultiOutputRegressor(RandomForestRegressor(n_estimators=50, random_state=42))
    pipeline_rf.fit(X_train_h, Y_train_h)
    preds_rf = pipeline_rf.predict(X_val_h)
    metrics_rf = compute_metrics(Y_val_h, preds_rf)
    print('RandomForest validation metrics:', metrics_rf)
    save_model(pipeline_rf, 'outputs/models/pipeline_rf.joblib')

    # Train XGBoost pipeline
    print('Training XGBoost pipeline...')
    pipeline_xgb.named_steps['model'] = MultiOutputRegressor(XGBRegressor(n_estimators=50, random_state=42, verbosity=0))
    pipeline_xgb.fit(X_train_h, Y_train_h)
    preds_xgb = pipeline_xgb.predict(X_val_h)
    metrics_xgb = compute_metrics(Y_val_h, preds_xgb)
    print('XGBoost validation metrics:', metrics_xgb)
    save_model(pipeline_xgb, 'outputs/models/pipeline_xgb.joblib')

    # Optionally train LSTM: requires sliding windows. Disabled by default.
    train_lstm_flag = False
    if train_lstm_flag:
        print('Preparing sliding windows for LSTM...')
        # Create minimal sliding windows for LSTM (seq_len timesteps -> predict next hour)
        seq_len = 12
        X_arr = X_hourly.values
        Y_arr = Y_hourly.values
        T = X_arr.shape[0]
        N = T - seq_len
        if N <= 0:
            print('Not enough hourly data for LSTM sliding windows. Skipping LSTM.')
        else:
            Xs = np.stack([X_arr[i:i+seq_len] for i in range(N)], axis=0)
            Ys = np.stack([Y_arr[i+seq_len] for i in range(N)], axis=0)
            # Chronological split
            nN = len(Xs)
            tsize = int(0.7 * nN)
            vsize = int(0.15 * nN)
            Xtr, Xv, Xt = Xs[:tsize], Xs[tsize:tsize+vsize], Xs[tsize+vsize:]
            Ytr, Yv, Yt = Ys[:tsize], Ys[tsize:tsize+vsize], Ys[tsize+vsize:]
            # scale X per-feature
            ns, sl, nf = Xtr.shape
            med = np.median(Xtr.reshape(-1, nf), axis=0)
            iqr = np.percentile(Xtr.reshape(-1, nf), 75, axis=0) - np.percentile(Xtr.reshape(-1, nf), 25, axis=0)
            iqr[iqr==0] = 1.0
            def scale_seq(arr):
                a2 = (arr.reshape(-1, nf) - med) / iqr
                return a2.reshape(arr.shape)
            Xtr = scale_seq(Xtr); Xv = scale_seq(Xv); Xt = scale_seq(Xt)
            # convert to tensors and train small LSTM
            train_ds = TimeSeriesDataset(Xtr, Ytr)
            val_ds = TimeSeriesDataset(Xv, Yv)
            tr_loader = DataLoader(train_ds, batch_size=64, shuffle=False)
            vl_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
            model = LSTMModel(input_size=nf, hidden_size=64, num_layers=1, output_size=Ytr.shape[1])
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.MSELoss()
            print('Training LSTM for a few epochs...')
            train_lstm(model, tr_loader, vl_loader, criterion, optimizer, n_epochs=5)
            save_model(model, 'outputs/models/lstm_hourly.pth')

    print('\nAll done. Models saved to outputs/models/')