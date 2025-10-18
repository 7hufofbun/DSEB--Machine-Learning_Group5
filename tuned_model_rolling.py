import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import optuna
# For ONNX: Install via pip install skl2onnx onnxruntime
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt

def objective(trial, X_train_val, y_train_val):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
        'random_state': 42
    }
    
    model = RandomForestRegressor(**params)
    
    # Rolling validation
    tscv = TimeSeriesSplit(n_splits=5)  # Adjust for data size
    rmses = []
    
    for train_idx, val_idx in tscv.split(X_train_val):
        X_tr, X_vl = X_train_val.iloc[train_idx], X_train_val.iloc[val_idx]
        y_tr, y_vl = y_train_val.iloc[train_idx], y_train_val.iloc[val_idx]
        if len(X_vl) == 0 or len(X_tr) == 0:
            continue
        model.fit(X_tr, y_tr)
        y_pr = model.predict(X_vl)
        rmse = np.sqrt(mean_squared_error(y_vl, y_pr))
        rmses.append(rmse)
    
    return np.mean(rmses) if rmses else float('inf')

def tune_and_build_model(x_path='outputs/X_features.csv', y_path='outputs/Y_target.csv', onnx_path='model.onnx'):
    X = pd.read_csv(x_path, parse_dates=['datetime'])
    y = pd.read_csv(y_path, parse_dates=['datetime'])
    data = pd.merge(X, y, on='datetime', how='inner')
    data = data.sort_values('datetime').set_index('datetime')
    data['temp_next_day'] = data['temp'].shift(-1)
    data = data.dropna()
    features = data.columns.drop(['temp', 'temp_next_day'])
    X = data[features]
    y_shifted = data['temp_next_day']
    split_idx = int(0.8 * len(X))
    X_train_val, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train_val, y_test = y_shifted.iloc[:split_idx], y_shifted.iloc[split_idx:]
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train_val, y_train_val), n_trials=50)
    best_params = study.best_params
    print("Best params:", best_params)
    model = RandomForestRegressor(**best_params, random_state=42)
    model.fit(X_train_val, y_train_val)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"RMSE: {rmse:.2f}, MAPE: {mape:.2%}, R2: {r2:.2f}")
    
    # Export to ONNX
    initial_type = [('input', FloatTensorType([None, len(features)]))]
    onx = convert_sklearn(model, initial_types=initial_type)
    with open(onnx_path, "wb") as f:
        f.write(onx.SerializeToString())
    print(f"Model saved to {onnx_path}")
    
    # Validate ONNX inference
    sess = rt.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    onnx_pred = sess.run([label_name], {input_name: X_test.values.astype(np.float32)})[0]
    print("ONNX RMSE:", np.sqrt(mean_squared_error(y_test, onnx_pred)))
    
    # Predict next day with ONNX
    last_row = data.iloc[[-1]][features].values.astype(np.float32)
    next_day_pred = sess.run([label_name], {input_name: last_row})[0][0]
    print("Predicted average temperature for next day after last:", round(next_day_pred, 1))
    
    return model, features, data

if __name__ == "__main__":
    tune_and_build_model()