# model_training.py
# Fixed ValueError: Added create_multi_targets to make Y 2D (next 5 days).
# Fixed TypeError: Added iteration=0 to ClearML report_scalar.
# Updated feature selection to use first target column for importances.
# Assumes updated data_preprocessing.py with concat fixes (no more warnings).

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from optuna import create_study
from clearml import Task
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import os

# From data_preprocessing.py (assume fixed with concat)
from data_preprocessing import get_data, basic_cleaning, feature_engineer_X, Preprocessing, PipelineWithY, SafeAligner, split_data

# Multi-target creation
def create_multi_targets(y, horizons=5):
    targets = pd.DataFrame(index=y.index)
    for h in range(1, horizons + 1):
        targets[f'temp_target_{h}'] = y.shift(-h)
    return targets.dropna()

# Feature selection
def select_features_with_rf_xgb(X, y, importance_threshold=0.0065):
    valid_idx = y.notna().all(axis=1)
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y.iloc[:, 0])  # Use first target for selection
    rf_imp = pd.Series(rf.feature_importances_, index=X.columns)

    xgb = XGBRegressor(n_estimators=100, random_state=42)
    xgb.fit(X, y.iloc[:, 0])
    xgb_imp = pd.Series(xgb.feature_importances_, index=X.columns)

    combined_imp = (rf_imp + xgb_imp) / 2
    combined_imp = combined_imp.sort_values(ascending=False)
    selected_features = combined_imp[combined_imp >= importance_threshold].index
    X_selected = X[selected_features]
    print(f"Selected {len(selected_features)} features.")
    return X_selected, combined_imp

# Initialize ClearML
task = Task.init(project_name='HCM Temperature Forecasting', task_name='Model Training')

# Load and process
path = "https://raw.githubusercontent.com/7hufofbun/DSEB--Machine-Learning_Group5/refs/heads/main/data/weather_hcm_daily.csv"
data = get_data(path)
data = basic_cleaning(data)

X_train, y_train, X_val, y_val, X_test, y_test = split_data(data, train_ratio=0.7, val_ratio=0.15)

X_train, y_train = feature_engineer_X(X_train, y_train)
X_val, y_val = feature_engineer_X(X_val, y_val)
X_test, y_test = feature_engineer_X(X_test, y_test)

pipe = PipelineWithY([
    ('preprocess', Preprocessing()),
    ('align', SafeAligner())
])

X_train, y_train = pipe.fit_transform(X_train, y_train)
X_val, y_val = pipe.transform(X_val, y_val)
X_test, y_test = pipe.transform(X_test, y_test)

# Multi-output Y
Y_train = create_multi_targets(y_train)
Y_val = create_multi_targets(y_val)
Y_test = create_multi_targets(y_test)

# Align X
X_train = X_train.loc[Y_train.index]
X_val = X_val.loc[Y_val.index]
X_test = X_test.loc[Y_test.index]

# Selection
X_train_selected, importances = select_features_with_rf_xgb(X_train, Y_train)
selected_cols = X_train_selected.columns
X_val = X_val[selected_cols]
X_test = X_test[selected_cols]

print(f"Train: {X_train_selected.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Evaluate
def evaluate_and_log(model, X, y, model_name, set_type='Test'):
    preds = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, preds))
    mape = mean_absolute_percentage_error(y, preds)
    r2 = r2_score(y, preds)
    print(f"{model_name} {set_type} - RMSE: {rmse:.4f}, MAPE: {mape:.4f}, R2: {r2:.4f}")
    task.logger.report_scalar(f"{set_type} RMSE", model_name, rmse, iteration=0)
    task.logger.report_scalar(f"{set_type} MAPE", model_name, mape, iteration=0)
    task.logger.report_scalar(f"{set_type} R2", model_name, r2, iteration=0)
    return rmse, mape, r2

models = {}
metrics = {}

# RF
def objective_rf(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10)
    }
    model = MultiOutputRegressor(RandomForestRegressor(random_state=42, **params))
    model.fit(X_train_selected, Y_train)
    preds_val = model.predict(X_val)
    return np.sqrt(mean_squared_error(Y_val, preds_val))

study_rf = create_study(direction='minimize')
study_rf.optimize(objective_rf, n_trials=20)
best_params_rf = study_rf.best_params
task.connect(best_params_rf, name='RF Params')

model_rf = MultiOutputRegressor(RandomForestRegressor(random_state=42, **best_params_rf))
model_rf.fit(X_train_selected, Y_train)
metrics['RF'] = evaluate_and_log(model_rf, X_test, Y_test, "RandomForest")
models['RF'] = model_rf

# XGB
def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
    }
    model = MultiOutputRegressor(XGBRegressor(random_state=42, verbosity=0, **params))
    model.fit(X_train_selected, Y_train)
    preds_val = model.predict(X_val)
    return np.sqrt(mean_squared_error(Y_val, preds_val))

study_xgb = create_study(direction='minimize')
study_xgb.optimize(objective_xgb, n_trials=20)
best_params_xgb = study_xgb.best_params
task.connect(best_params_xgb, name='XGB Params')

model_xgb = MultiOutputRegressor(XGBRegressor(random_state=42, verbosity=0, **best_params_xgb))
model_xgb.fit(X_train_selected, Y_train)
metrics['XGB'] = evaluate_and_log(model_xgb, X_test, Y_test, "XGBoost")
models['XGB'] = model_xgb

# LGBM
def objective_lgbm(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
    }
    model = MultiOutputRegressor(LGBMRegressor(random_state=42, verbosity=-1, **params))
    model.fit(X_train_selected, Y_train)
    preds_val = model.predict(X_val)
    return np.sqrt(mean_squared_error(Y_val, preds_val))

study_lgbm = create_study(direction='minimize')
study_lgbm.optimize(objective_lgbm, n_trials=20)
best_params_lgbm = study_lgbm.best_params
task.connect(best_params_lgbm, name='LGBM Params')

model_lgbm = MultiOutputRegressor(LGBMRegressor(random_state=42, verbosity=-1, **best_params_lgbm))
model_lgbm.fit(X_train_selected, Y_train)
metrics['LGBM'] = evaluate_and_log(model_lgbm, X_test, Y_test, "LightGBM")
models['LGBM'] = model_lgbm

# Best model
best_model_name = min(metrics, key=lambda k: metrics[k][0])
best_model = models[best_model_name]
print(f"Best model: {best_model_name}")

# ONNX export
initial_type = [('float_input', FloatTensorType([None, X_train_selected.shape[1]]))]

if 'XGB' in best_model_name:
    from onnxmltools import convert_xgboost
    onnx_model = convert_xgboost(best_model.estimators_[0].get_booster(), initial_types=initial_type)
elif 'LGBM' in best_model_name:
    from onnxmltools import convert_lightgbm
    onnx_model = convert_lightgbm(best_model.estimators_[0], initial_types=initial_type)
else:  # RF
    onnx_model = convert_sklearn(best_model.estimators_[0], initial_types=initial_type)

with open(f"best_model_{best_model_name}_estimator0.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())