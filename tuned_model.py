import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import optuna

def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
        'random_state': 42
    }
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    return rmse

def tune_and_build_model(x_path='outputs/X_features.csv', y_path='outputs/Y_target.csv'):
    # Load preprocessed files
    X = pd.read_csv(x_path, parse_dates=['datetime'])
    y = pd.read_csv(y_path, parse_dates=['datetime'])

    # Merge and sort
    data = pd.merge(X, y, on='datetime', how='inner')
    data = data.sort_values('datetime').set_index('datetime')

    # Shift for 1-day forecast
    data['temp_next_day'] = data['temp'].shift(-1)
    data = data.dropna()

    # Features (all except temp)
    features = data.columns.drop(['temp', 'temp_next_day'])
    X = data[features]
    y_shifted = data['temp_next_day']

    # Time-based split: 60/20/20
    n = len(X)
    train_idx = int(0.6 * n)
    val_idx = int(0.8 * n)
    X_train, X_val, X_test = X.iloc[:train_idx], X.iloc[train_idx:val_idx], X.iloc[val_idx:]
    y_train, y_val, y_test = y_shifted.iloc[:train_idx], y_shifted.iloc[train_idx:val_idx], y_shifted.iloc[val_idx:]

    # Optuna tuning
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=50)
    best_params = study.best_params
    print("Best params:", best_params)
    "Quên không save model nên t để output luôn mng đỡ phải chạy lại (khoảng 16 phút cho tổng 50 trials)"
    "Best params: {'n_estimators': 300, 'max_depth': 5, 'min_samples_split': 4, 'min_samples_leaf': 3} - Trial 42"

    # Train final model on train+val with best params
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])
    model = RandomForestRegressor(**best_params, random_state=42)
    model.fit(X_train_val, y_train_val)

    # Evaluate on test
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"RMSE: {rmse:.2f}, MAPE: {mape:.2%}, R2: {r2:.2f}")

    return model, features, data

if __name__ == "__main__":
    model, features, data = tune_and_build_model()
    
    # Predict one day ahead from last row
    last_row = data.iloc[[-1]][features]
    next_day_pred = model.predict(last_row)[0]
    print("Predicted average temperature for next day after last (2021-06-14):", round(next_day_pred, 1))