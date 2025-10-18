import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

def build_basic_model(x_path='outputs/X_features.csv', y_path='outputs/Y_target.csv'):
    # Load preprocessed files
    X = pd.read_csv(x_path, parse_dates=['datetime'])
    y = pd.read_csv(y_path, parse_dates=['datetime'])

    # Merge on datetime
    data = pd.merge(X, y, on='datetime', how='inner')
    data = data.sort_values('datetime').set_index('datetime')

    # Shift for 1-day forecast
    data['temp_next_day'] = data['temp'].shift(-1)
    data = data.dropna()

    # Features (all except temp)
    features = data.columns.drop(['temp', 'temp_next_day'])
    X = data[features]
    y_shifted = data['temp_next_day']

    # Time-based split (80/20)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y_shifted.iloc[:split_idx], y_shifted.iloc[split_idx:]

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"RMSE: {rmse:.2f}, MAPE: {mape:.2%}, R2: {r2:.2f}")

    return model, features, data

if __name__ == "__main__":
    model, features, data = build_basic_model()
    
    # Predict one day ahead from last row
    last_row = data.iloc[[-1]][features]
    next_day_pred = model.predict(last_row)[0]
    print("Predicted average temperature for next day after last (2021-06-14):", round(next_day_pred, 1))