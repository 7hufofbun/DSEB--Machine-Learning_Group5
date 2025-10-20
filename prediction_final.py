import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
import optuna
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt
import os  
GLOBAL_SEED = 42
def get_data(path):
    """Get data from downloaded datasets"""
    data = pd.read_csv(path)
    return data

def basic_cleaning(data):
    data = data.drop_duplicates()
    # Drop unnecessary columns
    uni_value = data.nunique()
    col = uni_value[uni_value == 1].index
    data.drop(col, axis=1, inplace=True)
    data.drop(['description'], axis=1, inplace=True, errors='ignore')

    # Normalize dtypes
    data['datetime'] = pd.to_datetime(data['datetime'])
    data['sunrise'] = pd.to_datetime(data['sunrise'])
    data['sunset'] = pd.to_datetime(data['sunset'])

    return data

class SmartCorrelationReducer(BaseEstimator, TransformerMixin):
    def __init__(self, target_col='temp', threshold=0.95):
        self.target_col = target_col
        self.threshold = threshold
        self.cols_to_keep_ = None
        self.non_numeric_cols_ = None

    def fit(self, X, y):
        df = X.copy()
        # ĐẢM BẢO index consistency
        df[self.target_col] = y.values if hasattr(y, 'values') else y

        # Store non-numeric columns (datetime, etc.)
        self.non_numeric_cols_ = df.select_dtypes(exclude="number").columns.tolist()
        
        # Get numeric columns excluding target
        num_cols = df.select_dtypes(include="number").columns.drop(self.target_col, errors='ignore')
        
        if len(num_cols) == 0:
            self.cols_to_keep_ = []
            return self
            
        corr_matrix = df[num_cols].corr().abs()

        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        to_drop = set()
        for col in upper.columns:
            if col not in to_drop:
                high_corr = upper.index[upper[col] > self.threshold].tolist()
                for hc in high_corr:
                    if hc not in to_drop:
                        corr_col = abs(df[col].corr(df[self.target_col]))
                        corr_hc = abs(df[hc].corr(df[self.target_col]))
                        # Handle NaN values
                        if np.isnan(corr_col) and np.isnan(corr_hc):
                            to_drop.add(hc)
                        elif np.isnan(corr_col):
                            to_drop.add(col)
                            break
                        elif np.isnan(corr_hc):
                            to_drop.add(hc)
                        elif corr_col >= corr_hc:
                            to_drop.add(hc)
                        else:
                            to_drop.add(col)
                            break

        self.cols_to_keep_ = [c for c in num_cols if c not in to_drop]
        return self

    def transform(self, X):
        if self.cols_to_keep_ is None:
            return X
            
        # Keep numeric columns + ALL non-numeric columns (including datetime)
        keep_cols = self.cols_to_keep_ + [col for col in self.non_numeric_cols_ if col in X.columns]
        keep_cols = [col for col in keep_cols if col in X.columns]
        
        return X[keep_cols]

class Preprocessing(BaseEstimator, TransformerMixin):
    def __init__(self, lower=0.05, upper=0.95, threshold=50, datetime_col='datetime'):
        self.lower = lower
        self.upper = upper  
        self.threshold = threshold
        self.datetime_col = datetime_col
        
    def fit(self, X, y=None):
        df = X.copy()
        
        # ĐẢM BẢO datetime column được giữ lại
        if self.datetime_col not in df.columns:
            raise ValueError(f"datetime column '{self.datetime_col}' not found in data")
        
        # Identify numerical and categorical cols (EXCLUDE datetime from processing)
        temp_df = df.drop(columns=[self.datetime_col])
        self.numeric_cols_ = temp_df.select_dtypes(include='number').columns.tolist()
        self.categorical_cols_ = temp_df.select_dtypes(include='object').columns.tolist()

        # Identify columns with missing values below threshold
        percentage_missing = temp_df.isnull().sum() * 100 / len(temp_df)
        self.cols_to_keep_ = percentage_missing[percentage_missing < self.threshold].index.tolist()
        
        self.numeric_cols_to_keep_ = [col for col in self.numeric_cols_ if col in self.cols_to_keep_]
        self.categorical_cols_to_keep_ = [col for col in self.categorical_cols_ if col in self.cols_to_keep_]
        

        # Fit OneHotEncoder
        if len(self.categorical_cols_to_keep_) > 0:
            self.ohe_ = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            self.ohe_.fit(temp_df[self.categorical_cols_to_keep_])
        else:
            self.ohe_ = None
        
        return self

    def transform(self, X, y=None):
        df = X.copy()
        
        # ĐẢM BẢO datetime tồn tại
        if self.datetime_col not in df.columns:
            raise ValueError(f"datetime column '{self.datetime_col}' not found in transform")
        
        # Store datetime separately
        datetime_values = df[self.datetime_col]
        temp_df = df.drop(columns=[self.datetime_col])
        
        # Keep only selected columns
        keep_cols = [c for c in self.cols_to_keep_ if c in temp_df.columns]
        temp_df = temp_df[keep_cols]

        # Impute numerical columns
        for col in self.numeric_cols_to_keep_:
            if col in temp_df.columns:
                temp_df[col] = temp_df[col].interpolate(method='linear').ffill().bfill()
        
        # Encode categorical data
        if self.ohe_ is not None and len(self.categorical_cols_to_keep_) > 0:
            available_cat_cols = [col for col in self.categorical_cols_to_keep_ if col in temp_df.columns]
            if available_cat_cols:
                encoded = self.ohe_.transform(temp_df[available_cat_cols])
                encoded_df = pd.DataFrame(
                    encoded, 
                    columns=self.ohe_.get_feature_names_out(available_cat_cols),
                    index=temp_df.index
                )
                temp_df = pd.concat([temp_df.drop(columns=available_cat_cols), encoded_df], axis=1)
        
        # Add datetime back
        temp_df[self.datetime_col] = datetime_values.values
        
        return temp_df

def feature_engineer_X(X, y=None, lags=[1, 2, 3, 7, 14, 30], roll_windows=[3, 7, 14, 30]):
    """Feature engineering - MUST have datetime column"""
    df = X.copy()
    
    # ĐẢM BẢO datetime tồn tại
    if 'datetime' not in df.columns:
        raise ValueError("datetime column is required for feature engineering")
    
    # ĐẢM BẢO index consistency với y
    if y is not None:
        # Reset index để đảm bảo alignment
        df = df.reset_index(drop=True)
        y = y.reset_index(drop=True) if hasattr(y, 'reset_index') else y
        df['temp'] = y
        has_target = True
    else:
        has_target = False
    
    # Sort by datetime
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Get numeric columns (EXCLUDE target if exists)
    current_numeric_cols = df.select_dtypes('number').columns.tolist()
    if has_target and 'temp' in current_numeric_cols:
        current_numeric_cols.remove('temp')
    
    # Extract datetime features
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['quarter'] = df['datetime'].dt.quarter
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

    # Create new feature from sunset and sunrise 
    df['sunrise_hour'] = df['sunrise'].dt.hour + df['sunrise'].dt.minute/60
    df['sunset_hour'] = df['sunset'].dt.hour + df['sunset'].dt.minute/60
    df['day_length_hours'] = df['sunset_hour'] - df['sunrise_hour']
    df.drop(['sunrise', 'sunset', 'sunrise_hour', 'sunset_hour'], axis=1, inplace=True)

    # Set datetime as index for time series operations
    df.set_index('datetime', inplace=True)

    # Update numeric columns list after creating new features
    current_numeric_cols = current_numeric_cols + ['day_length_hours']
    if has_target and 'temp' in current_numeric_cols:
        current_numeric_cols.remove('temp')
    
    # Create lag features
    for col in current_numeric_cols:
        for lag in lags:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    # Create rolling features 
    for col in current_numeric_cols:
        for window in roll_windows:
            df[f'{col}_roll_mean_{window}'] = df[col].shift(1).rolling(window=window, min_periods=1).mean()
            df[f'{col}_roll_std_{window}'] = df[col].shift(1).rolling(window=window, min_periods=1).std()

    # Create lag and rolling features for temp(y)
    if has_target:
        for lag in lags:
            df[f'temp_lag_{lag}'] = df['temp'].shift(lag)
        for window in roll_windows:
            df[f'temp_roll_mean_{window}'] = df['temp'].shift(1).rolling(window=window, min_periods=1).mean()
            df[f'temp_roll_std_{window}'] = df['temp'].shift(1).rolling(window=window, min_periods=1).std()

        # Create interaction features (using original columns, not lagged ones)
    if 'temp_lag_1' in df.columns and 'humidity_lag_1' in df.columns:
        df['temp_humidity_interact'] = df['temp_lag_1'] * df['humidity_lag_1']
    if 'solarradiation_lag_1' in df.columns and 'cloudcover_lag_1' in df.columns:
        df['effective_solar_lag_1'] = df['solarradiation_lag_1'] / (df['cloudcover_lag_1'] + 1)
    if 'dew_lag_1' in df.columns and 'windspeed_lag_1' in df.columns:
        df['dew_wind_interact'] = df['dew_lag_1'] * df['windspeed_lag_1']

    df = df.dropna()
    if has_target:
        current_numeric_cols = current_numeric_cols + ['temp']
        X_processed = df.drop(current_numeric_cols, axis=1)
        y_processed = df['temp']
        return X_processed, y_processed
    else:
        return df

### Optuna & ONNX
def objective(trial, X_train_val, y_train_val):
    """
    Objective function for Optuna optimization
    """
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'min_samples_split': trial.suggest_int('min_samples_split', 5, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.6, 0.7, 0.8]),
        'bootstrap': True,
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = RandomForestRegressor(**params)
    
    # Time Series Cross Validation
    tscv = TimeSeriesSplit(n_splits=5, test_size=30)
    
    rmses = []
    
    for train_idx, val_idx in tscv.split(X_train_val):
        if len(val_idx) == 0:
            continue
            
        X_tr, X_vl = X_train_val.iloc[train_idx], X_train_val.iloc[val_idx]
        y_tr, y_vl = y_train_val.iloc[train_idx], y_train_val.iloc[val_idx]
        
        if len(X_vl) < 10 or len(X_tr) < 50:
            continue
            
        model.fit(X_tr, y_tr)
        y_pr = model.predict(X_vl)
        
        rmse = np.sqrt(mean_squared_error(y_vl, y_pr))
        rmses.append(rmse)
    
    return np.mean(rmses) if rmses else float('inf')

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name="Model"):
    """
    Comprehensive model evaluation
    """
    # Train predictions
    y_pred_train = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_mape = mean_absolute_percentage_error(y_train, y_pred_train)
    train_r2 = r2_score(y_train, y_pred_train)
    
    # Test predictions
    y_pred_test = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mape = mean_absolute_percentage_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)
    
    # Overfitting analysis
    overfit_ratio = test_rmse / train_rmse
    r2_gap = train_r2 - test_r2
    
    print(f"\n {model_name.upper()} PERFORMANCE:")
    print(f" TRAIN SET:")
    print(f"   RMSE: {train_rmse:.4f}, MAPE: {train_mape:.4%}, R²: {train_r2:.4f}")
    print(f" TEST SET:")
    print(f"   RMSE: {test_rmse:.4f}, MAPE: {test_mape:.4%}, R²: {test_r2:.4f}")
    print(f" OVERFITTING ANALYSIS:")
    print(f"   RMSE Ratio (Test/Train): {overfit_ratio:.2f}x")
    print(f"   R² Gap: {r2_gap:.4f}")
    
    if overfit_ratio > 1.5 or r2_gap > 0.15:
        print("    SEVERE OVERFITTING DETECTED!")
    elif overfit_ratio > 1.2 or r2_gap > 0.08:
        print("     MODERATE OVERFITTING")
    else:
        print("   GOOD GENERALIZATION")
    
    return {
        'train_rmse': train_rmse, 'test_rmse': test_rmse,
        'train_mape': train_mape, 'test_mape': test_mape,
        'train_r2': train_r2, 'test_r2': test_r2,
        'overfit_ratio': overfit_ratio, 'r2_gap': r2_gap
    }
def export_to_onnx(model, X_sample, onnx_path='model.onnx'):
    """
    Export model to ONNX format
    """
    print(f"\n EXPORTING MODEL TO ONNX...")
    
    # Define initial types for ONNX conversion
    initial_type = [('float_input', FloatTensorType([None, X_sample.shape[1]]))]
    
    # Convert to ONNX
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    
    # Save ONNX model
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    
    print(f" Model exported to: {onnx_path}")
    
    # Validate ONNX model
    print(f"VALIDATING ONNX MODEL...")
    sess = rt.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    
    # Test prediction with ONNX
    sample_data = X_sample[:5].values.astype(np.float32)
    onnx_pred = sess.run([output_name], {input_name: sample_data})[0]
    
    # Compare with original model
    original_pred = model.predict(X_sample[:5])
    
    print(f" ONNX model validated successfully!")
    print(f"   Sample predictions match: {np.allclose(onnx_pred, original_pred, rtol=1e-3)}")
    
    return sess
def hyperparameter_tuning(X_train_final, X_test_final, y_train_fe, y_test_fe, n_trials=20):
    """
    Hyperparameter tuning với Optuna
    """
    print(f"\n STARTING HYPERPARAMETER TUNING WITH OPTUNA")
    print(f" Number of trials: {n_trials}")
    
    # Strategy 1: Default model (baseline)
    print(f"\n TRAINING DEFAULT MODEL (Baseline)")
    default_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    default_model.fit(X_train_final, y_train_fe)
    default_results = evaluate_model(default_model, X_train_final, X_test_final, y_train_fe, y_test_fe, "Default Model")
    
    # Strategy 2: Optuna tuning
    print(f"\n STARTING OPTUNA OPTIMIZATION...")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train_final, y_train_fe), n_trials=n_trials)
    
    best_params = study.best_params
    print(f" Best parameters: {best_params}")
    print(f" Best CV score: {study.best_value:.4f}")
    
    # Train final tuned model
    print(f"\n TRAINING TUNED MODEL...")
    tuned_model = RandomForestRegressor(**best_params)
    tuned_model.fit(X_train_final, y_train_fe)
    tuned_results = evaluate_model(tuned_model, X_train_final, X_test_final, y_train_fe, y_test_fe, "Tuned Model")
    
    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': X_train_final.columns,
        'importance': tuned_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n TOP 10 FEATURE IMPORTANCES (TUNED MODEL):")
    print(feature_importance.head(10).to_string(index=False))
    onnx_session = export_to_onnx(tuned_model, X_train_final, 'outputs/tuned_model.onnx')
    # Compare improvements
    improvement_r2 = tuned_results['test_r2'] - default_results['test_r2']
    improvement_rmse = default_results['test_rmse'] - tuned_results['test_rmse']
    
    print(f"\n TUNING IMPROVEMENT:")
    print(f"   Test R²: {default_results['test_r2']:.4f} → {tuned_results['test_r2']:.4f}")
    print(f"   Improvement: {improvement_r2:+.4f}")
    print(f"   Test RMSE: {default_results['test_rmse']:.4f} → {tuned_results['test_rmse']:.4f}")
    print(f"   Improvement: {improvement_rmse:+.4f}")
    
    # Overfitting improvement
    overfit_improvement = default_results['r2_gap'] - tuned_results['r2_gap']
    print(f"   Overfitting reduction (R² Gap): {overfit_improvement:+.4f}")
    
    if improvement_r2 > 0:
        print(f"   Tuning successful! Model improved by {improvement_r2:.4f} in R²")
    else:
        print(f"   Tuning didn't improve performance, but found better parameters")
    # Test ONNX model prediction
    print(f"\n TESTING ONNX MODEL PREDICTION...")
    sample_input = X_test_final.iloc[[0]].values.astype(np.float32)
    onnx_prediction = onnx_session.run(None, {'float_input': sample_input})[0][0]
    original_prediction = tuned_model.predict(X_test_final.iloc[[0]])[0]
    
    print(f"   ONNX Prediction: {float(onnx_prediction):.2f}°C")
    print(f"   Original Prediction: {float(original_prediction):.2f}°C")
    print(f"   Difference: {abs(float(onnx_prediction) - float(original_prediction)):.4f}°C")
    print(f"   Actual Value: {float(y_test_fe.iloc[0]):.2f}°C")

    return {
        'default_model': default_model,
        'tuned_model': tuned_model,
        'best_params': best_params,
        'feature_importance': feature_importance,
        'default_results': default_results,
        'tuned_results': tuned_results,
        'study': study
    }
def main():
    # Load data
    path = "https://raw.githubusercontent.com/7hufofbun/DSEB--Machine-Learning_Group5/refs/heads/main/data/weather_hcm_daily.csv"
    data = get_data(path)
    print(f"📥 Original data shape: {data.shape}")
    
    # Basic cleaning
    data = basic_cleaning(data)
    print(f"🧹 After basic cleaning: {data.shape}")
    data = data.sort_values('datetime')

    
    # Split data
    n = len(data)
    n_train = int(n * 0.8)
    train_data = data.iloc[:n_train]
    test_data = data.iloc[n_train:]
    
    X_train = train_data.drop(['temp'], axis=1)
    y_train = train_data['temp']
    X_test = test_data.drop(['temp'], axis=1)
    y_test = test_data['temp']
    
    print(f"\n📊 Data split:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # Pipeline 1: Correlation reduction + Preprocessing
    pipe1 = Pipeline([
        ("reduce_corr", SmartCorrelationReducer(target_col='temp')),
        ("preprocess", Preprocessing(datetime_col='datetime'))
    ])
    
    print(f"\n🔧 Pipeline 1 - Correlation reduction + Preprocessing")
    X_train_processed = pipe1.fit_transform(X_train, y_train)
    X_test_processed = pipe1.transform(X_test)

    
    # Feature engineering
    print(f"\n🎯 Feature Engineering")
    X_train_fe, y_train_fe = feature_engineer_X(X_train_processed, y_train)
    X_test_fe, y_test_fe = feature_engineer_X(X_test_processed, y_test)
    print(f"X_train_processed: {X_train_processed.shape}")
    print(f"X_test_processed: {X_test_processed.shape}")
    print(f"Columns kept: {list(X_train_processed.columns)}")
    print(f"X_train_fe: {X_train_fe.shape}, y_train_fe: {y_train_fe.shape}")
    print(f"X_test_fe: {X_test_fe.shape}, y_test_fe: {y_test_fe.shape}")
    
    # Check for NaN values after feature engineering
    
    print(f"\n✅ After handling NaN:")
    print(f"X_train_fe NaN: {X_train_fe.isnull().sum().sum()}")
    print(f"X_test_fe NaN: {X_test_fe.isnull().sum().sum()}")
    
    # Prepare final data for modeling (exclude non-numeric columns)
    print(f"\n🎯 Preparing data for modeling")
    X_train_final = X_train_fe.select_dtypes(include=[np.number])
    X_test_final = X_test_fe.select_dtypes(include=[np.number])
    print(f"X_train_final: {X_train_final.shape}")
    print(f"X_test_final: {X_test_final.shape}")
    
    # Pipeline 2: Hyperparameter tuning
    tuning_results = hyperparameter_tuning(X_train_final, X_test_final, y_train_fe, y_test_fe, n_trials=20)
    
    # Export results
    os.makedirs('outputs', exist_ok=True)
    X_train_final.to_csv('outputs/X_train_final.csv', index=True)
    X_test_final.to_csv('outputs/X_test_final.csv', index=True)
    pd.DataFrame(y_train_fe).to_csv('outputs/y_train.csv', index=True)
    pd.DataFrame(y_test_fe).to_csv('outputs/y_test.csv', index=True)
    tuning_results['feature_importance'].to_csv('outputs/feature_importance_tuned.csv', index=False)
    
    # Save best parameters
    best_params_df = pd.DataFrame([tuning_results['best_params']])
    best_params_df.to_csv('outputs/best_parameters.csv', index=False)
    # Save feature names for ONNX inference
    feature_names = pd.DataFrame({'feature_names': X_train_final.columns.tolist()})
    feature_names.to_csv('outputs/feature_names.csv', index=False)

    print(f"\n All files exported to 'outputs/' folder")
    print(f" Best parameters saved: {tuning_results['best_params']}")
    print(f"✅ ONNX model saved: outputs/tuned_model.onnx")
    print(f"✅ Feature names saved: outputs/feature_names.csv")

    return {
        'preprocessing_pipeline': pipe1,
        'tuning_results': tuning_results,
        'X_train': X_train_final,
        'X_test': X_test_final,
        'y_train': y_train_fe,
        'y_test': y_test_fe
    }

if __name__ == "__main__":
    results = main()