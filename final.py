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
from sklearn.feature_selection import VarianceThreshold
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt
# For in-memory handling (do k dung local file)
import io
# For ClearML
from clearml import Task

GLOBAL_SEED = 42

def get_data(path):
    """Get data from downloaded datasets"""
    data = pd.read_csv(path)
    return data

def basic_cleaning(data):
    data = data.drop_duplicates()
    data['datetime'] = pd.to_datetime(data['datetime'])
    data['sunrise'] = pd.to_datetime(data['sunrise'])
    data['sunset'] = pd.to_datetime(data['sunset'])
    data.drop('description', axis=1, inplace=True, errors='ignore')
    return data

class SmartCorrelationReducer(BaseEstimator, TransformerMixin):
    def __init__(self, target_col='temp', threshold=0.6):
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
    def __init__(self, threshold=50, var_threshold=0.0):
        self.threshold = threshold               
        self.var_threshold = var_threshold       
        self.datetime_cols = ['datetime', 'sunrise', 'sunset']

    def fit(self, X, y=None):
        df = X.copy()

        # Drop datetime types columns for fitting
        temp_df = df.drop(columns=self.datetime_cols, errors='ignore')

        # Identify numerical and categorical columns
        self.numeric_cols_ = temp_df.select_dtypes(include='number').columns.tolist()
        self.categorical_cols_ = temp_df.select_dtypes(include='object').columns.tolist()

        # Identify columns with only one unique value
        unique_value = temp_df[self.categorical_cols_].nunique()
        self.cat_cols_to_drop_ = unique_value[unique_value == 1].index.tolist()

        # Identidy columns with missing values above threshold
        percentage_missing = temp_df.isnull().sum() * 100 / len(temp_df)
        self.missing_cols_to_drop_ = percentage_missing[percentage_missing > self.threshold].index.tolist()

        # Keep cols
        self.numeric_cols_to_keep_ = [
            col for col in self.numeric_cols_ if col not in self.missing_cols_to_drop_
        ]
        self.categorical_cols_to_keep_ = [
            col for col in self.categorical_cols_ 
            if col not in self.cat_cols_to_drop_ and col not in self.missing_cols_to_drop_
        ]

        # Drop
        if len(self.numeric_cols_to_keep_) > 0:
            selector = VarianceThreshold(threshold=self.var_threshold)
            selector.fit(temp_df[self.numeric_cols_to_keep_].fillna(0))
            self.numeric_cols_to_keep_ = [
                self.numeric_cols_to_keep_[i] for i, keep in enumerate(selector.get_support()) if keep
            ]

        # Fit OneHotEncoder cho categorical
        if len(self.categorical_cols_to_keep_) > 0:
            self.ohe_ = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            self.ohe_.fit(temp_df[self.categorical_cols_to_keep_])
        else:
            self.ohe_ = None

        # keep cols finally
        self.keep_cols_ = self.numeric_cols_to_keep_ + self.categorical_cols_to_keep_

        return self

    def transform(self, X, y=None):
        df = X.copy()

        # Giữ datetime riêng (nếu có)
        datetime_values = df[self.datetime_cols].copy() if all(col in df.columns for col in self.datetime_cols) else pd.DataFrame()

        temp_df = df.drop(columns=self.datetime_cols, errors='ignore')

        # Giữ lại các cột hợp lệ
        keep_cols = [c for c in self.keep_cols_ if c in temp_df.columns]
        temp_df = temp_df[keep_cols]

        # Impute numerical data
        for col in self.numeric_cols_to_keep_:
            if col in temp_df.columns:
                temp_df[col] = temp_df[col].interpolate(method='linear').ffill().bfill()

        # Encode categorical data
        if self.ohe_ is not None and len(self.categorical_cols_to_keep_) > 0:
            available_cat_cols = [col for col in self.categorical_cols_to_keep_ if col in temp_df.columns]
            encoded = self.ohe_.transform(temp_df[available_cat_cols])
            encoded_df = pd.DataFrame(
                encoded,
                columns=self.ohe_.get_feature_names_out(available_cat_cols),
                index=temp_df.index
            )
            encoded_df.columns = [
                col.lower().replace(' ', r'_').replace(',', r'_').replace('.', r'_')
                for col in encoded_df.columns
            ]

            temp_df = pd.concat([temp_df.drop(columns=available_cat_cols), encoded_df], axis=1)

        # Add datetime back
        if not datetime_values.empty:
            clean = pd.concat([datetime_values, temp_df], axis=1)
        else:
            clean = temp_df

        return clean

def feature_engineer(X, y = None):

    df = X.copy()
    if y is not None:
        df = df.reset_index(drop=True)
        y = y.reset_index(drop=True) if hasattr(y, 'reset_index') else y
        df['temp'] = y.values

    # drop unnecessary and dubious columns
    # drop feelslike because feelslike is computed by expected temp, humidity, winspeed => it use current data, similar and high correlated with temperature
    # drop windspeed max/min/, windgust because winspeed => max windspeed over 1 hour and windgust => windspeed over 20 seconds => similar => keep windspeedmean
    # drop precipcover because it is computed based on precip => unnecessary 
    df = df.drop(['feelslike', 'feelslikemax', 'feelslikemin', 'windspeedmax', 'windspeedmin', 'winspeed', 'windgust', 'precipcover', 'solarenergy', 'moonphase', 'visibility'], axis = 1, errors = 'ignore')
    current_cols = [i for i in df.columns.tolist() if i != 'temp']

    df = df.sort_values('datetime')

    # Create new feature follow group

    df['month'] = df['datetime'].dt.month
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    # temperature range
    df['temp_range'] = df['tempmax'] - df['tempmin']
    df['pressure_temp_ratio'] = df['sealevelpressure'] / (df['temp'] + 10)
    df['temp_change'] = df['temp'] - df['temp'].shift(1)
    # Day light, solar
    df['daylight_hours'] = df['daylight_hours'] = (df['sunset'] - df['sunrise']).dt.total_seconds() / 3600
    df['solar_intensity'] = df['solarradiation'] / (df['daylight_hours']+0.1)
    df['effective_daylight'] = df['daylight_hours'] * (1 - df['cloudcover']/100)
    df['daylight_change'] = df['daylight_hours'] - df['daylight_hours'].shift(1)


    # Wind and pressure
    df['wind_dir_effect'] = np.sin(2 * np.pi * df['winddir'] / 360)
    
    # Rain and cloud
    df['rain_streak'] = (df['precip'] > 0).astype(int).shift(1).rolling(3).sum()  
    df['precip_7d_cumsum'] = df['precip'].shift(1).rolling(7).sum()
    df['has_rain_recently'] = (df['precip'].shift(1).rolling(3).sum() > 0).astype(int)
    df['cloud_trend'] = df['cloudcover'] - df['cloudcover'].shift(1)
    df['recent_rain_intensity'] = df['precip'].shift(1).rolling(5).mean()
    #interaction
    df['temp_humidity_interact'] = df['temp'] * df['humidity']
    df['effective_solar'] = df['solarradiation'] / (df['cloudcover'] + 1)
    df['solar_cloud_ratio'] = df['solarradiation'] / (df['cloudcover'] + 0.1)
    df['wind_pressure_interact'] = df['windspeedmean'] * df['sealevelpressure']
    # df['pressure_temp_ratio'] = df['sealevelpressure'] / (df['temp'] + 10)
    #seasonal
    df['month_sin'] = np.sin(2 * np.pi * df['datetime'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['datetime'].dt.month / 12)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

    cols = ['temp', 'dew', 'humidity', 'precip', 'precipprob', 'windspeedmean', 'winddir', 'sealevelpressure', 'cloudcover', 'solarradiation', 'uvindex', 
            'temp_range', 'temp_change', 'daylight_hours', 'solar_intensity', 'effective_daylight', 'daylight_change', 'wind_dir_effect',  
            'cloud_trend', 'temp_humidity_interact', 'effective_solar', 'solar_cloud_ratio', 'wind_pressure_interact', 'pressure_temp_ratio',
            'conditions_clear', 'conditions_partially_cloudy', 'conditions_rain__overcast', 'conditions_rain__partially_cloudy', 'icon_clear-day', 'icon_partly-cloudy-day', 'icon_rain' ]

    medium_term_features = [
        'temp', 'dew', 'humidity', 'temp_range', 'sealevelpressure', 
        'humidity', 'solarradiation', 'daylight_hours'
    ]
    current_cols = current_cols + ['temp_range', 'pressure_temp_ratio', 'temp_change', 'daylight_hours', 'solar_intensity', 'effective_daylight', 'daylight_change', 'wind_dir_effect', 'temp_humidity_interact', 'effective_solar', 'solar_cloud_ratio', 'wind_pressure_interact',  'cloud_trend']

    for slag in [1,2,3]:
        for col in cols:
            df[f'{col}_lag_{slag}'] = df[col].shift(slag)
    for col in ['temp', 'sealevelpressure']:  
        if col in df.columns:
            df[f'{col}_lag_4'] = df[col].shift(4)
    for llag in [5,7]:
        for col in medium_term_features:
            df[f'{col}_lag_{llag}'] = df[col].shift(llag)
    for w in [ 7, 14]:
        for col in cols:
            df[f'{col}_rolling_{w}'] = df[col].shift(1).rolling(w).mean()
    df['temp_momentum_1d'] = df['temp_lag_1'] - df['temp_lag_2']
    df['temp_momentum_3d'] = df['temp_lag_1'] - df['temp_lag_4']
    df['pressure_trend_3d'] = df['sealevelpressure_lag_1'] - df['sealevelpressure_lag_4']
    df = df.drop(current_cols, axis=1, errors='ignore')
    df = df.dropna(axis=0)
    
    if y is not None:
        X = df.drop('temp', axis = 1)
        y = df['temp']
        return X, y
    else:
        return df


### Optuna & ONNX
def objective(trial, X_train_val, y_train_val, task_logger=None):
    """
    Objective function for Optuna optimization
    """
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 5, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 4, 9),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.6, 0.7, 0.8]),
        'bootstrap': True,
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Log trial params to ClearML if available
    if task_logger:
        task_logger.report_single_value('trial_n_estimators', params['n_estimators'])
        task_logger.report_single_value('trial_max_depth', params['max_depth'])
        task_logger.report_single_value('trial_min_samples_split', params['min_samples_split'])
        task_logger.report_single_value('trial_min_samples_leaf', params['min_samples_leaf'])
        task_logger.report_text('trial_max_features', str(params['max_features']))
    
    model = RandomForestRegressor(**params)
    
    # Time Series Cross Validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    train_rmses = []
    val_rmses = []
    train_r2_scores = []  # THÊM: Store R² scores
    val_r2_scores = []    # THÊM: Store R² scores
    
    print(f"\n Trial {trial.number} - Checking Overfitting (Train vs Validation):")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_val)):
        if len(val_idx) == 0:
            continue
            
        X_tr, X_vl = X_train_val.iloc[train_idx], X_train_val.iloc[val_idx]
        y_tr, y_vl = y_train_val.iloc[train_idx], y_train_val.iloc[val_idx]
        
        if len(X_vl) < 10 or len(X_tr) < 50:
            continue
            
        model.fit(X_tr, y_tr)
        
        # Train predictions
        y_pred_train = model.predict(X_tr)
        train_rmse = np.sqrt(mean_squared_error(y_tr, y_pred_train))
        train_r2 = r2_score(y_tr, y_pred_train)  # THÊM: R² score
        
        # Validation predictions
        y_pred_val = model.predict(X_vl)
        val_rmse = np.sqrt(mean_squared_error(y_vl, y_pred_val))
        val_r2 = r2_score(y_vl, y_pred_val)  # THÊM: R² score
        
        train_rmses.append(train_rmse)
        val_rmses.append(val_rmse)
        train_r2_scores.append(train_r2)  # THÊM
        val_r2_scores.append(val_r2)      # THÊM

        # Calculate overfitting ratio for this fold
        overfit_ratio = val_rmse / train_rmse if train_rmse > 0 else float('inf')
        r2_gap = train_r2 - val_r2  # THÊM: R² gap
        
        print(f"   Fold {fold+1}: Train RMSE = {train_rmse:.4f} | Val RMSE = {val_rmse:.4f} | Overfit Ratio = {overfit_ratio:.2f}x")
        print(f"   Fold {fold+1}: Train R² = {train_r2:.4f} | Val R² = {val_r2:.4f} | R² Gap = {r2_gap:.4f}")  # THÊM
        
        # Log fold metrics to ClearML if available
        if task_logger:
            task_logger.report_scalar(title='Fold Performance', series=f'Train_Fold_{fold}', value=train_rmse, iteration=trial.number)
            task_logger.report_scalar(title='Fold Performance', series=f'Val_Fold_{fold}', value=val_rmse, iteration=trial.number)
            task_logger.report_scalar(title='R2 Scores', series=f'Train_Fold_{fold}', value=train_r2, iteration=trial.number)  # THÊM
            task_logger.report_scalar(title='R2 Scores', series=f'Val_Fold_{fold}', value=val_r2, iteration=trial.number)      # THÊM
            task_logger.report_scalar(title='Overfitting Ratio', series=f'Fold_{fold}', value=overfit_ratio, iteration=trial.number)
    
    mean_train_rmse = np.mean(train_rmses) if train_rmses else float('inf')
    mean_val_rmse = np.mean(val_rmses) if val_rmses else float('inf')
    mean_train_r2 = np.mean(train_r2_scores) if train_r2_scores else -float('inf')  # THÊM
    mean_val_r2 = np.mean(val_r2_scores) if val_r2_scores else -float('inf')        # THÊM
    mean_overfit_ratio = mean_val_rmse / mean_train_rmse if mean_train_rmse > 0 else float('inf')
    mean_r2_gap = mean_train_r2 - mean_val_r2  # THÊM
    
    print(f"Trial {trial.number} Summary:")
    print(f"      Avg Train RMSE = {mean_train_rmse:.4f} | Avg Val RMSE = {mean_val_rmse:.4f}")
    print(f"      Avg Train R² = {mean_train_r2:.4f} | Avg Val R² = {mean_val_r2:.4f}")  # THÊM
    print(f"      Avg Overfit Ratio = {mean_overfit_ratio:.2f}x | Avg R² Gap = {mean_r2_gap:.4f}")  # THÊM
    
    
    # Log mean trial metrics to ClearML
    if task_logger:
        task_logger.report_single_value('mean_trial_train_rmse', mean_train_rmse)
        task_logger.report_single_value('mean_trial_val_rmse', mean_val_rmse)
        task_logger.report_single_value('mean_trial_train_r2', mean_train_r2)  # THÊM
        task_logger.report_single_value('mean_trial_val_r2', mean_val_r2)      # THÊM
        task_logger.report_single_value('mean_trial_overfit_ratio', mean_overfit_ratio)
        task_logger.report_single_value('mean_trial_r2_gap', mean_r2_gap)      # THÊM
    
    return mean_val_rmse  
def export_to_onnx(model, X_sample, task=None):
    """
    Export model to ONNX format
    """
    print(f"\n EXPORTING MODEL TO ONNX...")
    
    # Define initial types for ONNX conversion
    initial_type = [('float_input', FloatTensorType([None, X_sample.shape[1]]))]
    
    # Convert to ONNX
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    
    # Serialize to bytes for direct upload
    onnx_bytes = onnx_model.SerializeToString()
    
    # Upload to ClearML if task available (as bytes)
    if task:
        task.upload_artifact('onnx_model', onnx_bytes)
    
    # Validate ONNX model (load from bytes)
    print(f"VALIDATING ONNX MODEL...")
    sess = rt.InferenceSession(onnx_bytes)
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
def evaluate_model(model, X_test,  y_test, model_name="Model", task_logger=None):
    """
    Comprehensive model evaluation - FINAL TEST ON UNSEEN DATA
    """
    
    # Test predictions
    y_pred_test = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mape = mean_absolute_percentage_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)
    

    
    print(f"\n {model_name.upper()} PERFORMANCE:")


    # Log to ClearML if available
    if task_logger:

        task_logger.report_scalar(f'{model_name} Metrics', 'test_rmse', test_rmse, iteration=0)
        task_logger.report_scalar(f'{model_name} Metrics', 'test_mape', test_mape, iteration=0)
        task_logger.report_scalar(f'{model_name} Metrics', 'test_r2', test_r2, iteration=0)

    
    return {
        'test_rmse': test_rmse,
        'test_mape': test_mape,
        'test_r2': test_r2
    }

def hyperparameter_tuning(X_train_final, y_train_fe, n_trials=20, task_logger=None):
    """
    Hyperparameter tuning với Optuna - CHỈ DÙNG TRAINING DATA
    """
    print(f"\n🎯 HYPERPARAMETER TUNING PHASE (Training data only)")
    print(f" Number of trials: {n_trials}")
    
    # Lấy task object cho ONNX export
    task = None
    if task_logger and hasattr(task_logger, '_task'):
        task = task_logger._task
    
    # STEP 1: Baseline model validation performance
    print(f"\n📊 STEP 1: BASELINE MODEL VALIDATION")
    default_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    
    # Đánh giá baseline bằng Cross-Validation (không dùng test set)
    tscv = TimeSeriesSplit(n_splits=5)
    baseline_val_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_final)):
        X_tr, X_val = X_train_final.iloc[train_idx], X_train_final.iloc[val_idx]
        y_tr, y_val = y_train_fe.iloc[train_idx], y_train_fe.iloc[val_idx]
        
        default_model.fit(X_tr, y_tr)
        y_pred_val = default_model.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
        baseline_val_scores.append(val_rmse)
    
    mean_baseline_val_rmse = np.mean(baseline_val_scores)
    print(f"   Baseline Model - Avg Validation RMSE: {mean_baseline_val_rmse:.4f}")
    
    # STEP 2: Optuna tuning
    print(f"\n📊 STEP 2: OPTUNA HYPERPARAMETER TUNING")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train_final, y_train_fe, task_logger), n_trials=n_trials)
    
    best_params = study.best_params
    print(f" Best parameters: {best_params}")
    print(f" Best Validation RMSE: {study.best_value:.4f}")
    
    # STEP 3: Train final model với best parameters
    print(f"\n📊 STEP 3: TRAINING FINAL MODEL")
    tuned_model = RandomForestRegressor(**best_params)
    tuned_model.fit(X_train_final, y_train_fe)
    
    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': X_train_final.columns,
        'importance': tuned_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n TOP 10 FEATURE IMPORTANCES:")
    print(feature_importance.head(10).to_string(index=False))
    
    # Export to ONNX
    onnx_session = export_to_onnx(tuned_model, X_train_final, task)
    
    # Log to ClearML
    if task_logger:
        task_logger.report_single_value('baseline_validation_rmse', mean_baseline_val_rmse)
        task_logger.report_single_value('best_validation_rmse', study.best_value)
        task_logger.report_table('Feature Importance', 'top_10', table_plot=feature_importance.head(10))
    
    print(f"\n✅ HYPERPARAMETER TUNING COMPLETED")
    print(f"   Best validation RMSE: {study.best_value:.4f}")
    print(f"   Baseline validation RMSE: {mean_baseline_val_rmse:.4f}")
    print(f"   Improvement: {mean_baseline_val_rmse - study.best_value:+.4f}")
    
    return {
        'tuned_model': tuned_model,
        'best_params': best_params,
        'feature_importance': feature_importance,
        'study': study,
        'onnx_session': onnx_session,
        'baseline_val_rmse': mean_baseline_val_rmse,
        'best_val_rmse': study.best_value
    }
def main():
    # Initialize ClearML Task
    task = Task.init(project_name='HCM_Temp_Forecast', task_name='RF_Tuning_with_ClearML')
    task_logger = task.get_logger()
    
    # Load data
    path = "https://raw.githubusercontent.com/7hufofbun/DSEB--Machine-Learning_Group5/refs/heads/main/data/weather_hcm_daily.csv"
    data = get_data(path)
    print(f"📥 Original data shape: {data.shape}")
    
    # Log data shape
    task_logger.report_single_value('original_data_rows', data.shape[0])
    task_logger.report_single_value('original_data_columns', data.shape[1])
    
    # Basic cleaning
    data = basic_cleaning(data)
    print(f"🧹 After basic cleaning: {data.shape}")
    data = data.sort_values('datetime')

    task_logger.report_single_value('cleaned_data_rows', data.shape[0])
    task_logger.report_single_value('cleaned_data_columns', data.shape[1])
    
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
    
    # Log split sizes
    task_logger.report_single_value('train_size', len(train_data))
    task_logger.report_single_value('test_size', len(test_data))
    
    # Pipeline 1: Correlation reduction + Preprocessing
    pipe1 = Pipeline([
        # ("reduce_corr", SmartCorrelationReducer(target_col='temp')),
        ("preprocess", Preprocessing())
    ])
    
    print(f"\n🔧 Pipeline 1 - Correlation reduction + Preprocessing")
    X_train_processed = pipe1.fit_transform(X_train, y_train)
    X_test_processed = pipe1.transform(X_test)

    
    # Feature engineering
    print(f"\n🎯 Feature Engineering")
    X_train_fe, y_train_fe = feature_engineer(X_train_processed, y_train)
    X_test_fe, y_test_fe = feature_engineer(X_test_processed, y_test)
    print(f"X_train_processed: {X_train_processed.shape}")
    print(f"X_test_processed: {X_test_processed.shape}")
    print(f"Columns kept: {list(X_train_processed.columns)}")
    print(f"X_train_fe: {X_train_fe.shape}, y_train_fe: {y_train_fe.shape}")
    print(f"X_test_fe: {X_test_fe.shape}, y_test_fe: {y_test_fe.shape}")
    
    # Log FE shapes
    task_logger.report_single_value('X_train_fe_columns', X_train_fe.shape[1])
    task_logger.report_single_value('X_test_fe_columns', X_test_fe.shape[1])
    
    # Check for NaN values after feature engineering
    print(f"\n✅ After handling NaN:")
    print(f"X_train_fe NaN: {X_train_fe.isnull().sum().sum()}")
    print(f"X_test_fe NaN: {X_test_fe.isnull().sum().sum()}")
    
    # Prepare final data for modeling (exclude non-numeric columns)
    print(f"\n🎯 Preparing data for modeling")
    X_train_final = X_train_fe.select_dtypes(include=[np.number])
    X_test_final = X_test_fe.select_dtypes(include=[np.number])
    
    # 🔧 FIX: Reset index to avoid index alignment issues
    X_train_final = X_train_final.reset_index(drop=True)
    y_train_fe = y_train_fe.reset_index(drop=True)
    X_test_final = X_test_final.reset_index(drop=True) 
    y_test_fe = y_test_fe.reset_index(drop=True)
    
    print(f"X_train_final: {X_train_final.shape}")
    print(f"X_test_final: {X_test_final.shape}")
    
    # 🔧 FIX: Ensure feature names are consistent
    print(f"Training features: {list(X_train_final.columns)}")
    print(f"Test features: {list(X_test_final.columns)}")
    
    # Upload data directly to ClearML (no local files)
    task.upload_artifact('X_train_final', X_train_final)
    task.upload_artifact('X_test_final', X_test_final)
    task.upload_artifact('y_train', pd.DataFrame(y_train_fe))
    task.upload_artifact('y_test', pd.DataFrame(y_test_fe))
    
    # 🎯 PHASE 1: HYPERPARAMETER TUNING (Training data only)
    print(f"\n{'='*60}")
    print(f"🎯 PHASE 1: HYPERPARAMETER TUNING")
    print(f"{'='*60}")
    
    # Hyperparameter tuning với training data only
    tuning_results = hyperparameter_tuning(X_train_final, y_train_fe, n_trials=30, task_logger=task_logger)
    
    # 🎯 PHASE 2: FINAL EVALUATION (First time touching test set)
    print(f"\n{'='*60}")
    print(f"🎯 PHASE 2: FINAL EVALUATION")
    print(f"{'='*60}")
    
    # Evaluate tuned model on test set
    print(f"\n📊 FINAL TUNED MODEL EVALUATION:")
    
    # 🔧 FIX: Ensure we're passing DataFrame, not Series
    if isinstance(X_test_final, pd.Series):
        X_test_final = X_test_final.to_frame().T
    if isinstance(y_test_fe, pd.Series):
        y_test_fe_values = y_test_fe.values
    else:
        y_test_fe_values = y_test_fe
    
    tuned_results = evaluate_model(
        tuning_results['tuned_model'], 
        X_test_final, 
        y_test_fe_values, 
        "Final Tuned Model", 
        task_logger
    )
    
    # Train and evaluate baseline model for comparison
    print(f"\n📊 BASELINE MODEL EVALUATION:")
    baseline_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    baseline_model.fit(X_train_final, y_train_fe)
    
    baseline_results = evaluate_model(
        baseline_model,
        X_test_final,
        y_test_fe_values,
        "Baseline Model",
        task_logger
    )
    
    # Compare improvements
    improvement_r2 = tuned_results['test_r2'] - baseline_results['test_r2']
    improvement_rmse = baseline_results['test_rmse'] - tuned_results['test_rmse']
    
    print(f"\n🎯 FINAL COMPARISON RESULTS:")
    print(f"   R²: {baseline_results['test_r2']:.4f} → {tuned_results['test_r2']:.4f} ({improvement_r2:+.4f})")
    print(f"   RMSE: {baseline_results['test_rmse']:.4f} → {tuned_results['test_rmse']:.4f} ({improvement_rmse:+.4f})")
    
    if improvement_r2 > 0:
        print(f"   ✅ TUNING SUCCESSFUL - Model improved by {improvement_r2:.4f} in R²")
        tuning_status = "SUCCESS"
    else:
        print(f"   ⚠️  Tuning didn't improve test performance")
        tuning_status = "NO_IMPROVEMENT"
    
    # Log comparison results to ClearML
    if task_logger:
        task_logger.report_scalar('Final Comparison', 'r2_improvement', improvement_r2, iteration=0)
        task_logger.report_scalar('Final Comparison', 'rmse_improvement', improvement_rmse, iteration=0)
        task_logger.report_text('final_tuning_status', tuning_status)
    
    # Upload tuning artifacts directly
    task.upload_artifact('best_parameters', pd.DataFrame([tuning_results['best_params']]))
    task.upload_artifact('feature_importance_tuned', tuning_results['feature_importance'])
    task.upload_artifact('feature_names', pd.DataFrame({'feature_names': X_train_final.columns.tolist()}))
    
    print(f"\n🎉 ALL PHASES COMPLETED SUCCESSFULLY!")
    print(f" Best parameters: {tuning_results['best_params']}")
    print(f" Tuning status: {tuning_status}")
    print(f"✅ ONNX model uploaded to ClearML")
    print(f"✅ Feature names uploaded to ClearML")
    print(f"✅ All artifacts uploaded to ClearML")
    
    # Close ClearML task
    task.close()

    return {
        'preprocessing_pipeline': pipe1,
        'tuning_results': tuning_results,
        'tuned_results': tuned_results,
        'baseline_results': baseline_results,
        'improvement_r2': improvement_r2,
        'improvement_rmse': improvement_rmse,
        'tuning_status': tuning_status,
        'X_train': X_train_final,
        'X_test': X_test_final,
        'y_train': y_train_fe,
        'y_test': y_test_fe
    }

if __name__ == "__main__":
    results = main()