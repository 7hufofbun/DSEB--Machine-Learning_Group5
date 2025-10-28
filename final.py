import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)
from sklearn.feature_selection import VarianceThreshold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')
from clearml import Task, Logger


import random
import os

def set_seed(seed=42):
    """Set global random seed for full reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # LightGBM và Optuna sẽ dùng seed từ tham số random_state/seed
    print(f"🔒 Random seed set to {seed}")

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
def split_data(data):
    n = len(data)
    n_train = int(n * 0.85)
    train_data = data.iloc[:n_train]
    test_data = data.iloc[n_train:]

    X_train = train_data.drop(['temp'], axis=1)
    y_train = train_data['temp']
    X_test = test_data.drop(['temp'], axis=1)
    y_test = test_data['temp']

    return X_train, y_train, X_test, y_test

class Preprocessor(BaseEstimator, TransformerMixin):
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

        # Compute quantiles from train set
        self.quantiles = {}
        for col in self.numeric_cols_to_keep_:
            self.quantiles[col] = (df[col].quantile(0.05), df[col].quantile(0.95)) 
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
                # clip data by quantile
                low, high = self.quantiles[col]
                df[col] = df[col].clip(lower=low, upper=high)
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
def feature_engineer(X, y):
    df = X.copy()
    df['temp'] = y.values

    # drop unnecessary and dubious columns
    df = df.drop([ 'feelslikemax', 'feelslikemin', 'windspeedmax', 'windspeedmin', 'winspeed', 'precipcover,' 'solarenergy' 'moonphase', 'visibility'], axis = 1, errors = 'ignore')

    df = df.sort_values('datetime')

    # Create new feature follow group
    df['month'] = df['datetime'].dt.month
    df['dayofweek'] = df['datetime'].dt.dayofweek
    # df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['day_of_year'] = df['datetime'].dt.dayofyear

    # temperature range
    df['temp_range'] = df['tempmax'] - df['tempmin']

    # Day light, solar
    df['daylight_hours'] = df['daylight_hours'] = (df['sunset'] - df['sunrise']).dt.total_seconds() / 3600

    
    # Rain and cloud
    # df['rain_streak'] = (df['precip'] > 0).astype(int).shift(1).rolling(3).sum()  
    # df['has_rain_recently'] = (df['precip'].shift(1).rolling(3).sum() > 0).astype(int)
    # df['cloud_trend'] = df['cloudcover'] - df['cloudcover'].shift(1)
    # df['recent_rain_intensity'] = df['precip'].shift(1).rolling(5).mean()
    #interaction
    df['temp_humidity_interact'] = df['temp_range'] * df['humidity']
    #seasonal
    df['month_sin'] = np.sin(2 * np.pi * df['datetime'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['datetime'].dt.month / 12)
    df['dayofyear_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['dayofyear_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    # df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    # df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

    cols = ['temp', 'feelslike', 'tempmin', 'tempmax', 'dew', 'humidity',  'sealevelpressure', 'solarradiation', 'uvindex', 
            'temp_range', 'conditions_clear', 'conditions_partially_cloudy', 'conditions_rain__overcast', 'conditions_rain__partially_cloudy', 'icon_clear-day', 'icon_partly-cloudy-day', 'icon_rain' ]

    medium_term_features = [
        'temp', 'dew', 'humidity',  'sealevelpressure', 
        'humidity', 'solarradiation', 'daylight_hours'
    ]


    for slag in [1,2,3]:
        for col in cols:
            df[f'{col}_lag_{slag}'] = df[col].shift(slag)
    # for col in ['temp']:  
    #     if col in df.columns:
    #         for i in [365, 366, 367]:
    #             df[f'{col}_lag_{i}'] = df[col].shift(i)
    for llag in [5,7]:
        for col in medium_term_features:
            df[f'{col}_lag_{llag}'] = df[col].shift(llag)
    for w in [ 3,5, 7]:
        for col in cols:
            df[f'{col}_rolling_mean_{w}'] = df[col].rolling(w).mean()
            df[f'{col}_rolling_std_{w}'] = df[col].rolling(w).std()
    # df['temp_rolling_mean_60'] = df['temp'].rolling(60).mean()
    # df['temp_rolling_std_60'] = df['temp'].rolling(60).std()
    # df['temp_momentum_1d'] = df['temp_lag_1'] - df['temp_lag_2']
    # df['temp_momentum_4d'] = df['temp_lag_1'] - df['temp_lag_5']
    # df["pressure_temp_ratio_lag_1"] = df["sealevelpressure_lag_1"] / df["temp_lag_1"]
    # df['temp_trend_7_30'] = df['temp_rolling_mean_7'] - df['temp_rolling_mean_30']
    df['temp_humidity_interact'] = df['temp_range'] * df['humidity']
    df['effective_solar'] = df['solarradiation'] / (df['cloudcover'] + 1)
    df['dew_wind_interact'] = df['dew'] * df['windspeedmean']
    df = df.drop(['datetime', 'sunset', 'sunrise']  , axis=1, errors='ignore')
    df = df.dropna(axis=0)
    

    y = pd.DataFrame()
    for h in range(1, 6):
        y[f'temp_t+{h}'] = df['temp'].shift(-h)
    valid_idx = y.dropna().index
    # X= df.drop('temp', axis = 1)
    X = df.loc[valid_idx].reset_index(drop=True)
    y = y.loc[valid_idx].reset_index(drop=True)
    return X, y




# ============================================================================
# 1. CÁC HÀM TIỆN ÍCH CƠ BẢN
# ============================================================================

def calculate_all_metrics(y_true, y_pred):
    """Tính tất cả metrics quan trọng"""
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred)
    }

def prepare_data(X, y):
    """Chuẩn bị dữ liệu - sử dụng tất cả features số"""
    X_prepared = X.select_dtypes(include=[np.number])
    X_prepared = X_prepared.fillna(X_prepared.median())
    
    # Align X and y
    common_idx = X_prepared.index.intersection(y.dropna().index)
    X_final = X_prepared.loc[common_idx]
    y_final = y.loc[common_idx]
    
    return X_final, y_final

def train_final_lgb_model_all_features(X_train, y_train, best_params):
    """Train final LightGBM model trên toàn bộ training data với tất cả features"""
    
    # Prepare data - sử dụng tất cả features
    X_train_final, y_train_final = prepare_data(X_train, y_train)
    
    print(f"   Final training data: {len(X_train_final)} samples, {X_train_final.shape[1]} features")
    
    # Train LightGBM trên toàn bộ train data
    lgb_model = lgb.LGBMRegressor(
        n_estimators=best_params['n_estimators'],
        learning_rate=best_params['learning_rate'],
        max_depth=best_params['max_depth'],
        num_leaves=best_params['num_leaves'],
        min_child_samples=best_params['min_child_samples'],
        subsample=best_params['subsample'],
        colsample_bytree=best_params['colsample_bytree'],
        reg_alpha=best_params['reg_alpha'],
        reg_lambda=best_params['reg_lambda'],
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    lgb_model.fit(X_train_final, y_train_final)
    
    # Evaluate trên chính training data
    y_pred_train = lgb_model.predict(X_train_final)
    train_metrics = calculate_all_metrics(y_train_final, y_pred_train)
    
    print(f"   Final model performance on training data:")
    print(f"     RMSE: {train_metrics['rmse']:.4f}, MAE: {train_metrics['mae']:.4f}, R²: {train_metrics['r2']:.4f}")
    
    return {
        'model': lgb_model,
        'train_metrics': train_metrics,
        'feature_names': list(X_train_final.columns),  # Tất cả features
        'best_params': best_params,
        'n_features': X_train_final.shape[1]
    }

# ============================================================================
# 2. LIGHTGBM HYPERPARAMETER TUNING VỚI REGULARIZATION MẠNH
# ============================================================================

def objective_lgb_all_features(trial, X_train, y_train, target_name):
    """Objective function cho Optuna với tất cả features"""
    horizon = int(target_name.split('+')[-1])

    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'num_leaves': trial.suggest_int('num_leaves', 15, 80),
        'min_child_samples': trial.suggest_int('min_child_samples', 30, 200),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 1.0),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 0.3),
    }

    # Horizon xa -> regularization mạnh hơn
    if horizon == 1:
        params['learning_rate'] *= 1.2
        params['reg_alpha'] *= 0.8
        params['reg_lambda'] *= 0.8
    elif horizon == 2:
        pass  # giữ nguyên
    elif horizon == 3:
        params['learning_rate'] *= 0.8
        params['reg_alpha'] *= 1.2
        params['reg_lambda'] *= 1.2
    elif horizon >= 4:
        params['learning_rate'] *= 0.6
        params['reg_alpha'] *= 2.0
        params['reg_lambda'] *= 2.0
        params['min_child_samples'] = min(400, params['min_child_samples'] * 2)
        params['num_leaves'] = max(20, params['num_leaves'] - 15)

    params['n_estimators'] = int(params['n_estimators'] * (1 + 0.1 * horizon))

    # Sử dụng tất cả features
    X_final, y_final = prepare_data(X_train, y_train)

    tscv = TimeSeriesSplit(n_splits=5)
    all_metrics = {'mae': [], 'rmse': [], 'r2': [], 'mse': []}

    for train_idx, val_idx in tscv.split(X_final):
        X_tr, X_val = X_final.iloc[train_idx], X_final.iloc[val_idx]
        y_tr, y_val = y_final.iloc[train_idx], y_final.iloc[val_idx]

        model = lgb.LGBMRegressor(**params, random_state=42, n_jobs=-1)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric='rmse',
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(0)
            ]
        )
        y_pred = model.predict(X_val)
        metrics = calculate_all_metrics(y_val, y_pred)
        for k, v in metrics.items():
            all_metrics[k].append(v)

    mean_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    trial.set_user_attr('metrics', mean_metrics)

    # Penalize low R² models
    score = mean_metrics['rmse'] * (1.0 + 0.3 * (1 - mean_metrics['r2']))
    return score

def optimize_lgb_all_features(X_train, y_train, target_name, n_trials=50):
    """Optimize với tất cả features"""
    print(f"🔍 Optimizing LightGBM with ALL FEATURES for {target_name}")
    
    # Sử dụng tất cả features
    X_final, y_final = prepare_data(X_train, y_train)
    print(f"   Using ALL {X_final.shape[1]} features")
    
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='minimize', sampler=sampler)

    study.optimize(
        lambda trial: objective_lgb_all_features(trial, X_train, y_train, target_name),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    best_metrics = study.best_trial.user_attrs['metrics']
    
    print(f"✅ Best Results for {target_name} (ALL FEATURES):")
    print(f"   RMSE: {best_metrics['rmse']:.4f}, R²: {best_metrics['r2']:.4f}")
    print(f"   Best params: n_est={study.best_params['n_estimators']}, lr={study.best_params['learning_rate']:.4f}")
    print(f"   Regularization: alpha={study.best_params['reg_alpha']:.2f}, lambda={study.best_params['reg_lambda']:.2f}")
    
    return study.best_params, best_metrics

# ============================================================================
# 3. COMPLETE PIPELINE VỚI TẤT CẢ FEATURES
# ============================================================================

def complete_lgb_pipeline_all_features(X_train, y_train, optimization_params=None):
    """Pipeline sử dụng TẤT CẢ FEATURES cho từng target"""
    
    if optimization_params is None:
        optimization_params = {'n_trials': 50}
    
    print("🎯 LIGHTGBM PIPELINE WITH ALL FEATURES")
    print("=" * 70)
    print("🔧 Mỗi target sử dụng TẤT CẢ FEATURES có sẵn")
    print(f"📊 Training data: {X_train.shape}")
    
    # Sử dụng tất cả features số
    X_numeric = X_train.select_dtypes(include=[np.number])
    print(f"📈 Using ALL {X_numeric.shape[1]} numeric features for all targets")
    
    # STEP 1: Model Training với Tất Cả Features
    print("\n🔧 STEP 1: Model Training with ALL Features")
    models = {}
    best_params_dict = {}
    train_metrics_dict = {}
    cv_metrics_dict = {}
    
    for col in y_train.columns:
        print("\n" + "="*60)
        print(f"🚀 Training with ALL features for: {col}")
        print("="*60)
        
        # Optimize với tất cả features
        best_params, cv_metrics = optimize_lgb_all_features(
            X_train=X_train,
            y_train=y_train[col],
            target_name=col,
            n_trials=optimization_params['n_trials']
        )
        
        # Train final model
        model_result = train_final_lgb_model_all_features(
            X_train=X_train,
            y_train=y_train[col],
            best_params=best_params
        )
        
        models[col] = model_result
        best_params_dict[col] = best_params
        train_metrics_dict[col] = model_result['train_metrics']
        cv_metrics_dict[col] = cv_metrics
        
        # Kiểm tra overfitting
        train_test_gap = model_result['train_metrics']['r2'] - cv_metrics['r2']
        print(f"✅ {col}: Model trained with ALL {model_result['n_features']} features")
        print(f"   Train R²: {model_result['train_metrics']['r2']:.4f}")
        print(f"   Train-CV gap: {train_test_gap:.4f} {'⚠️' if train_test_gap > 0.3 else '✅'}")
    
    # STEP 2: Return pipeline
    pipeline = {
        "models": models,
        "features": {col: model_info['feature_names'] for col, model_info in models.items()},
        "best_params": best_params_dict,
        "train_metrics": train_metrics_dict,
        "cv_metrics": cv_metrics_dict,
        "feature_selection_method": "ALL FEATURES",
        "pipeline_type": "all_features"
    }
    
    # STEP 3: Summary
    print("\n🎯 ALL FEATURES PIPELINE SUMMARY")
    print("=" * 100)
    print(f"{'Target':<12} {'Features':<8} {'CV R²':<8} {'Train R²':<8} {'Gap':<8} {'n_est':<6} {'Reg Alpha':<10} {'Reg Lambda':<10}")
    print("-" * 100)
    
    for col in y_train.columns:
        if models[col] is not None:
            cv_metrics = cv_metrics_dict[col]
            train_metrics = train_metrics_dict[col]
            best_params = best_params_dict[col]
            gap = train_metrics['r2'] - cv_metrics['r2']
            
            print(f"{col:<12} {models[col]['n_features']:<8} "
                f"{cv_metrics['r2']:<8.4f} {train_metrics['r2']:<8.4f} "
                f"{gap:<8.4f} {best_params['n_estimators']:<6} "
                f"{best_params['reg_alpha']:<10.2f} {best_params['reg_lambda']:<10.2f}")
    
    return pipeline

# ============================================================================
# 4. EVALUATE ON TEST SET
# ============================================================================

def evaluate_on_test_set_summary(pipeline, X_test, y_test):
    """
    Evaluate final models trên test set và in ra metrics trung bình.
    Chỉ tập trung vào test set.
    """
    
    print("\n" + "="*80)
    print("🧪 FINAL EVALUATION ON TEST SET (SUMMARY)")
    print("="*80)
    
    test_metrics_dict = {}
    
    # Duyệt qua tất cả targets
    for col, model_info in pipeline["models"].items():
        if model_info is None:
            continue
        
        lgb_model = model_info['model']
        
        # Chuẩn bị test data với tất cả features
        X_test_final, y_test_final = prepare_data(X_test, y_test[col])
        
        # Predict và tính metrics
        y_pred_test = lgb_model.predict(X_test_final)
        test_metrics = calculate_all_metrics(y_test_final, y_pred_test)
        test_metrics_dict[col] = test_metrics
        
        print(f"{col:<12} RMSE: {test_metrics['rmse']:.4f} | MAE: {test_metrics['mae']:.4f} | R²: {test_metrics['r2']:.4f}")
    
    # Trung bình metrics cho tất cả target
    avg_rmse = np.mean([m['rmse'] for m in test_metrics_dict.values()])
    avg_mae = np.mean([m['mae'] for m in test_metrics_dict.values()])
    avg_r2 = np.mean([m['r2'] for m in test_metrics_dict.values()])
    
    print("\n📊 Average metrics across all targets:")
    print(f"   RMSE: {avg_rmse:.4f} | MAE: {avg_mae:.4f} | R²: {avg_r2:.4f}")
    
    return test_metrics_dict

# ============================================================================
# 5. COMPREHENSIVE FINAL EVALUATION - ALL DATASETS
# ============================================================================

def comprehensive_final_evaluation_with_avg(pipeline, X_train, y_train, X_test, y_test):
    """Đánh giá toàn diện trên cả 3 tập: Train, Validation (CV), và Test,
    đồng thời tính trung bình metrics trên tất cả target và tập dữ liệu."""
    
    print("\n" + "="*120)
    print("📊 COMPREHENSIVE FINAL EVALUATION - ALL DATASETS")
    print("="*120)
    
    final_results = {}
    
    # Lưu metrics để tính trung bình
    metrics_accumulator = {'train': [], 'validation': [], 'test': []}
    
    for col, model_info in pipeline["models"].items():
        if model_info is None:
            continue
            
        lgb_model = model_info['model']
        
        print(f"\n🎯 {col} - Comprehensive Evaluation:")
        print("-" * 80)
        
        # 1. TRAIN SET EVALUATION
        X_train_final, y_train_final = prepare_data(X_train, y_train[col])
        y_pred_train = lgb_model.predict(X_train_final)
        train_metrics = calculate_all_metrics(y_train_final, y_pred_train)
        
        # 2. VALIDATION SET (CV metrics from pipeline)
        cv_metrics = pipeline["cv_metrics"][col]
        
        # 3. TEST SET EVALUATION
        X_test_final, y_test_final = prepare_data(X_test, y_test[col])
        y_pred_test = lgb_model.predict(X_test_final)
        test_metrics = calculate_all_metrics(y_test_final, y_pred_test)
        
        # Store results
        final_results[col] = {
            'train': train_metrics,
            'validation': cv_metrics,
            'test': test_metrics
        }
        
        # Cộng metrics vào accumulator để tính trung bình sau
        metrics_accumulator['train'].append(train_metrics)
        metrics_accumulator['validation'].append(cv_metrics)
        metrics_accumulator['test'].append(test_metrics)
        
        # Print detailed comparison
        print(f"   {'Dataset':<12} {'R²':<8} {'RMSE':<10} {'MAE':<10}")
        print(f"   {'-'*12} {'-'*8} {'-'*10} {'-'*10}")
        print(f"   {'Train':<12} {train_metrics['r2']:<8.4f} {train_metrics['rmse']:<10.4f} {train_metrics['mae']:<10.4f}")
        print(f"   {'Validation':<12} {cv_metrics['r2']:<8.4f} {cv_metrics['rmse']:<10.4f} {cv_metrics['mae']:<10.4f}")
        print(f"   {'Test':<12} {test_metrics['r2']:<8.4f} {test_metrics['rmse']:<10.4f} {test_metrics['mae']:<10.4f}")
        
        # Generalization analysis
        train_test_gap = train_metrics['r2'] - test_metrics['r2']
        if train_test_gap > 0.3:
            print(f"   ⚠️  High train-test gap: {train_test_gap:.3f} (potential overfitting)")
        elif train_test_gap > 0.15:
            print(f"   🔶 Moderate train-test gap: {train_test_gap:.3f}")
        else:
            print(f"   ✅ Good generalization: train-test gap = {train_test_gap:.3f}")
    
    # Tính trung bình metrics cho tất cả target và tập dữ liệu
    def average_metrics(metrics_list):
        avg = {}
        for key in ['rmse', 'mae', 'r2', 'mse']:
            avg[key] = np.mean([m[key] for m in metrics_list])
        return avg
    
    avg_train = average_metrics(metrics_accumulator['train'])
    avg_validation = average_metrics(metrics_accumulator['validation'])
    avg_test = average_metrics(metrics_accumulator['test'])
    
    print("\n📊 Average metrics across all targets:")
    print(f"{'Dataset':<12} {'R²':<8} {'RMSE':<10} {'MAE':<10}")
    print(f"{'-'*12} {'-'*8} {'-'*10} {'-'*10}")
    print(f"{'Train':<12} {avg_train['r2']:<8.4f} {avg_train['rmse']:<10.4f} {avg_train['mae']:<10.4f}")
    print(f"{'Validation':<12} {avg_validation['r2']:<8.4f} {avg_validation['rmse']:<10.4f} {avg_validation['mae']:<10.4f}")
    print(f"{'Test':<12} {avg_test['r2']:<8.4f} {avg_test['rmse']:<10.4f} {avg_test['mae']:<10.4f}")
    
    return final_results, {'train': avg_train, 'validation': avg_validation, 'test': avg_test}

# ============================================================================
# 6. VISUALIZATION
# ============================================================================

def plot_comprehensive_performance(final_results):
    """Visualize comprehensive performance across all datasets"""
    
    targets = list(final_results.keys())
    
    # Prepare data
    train_r2 = [final_results[t]['train']['r2'] for t in targets]
    val_r2 = [final_results[t]['validation']['r2'] for t in targets]
    test_r2 = [final_results[t]['test']['r2'] for t in targets]
    
    train_rmse = [final_results[t]['train']['rmse'] for t in targets]
    val_rmse = [final_results[t]['validation']['rmse'] for t in targets]
    test_rmse = [final_results[t]['test']['rmse'] for t in targets]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # R² comparison
    x = np.arange(len(targets))
    width = 0.25
    
    ax1.bar(x - width, train_r2, width, label='Train R²', alpha=0.7, color='green')
    ax1.bar(x, val_r2, width, label='Validation R²', alpha=0.7, color='blue')
    ax1.bar(x + width, test_r2, width, label='Test R²', alpha=0.7, color='red')
    
    ax1.set_xlabel('Targets')
    ax1.set_ylabel('R² Score')
    ax1.set_title('R² Comparison: Train vs Validation vs Test (ALL FEATURES)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(targets, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # RMSE comparison
    ax2.bar(x - width, train_rmse, width, label='Train RMSE', alpha=0.7, color='green')
    ax2.bar(x, val_rmse, width, label='Validation RMSE', alpha=0.7, color='blue')
    ax2.bar(x + width, test_rmse, width, label='Test RMSE', alpha=0.7, color='red')
    
    ax2.set_xlabel('Targets')
    ax2.set_ylabel('RMSE (°C)')
    ax2.set_title('RMSE Comparison: Train vs Validation vs Test (ALL FEATURES)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(targets, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# 7. PREDICTION FUNCTIONS
# ============================================================================

def predict_single_target_lgb_all_features(lgb_model, X_new):
    """Dự đoán cho một target với LightGBM sử dụng tất cả features"""
    if lgb_model is None:
        return None
    
    # Prepare data với tất cả features
    X_prepared = X_new.select_dtypes(include=[np.number])
    X_prepared = X_prepared.fillna(X_prepared.median())
    
    # Predict
    predictions = lgb_model.predict(X_prepared)
    
    return predictions

def predict_multi_step_lgb_all_features(pipeline, X_new):
    """Dự đoán cho tất cả targets với LightGBM models sử dụng tất cả features"""
    predictions = {}
    
    print("🔮 MAKING PREDICTIONS WITH LIGHTGBM (ALL FEATURES)")
    print("=" * 60)
    
    for col, model_info in pipeline["models"].items():
        if model_info is None:
            predictions[col] = None
            continue
            
        lgb_model = model_info['model']
        pred = predict_single_target_lgb_all_features(lgb_model, X_new)
        predictions[col] = pred
        
        if len(pred) == 1:
            print(f"📅 {col}: {pred[0]:.2f}°C")
        else:
            print(f"📅 {col}: {[f'{p:.2f}°C' for p in pred]}")
    
    return predictions

# ============================================================================
# 8. OVERFITTING ANALYSIS
# ============================================================================

def analyze_overfitting(final_results):
    """Phân tích overfitting và đề xuất cải tiến"""
    print("\n🔍 OVERFITTING ANALYSIS & RECOMMENDATIONS")
    print("=" * 80)
    
    for target, results in final_results.items():
        train_r2 = results['train']['r2']
        test_r2 = results['test']['r2']
        gap = train_r2 - test_r2
        
        print(f"\n{target}:")
        print(f"  Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}, Gap: {gap:.4f}")
        
        if gap > 0.4:
            print(f"  🔴 SEVERE OVERFITTING - Need strong regularization")
            print(f"  💡 Recommendations: Increase min_child_samples > 100, reg_alpha/lambda > 2.0")
        elif gap > 0.25:
            print(f"  🟡 MODERATE OVERFITTING - Need regularization")
            print(f"  💡 Recommendations: Reduce num_leaves, increase reg_alpha/lambda")
        elif gap > 0.15:
            print(f"  🟢 MILD OVERFITTING - Acceptable")
            print(f"  💡 Recommendations: Minor parameter adjustments")
        else:
            print(f"  ✅ GOOD GENERALIZATION - Well regularized")

# ============================================================================
# 9. MAIN EXECUTION FUNCTION
# ============================================================================

def main_all_features_pipeline(data):
    """Main function sử dụng TẤT CẢ FEATURES cho từng target"""
    
    print("🚀 ALL FEATURES PIPELINE - MỖI TARGET SỬ DỤNG TẤT CẢ FEATURES")
    print("=" * 80)
    print(f"🔧 Configuration: ALL FEATURES for all targets")
    # -------------------------
    # Step 2: Basic Cleaning
    # -------------------------
    data = basic_cleaning(data)
    print("🧹 Data cleaned successfully!")

    # -------------------------
    # Step 3: Split Data
    # -------------------------
    X_train, y_train, X_test, y_test = split_data(data)
    print(f"📊 Train: {X_train.shape}, Test: {X_test.shape}")

    # -------------------------
    # Step 4: Preprocessing
    # -------------------------
    pre = Preprocessor()
    pre.fit(X_train, y_train)
    X_train = pre.transform(X_train)
    X_test = pre.transform(X_test)
    print("🔧 Preprocessing done!")

    # -------------------------
    # Step 5: Feature Engineering
    # -------------------------
    X_train, y_train = feature_engineer(X_train, y_train)
    X_test, y_test = feature_engineer(X_test, y_test)
    print("🧠 Feature engineering completed!")

    # -------------------------
    # Step 6: Training with ALL features
    # -------------------------
    print("\n🚀 TRAINING PIPELINE USING ALL FEATURES FOR EACH TARGET")
    pipeline = complete_lgb_pipeline_all_features(
        X_train=X_train,
        y_train=y_train,
        optimization_params={'n_trials': 30}
    )

    # -------------------------
    # Step 7: Evaluation
    # -------------------------
    test_metrics_dict = evaluate_on_test_set_summary(pipeline, X_test, y_test)
    final_results, avg_metrics = comprehensive_final_evaluation_with_avg(
        pipeline, X_train, y_train, X_test, y_test
    )

    # Attach results back to pipeline
    pipeline['test_metrics'] = avg_metrics['test']
    pipeline['final_results'] = final_results

    # -------------------------
    # Step 8: Visualization & Analysis
    # -------------------------
    plot_comprehensive_performance(final_results)
    analyze_overfitting(final_results)

    # -------------------------
    # Step 9: Export models to ONNX
    # -------------------------
    save_lgb_models_to_onnx(pipeline, save_dir="models_onnx/")

    print("\n🎉 Pipeline completed successfully!")
    return pipeline, final_results

import onnxmltools
from skl2onnx.common.data_types import FloatTensorType

# ============================================================================
# 10.onnx
# ============================================================================
def save_lgb_models_to_onnx(pipeline, save_dir="models_onnx/"):
    """
    Lưu 5 model LightGBM (temp_t+1 → temp_t+5) sang định dạng ONNX.
    Mỗi model sẽ được lưu thành 1 file .onnx riêng.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    for target_name, model_info in pipeline["models"].items():
        model = model_info["model"]
        feature_names = model_info["feature_names"]

        # Input shape = (None, n_features)
        initial_type = [('float_input', FloatTensorType([None, len(feature_names)]))]

        print(f"💾 Converting {target_name} to ONNX format...")
        onnx_model = onnxmltools.convert_lightgbm(model, initial_types=initial_type)
        
        file_path = os.path.join(save_dir, f"{target_name}.onnx")
        with open(file_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        print(f"✅ Saved: {file_path}")

    print("\n🎉 All 5 models saved to ONNX format successfully!")


# ============================================================================
# 11. EXECUTION
# ============================================================================
if __name__ == "__main__":
    set_seed(42)
    # task = Task.init(
    #     project_name="Weather Forecast HCM",
    #     task_name="LGBM All Features Pipeline",
    #     task_type=Task.TaskTypes.training,
    #     output_uri=True
    # )
    # logger = task.get_logger()

    path = "https://raw.githubusercontent.com/7hufofbun/DSEB--Machine-Learning_Group5/refs/heads/main/data/weather_hcm_daily.csv"
    data = get_data(path)

    # ✅ Sử dụng pipeline với feature selection độc lập
    pipeline, final_results = main_all_features_pipeline(
        data
    )
    # ✅ Lưu 5 model sang ONNX
    save_lgb_models_to_onnx(pipeline, save_dir="models_onnx/")
