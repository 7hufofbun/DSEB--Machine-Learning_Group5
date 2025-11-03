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
            # --- Remove duplicate-variance columns (keep only one) ---
            variances = temp_df[self.numeric_cols_to_keep_].var()
            var_groups = {}

            for col, var in variances.items():
                var = round(var, 6)
                if var not in var_groups:
                    var_groups[var] = [col]
                else:
                    var_groups[var].append(col)

            cols_to_remove = []
            for var, cols in var_groups.items():
                if len(cols) > 1:
                    cols_to_remove.extend(cols[1:])

            self.numeric_cols_to_keep_ = [
                col for col in self.numeric_cols_to_keep_ if col not in cols_to_remove
            ]

        # Compute quantiles from train set
        # self.quantiles = {}
        # for col in self.numeric_cols_to_keep_:
        #     self.quantiles[col] = (df[col].quantile(0.05), df[col].quantile(0.95)) 
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
                # low, high = self.quantiles[col]
                # df[col] = df[col].clip(lower=low, upper=high)
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
    y_series = pd.Series(y).copy()

    # Đồng bộ index ban đầu
    df = df.reset_index(drop=True)
    y_series = y_series.iloc[df.index].reset_index(drop=True)

    # --- Tiền xử lý như cũ (drop, datetime features, v.v.) ---
    df = df.drop(['feelslikemax', 'feelslikemin', 'feelslike', 'moonphase', 'visibility'], axis=1, errors='ignore')

    encoded_fea = ['conditions_clear', 'conditions_partially_cloudy', 'conditions_rain__overcast',
                'conditions_rain__partially_cloudy', 'icon_clear-day', 'icon_partly-cloudy-day', 'icon_rain']
    base_fea = [col for col in df.columns if col not in encoded_fea and col != 'datetime']

    df = df.sort_values('datetime').reset_index(drop=True)
    y_series = y_series.loc[df.index].reset_index(drop=True)

    # Thêm datetime features
    df['hour'] = df['datetime'].dt.hour
    df['dayofyear'] = df['datetime'].dt.dayofyear
    df['month'] = df['datetime'].dt.month
    df['weekday'] = df['datetime'].dt.weekday
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)

    df['sin_day'] = np.sin(2 * np.pi * df['dayofyear'] / 365.25)
    df['cos_day'] = np.cos(2 * np.pi * df['dayofyear'] / 365.25)
    df['sin_month'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
    df['cos_month'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)

    df['daylighthour'] = (df['sunset'] - df['sunrise']).dt.total_seconds() / 3600
    base_fea.append('daylighthour')
    df['is_rainy_season'] = df['month'].isin([5,6,7,8,9,10,11]).astype(int)
    # Set index để tạo lag/rolling
    df_indexed = df.set_index('datetime')

    # Tạo lag/rolling (như cũ)
    for slag in [1, 3, 5]:
        for col in base_fea:
            if col in ['sunrise', 'sunset']:
                continue
            df_indexed[f'{col}_lag_{slag}'] = df_indexed[col].shift(slag)
            df_indexed[f'{col}_d{slag}'] = df_indexed[col].diff(slag)

    for w in [3, 5, 7, 14, 30]:
        for col in base_fea:
            if col in ['sunrise', 'sunset']:
                continue
            df_indexed[f'{col}_rolling_mean_{w}'] = df_indexed[col].shift(1).rolling(w, min_periods=1).mean()
            df_indexed[f'{col}_rolling_std_{w}'] = df_indexed[col].shift(1).rolling(w, min_periods=1).std()

    # Interaction features
    delta1_cols = [c for c in df_indexed.columns if '_d1' in c]
    rolling_cols = [c for c in df_indexed.columns if 'rolling_mean_7' in c or 'rolling_std_7' in c]
    for dcol in delta1_cols:
        for rcol in rolling_cols:
            df_indexed[f'{dcol}_x_{rcol}'] = df_indexed[dcol].fillna(0) * df_indexed[rcol].fillna(0)
        df_indexed[f'{dcol}_sq'] = df_indexed[dcol] ** 2
        df_indexed[f'{dcol}_sqrt_abs'] = np.sqrt(np.abs(df_indexed[dcol].fillna(0)))

    # Reset index
    df_final = df_indexed.reset_index()
    df_final = df_final.dropna(axis = 0)

    #  QUAN TRỌNG: Tạo Y TRÊN DỮ LIỆU GỐC (trước khi dropna)
    Y_dict = {}
    for h in [1, 2, 3, 4, 5]:
        Y_dict[f'y+{h}'] = y_series.shift(-h)  # shift trên y_series GỐC
        Y_dict[f'datetime+{h}'] = df_final['datetime'] + pd.Timedelta(days=h)
    Y_df = pd.DataFrame(Y_dict)

    #  Ghép X và Y rồi dropna ĐỒNG BỘ
    combined = pd.concat([df_final, Y_df], axis=1)
    # Chỉ giữ hàng có đủ tất cả y+1 đến y+5
    target_cols = [f'y+{h}' for h in [1,2,3,4,5]] + [f'datetime+{h}' for h in [1,2,3,4,5]]
    combined = combined.dropna(subset=target_cols + ['datetime', 'sunset', 'sunrise']).reset_index(drop=True)

    # Tách lại
    X_final = combined[df_final.columns]
    Y_final = combined[target_cols]

    return X_final, Y_final

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

def get_model( best_params):
    """Lựa chọn model dựa trên model_type"""

    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(
        n_estimators=best_params.get('n_estimators', 300),
        max_depth=best_params.get('max_depth', 12),
        min_samples_split=best_params.get('min_samples_split', 2),
        min_samples_leaf=best_params.get('min_samples_leaf', 10),
        max_features = 'sqrt',
        random_state=42,
        n_jobs=-1
    )
def train_final_model_all_features(X_train, y_train, best_params):
    """Train final model trên toàn bộ training data với tất cả features"""
    
    X_train_final, y_train_final = prepare_data(X_train, y_train)
    # Get model
    model = get_model( best_params)
    
    # Train model
    model.fit(X_train_final, y_train_final)
    
    # Evaluate trên chính training data
    y_pred_train = model.predict(X_train_final)
    train_metrics = calculate_all_metrics(y_train_final, y_pred_train)
    return {
        'model': model,
        'train_metrics': train_metrics,
        'feature_names': list(X_train_final.columns),
        'best_params': best_params,
        'n_features': X_train_final.shape[1]
    }


# ============================================================================
# 2. HYPERPARAMETER TUNING VỚI REGULARIZATION MẠNH
# ============================================================================
def get_model_params(trial,  target_name):
    """Lấy parameters cho từng model type"""
    horizon = int(target_name.split('+')[-1])

    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 6),
        'min_samples_split': trial.suggest_int('min_samples_split', 40, 100),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 20, 50),
        'max_features': trial.suggest_float('max_features', 0.05, 0.15),
    }
    
    # Horizon-based adjustments for Random Forest
    if horizon >= 4:
        params['min_samples_split'] = min(50, params['min_samples_split'] * 2)
        params['min_samples_leaf'] = min(20, params['min_samples_leaf'] * 2)
    
    return params
def objective_all_features(trial, X_train, y_train, target_name):
    """Objective function cho Optuna với tất cả features và model type"""
    
    # Lấy parameters cho model type
    params = get_model_params(trial, target_name)
    
    # Sử dụng tất cả features
    X_final, y_final = prepare_data(X_train, y_train)

    tscv = TimeSeriesSplit(n_splits=5)
    all_metrics = {'mae': [], 'rmse': [], 'r2': [], 'mse': []}

    for train_idx, val_idx in tscv.split(X_final):
        X_tr, X_val = X_final.iloc[train_idx], X_final.iloc[val_idx]
        y_tr, y_val = y_final.iloc[train_idx], y_final.iloc[val_idx]

        model = get_model( params)
        model.fit(X_tr, y_tr)
            
        y_pred = model.predict(X_val)
        metrics = calculate_all_metrics(y_val, y_pred)
        for k, v in metrics.items():
            all_metrics[k].append(v)

    mean_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    trial.set_user_attr('metrics', mean_metrics)

    # Penalize low R² models
    score = mean_metrics['rmse'] * (1.0 + 0.3 * (1 - mean_metrics['r2']))
    return score

def optimize_model_all_features(X_train, y_train, target_name,  n_trials=30):
    """Optimize với tất cả features và model type"""
    # Sử dụng tất cả features
    X_final, y_final = prepare_data(X_train, y_train)
    
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='minimize', sampler=sampler)

    study.optimize(
        lambda trial: objective_all_features(trial, X_train, y_train, target_name),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    best_metrics = study.best_trial.user_attrs['metrics']

    return study.best_params, best_metrics

# ============================================================================
# 3. COMPLETE PIPELINE VỚI TẤT CẢ FEATURES
# ============================================================================

def complete_model_pipeline_all_features(X_train, y_train,  optimization_params=None):
    """Pipeline với lựa chọn model"""
    
    if optimization_params is None:
        optimization_params = {'n_trials': 20}
    

    X_numeric = X_train.select_dtypes(include=[np.number])
    
    # STEP 1: Model Training với Tất Cả Features

    models = {}
    best_params_dict = {}
    train_metrics_dict = {}
    cv_metrics_dict = {}
    
    target_columns = [c for c in y_train.columns if c.startswith("y+")]
    for col in target_columns:

        # Optimize với model đã chọn
        best_params, cv_metrics = optimize_model_all_features(
            X_train=X_train,
            y_train=y_train[col],
            target_name=col,
            n_trials=optimization_params['n_trials']
        )
        
        # Train final model
        model_result = train_final_model_all_features(
            X_train=X_train,
            y_train=y_train[col],
            best_params=best_params,
        )
        
        models[col] = model_result
        best_params_dict[col] = best_params
        train_metrics_dict[col] = model_result['train_metrics']
        cv_metrics_dict[col] = cv_metrics
        
        
        # Log hyperparameters & metrics vào ClearML
        logger.report_scalar(title=f"{col} - Train Metrics", series="R²", value=model_result['train_metrics']['r2'], iteration=0)
        logger.report_scalar(title=f"{col} - CV Metrics", series="R²", value=cv_metrics['r2'], iteration=0)
        logger.report_text(f"{col} - Best Params: {best_params}")
        logger.report_scalar(title=f"{col} - Train Metrics", series="RMSE", value=model_result['train_metrics']['rmse'], iteration=0)
        logger.report_scalar(title=f"{col} - CV Metrics", series="RMSE", value=cv_metrics['rmse'], iteration=0)
        logger.report_scalar(title=f"{col} - Train Metrics", series="MAE", value=model_result['train_metrics']['mae'], iteration=0)
        logger.report_scalar(title=f"{col} - CV Metrics", series="MAE", value=cv_metrics['mae'], iteration=0)

    # STEP 2: Return pipeline
    pipeline = {
        "models": models,
        "features": {col: model_info['feature_names'] for col, model_info in models.items()},
        "best_params": best_params_dict,
        "train_metrics": train_metrics_dict,
        "cv_metrics": cv_metrics_dict,
        "feature_selection_method": "ALL FEATURES",
        "pipeline_type": "all_features",
    }
    
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
        
        rf_model = model_info['model']
        
        # Chuẩn bị test data với tất cả features
        X_test_final, y_test_final = prepare_data(X_test, y_test[col])
        
        # Predict và tính metrics
        y_pred_test = rf_model.predict(X_test_final)
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
            
        model = model_info['model']
        
        print(f"\n🎯 {col} - Comprehensive Evaluation:")
        print("-" * 80)
        
        # 1. TRAIN SET EVALUATION
        X_train_final, y_train_final = prepare_data(X_train, y_train[col])
        y_pred_train = model.predict(X_train_final)
        train_metrics = calculate_all_metrics(y_train_final, y_pred_train)
        
        # 2. VALIDATION SET (CV metrics from pipeline)
        cv_metrics = pipeline["cv_metrics"][col]
        
        # 3. TEST SET EVALUATION
        X_test_final, y_test_final = prepare_data(X_test, y_test[col])
        y_pred_test = model.predict(X_test_final)
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
    pipeline = complete_model_pipeline_all_features(
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

    logger.report_text(f"Average Train Metrics: {avg_metrics['train']}")
    logger.report_text(f"Average Validation Metrics: {avg_metrics['validation']}")
    logger.report_text(f"Average Test Metrics: {avg_metrics['test']}")

    # Lưu model vào ClearML
    for target_name, model_info in pipeline['models'].items():
        model_path = f"models_onnx/{target_name}.onnx"
        task.upload_artifact(name=f"{target_name}_onnx_model", artifact_object=model_path)


    # -------------------------
    # Step 9: Export models to ONNX
    # -------------------------
    save_rf_models_to_onnx(pipeline, save_dir="models_onnx/")

    print("\n🎉 Pipeline completed successfully!")
    return pipeline, final_results


# ============================================================================
# 10.onnx
# ============================================================================
def save_rf_models_to_onnx(pipeline, save_dir="models_onnx/"):
    """
    Lưu 5 model RandomForest (y+1 → y+5) sang định dạng ONNX.
    Mỗi model sẽ được lưu thành 1 file .onnx riêng.
    """
    import os
    import joblib
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    os.makedirs(save_dir, exist_ok=True)

    for target_name, model_info in pipeline["models"].items():
        model = model_info["model"]
        feature_names = model_info["feature_names"]

        # Input shape = (None, n_features)
        initial_type = [('float_input', FloatTensorType([None, len(feature_names)]))]

        print(f"💾 Converting {target_name} to ONNX format...")
        try:
            onnx_model = convert_sklearn(model, initial_types=initial_type)
        except Exception as e:
            print(f"❌ Failed to convert {target_name}: {e}")
            continue

        file_path = os.path.join(save_dir, f"{target_name}.onnx")
        with open(file_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        print(f"✅ Saved: {file_path}")

    print("\n🎉 All models saved to ONNX format successfully!")

# ============================================================================
# 11. EXECUTION
# ============================================================================
if __name__ == "__main__":
    set_seed(42)
    task = Task.init(
        project_name="Weather Forecast HCM",
        task_name="LGBM All Features Pipeline",
        task_type=Task.TaskTypes.training,
        output_uri=True
    )
    logger = task.get_logger()

    path = "https://raw.githubusercontent.com/7hufofbun/DSEB--Machine-Learning_Group5/refs/heads/main/data/weather_hcm_daily.csv"
    data = get_data(path)

    # ✅ Sử dụng pipeline với feature selection độc lập
    pipeline, final_results = main_all_features_pipeline(
        data
    )
    # ✅ Lưu 5 model sang ONNX
    save_rf_models_to_onnx(pipeline, save_dir="models_onnx/")
    for target_name in pipeline['models'].keys():
        model_path = f"models_onnx/{target_name}.onnx"
        task.upload_artifact(name=f"{target_name}_onnx_model", artifact_object=model_path)