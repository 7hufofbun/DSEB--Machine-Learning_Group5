import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.feature_selection import VarianceThreshold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')
from clearml import Task, Logger

import random
import os

# Create environment from YAML
# conda env create -f environment.yml

# Activate environment
# conda activate weather_forecast

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
    def __init__(self, threshold=50, var_threshold=0.0, smoothing = 10):
        self.threshold = threshold               
        self.var_threshold = var_threshold       
        self.datetime_cols = ['datetime', 'sunrise', 'sunset']
        self.smoothing = smoothing
        self.label_encoders_ = {}

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

        # Drop variance threshold
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
        # Label encoding
        for col in self.categorical_cols_to_keep_:
            le = LabelEncoder()
            le.fit(temp_df[col].astype(str))
            self.label_encoders_[col] = le
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
        # Label encode categorical data
        for col in self.categorical_cols_to_keep_:
            if col in temp_df.columns:
                col = col.lower().replace(' ', r'_').replace(',', r'_').replace('.', r'_')
                temp_df[col] = self.label_encoders_[col].transform(temp_df[col].astype(str))

        # Add datetime back
        if not datetime_values.empty:
            clean = pd.concat([datetime_values, temp_df], axis=1)
        else:
            clean = temp_df

        return clean



def feature_engineer(X, y):

    df = X.copy()
    df['temp'] = y.values
    
    # === Bước 1: Đảm bảo index là datetime và chuẩn hóa tần suất ===
    if 'datetime' not in df.columns:
        raise ValueError("X must contain 'datetime' column.")
    
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime').sort_index()
    

    # === Bước 2: Loại bỏ biến thừa ===
    cols_to_drop = ['feelslikemax', 'feelslikemin', 'feelslike', 'moonphase', 'visibility']
    df.drop(cols_to_drop, axis=1, errors='ignore', inplace = True)

    # === Bước 3: Tách base features (không phải encoded, không phải datetime) ===
    encoded_fea = ['conditions',	'icon']
    base_fea = [col for col in df.columns if col not in encoded_fea]

    # === Bước 4: Thêm datetime features ===
    df['trend'] = np.arange(len(df))
    df['dayofyear'] = df.index.dayofyear
    df['month'] = df.index.month
    df['weekday'] = df.index.weekday
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    df['quarter'] = df.index.quarter

    df['sin_day'] = np.sin(2 * np.pi * df['dayofyear'] / 365.25)
    df['cos_day'] = np.cos(2 * np.pi * df['dayofyear'] / 365.25)
    df['sin_month'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
    df['cos_month'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)

    # Tính daylighthour nếu có sunrise/sunset
    if 'sunrise' in df.columns and 'sunset' in df.columns:
        df['daylighthour'] = (df['sunset'] - df['sunrise']).dt.total_seconds() / 3600
        base_fea.append('daylighthour')
    
    df['season_dry'] = df['month'].isin([12,1,2,3]).astype(int)
    df['season_transition1'] = df['month'].isin([4,5]).astype(int)
    df['season_wet'] = df['month'].isin([6,7,8,9,10]).astype(int)
    df['season_transition2'] = df['month'].isin([11]).astype(int)


    # === Bước 5: Thêm LAG CỦA TARGET (nhiệt độ) — RẤT QUAN TRỌNG ===
    if 'temp' not in df.columns:
        raise ValueError("X must contain 'temp' column as target.")
    
    for h in [1, 2, 3, 7, 14, 21, 30]:
        df[f'temp_lag_{h}'] = df['temp'].shift(h)
        df[f'temp_d{h}'] = df['temp'].diff(h)

    for j in [3,7, 14, 30]:
        df[f'temp_rolling_mean_{j}'] = df['temp'].shift(1).rolling(j, min_periods = 1).mean()
        df[f'temp_rolling_std_{j}'] = df['temp'].shift(1).rolling(j, min_periods = 1).std()


    # === Bước 6: Tạo lag/diff/rolling cho base_fea (không bao gồm sunrise/sunset) ===
    for slag in [1, 3,4, 5, 6]:
        for col in base_fea:
            if col in ['sunrise', 'sunset']:
                continue
            df[f'{col}_lag_{slag}'] = df[col].shift(slag)
            df[f'{col}_d{slag}'] = df[col].diff(slag)

    for w in [3, 5, 7, 14, 30]:
        for col in base_fea:
            if col in ['sunrise', 'sunset']:
                continue
            # DÙNG .shift(1) để tránh leakage!
            df[f'{col}_rolling_mean_{w}'] = df[col].shift(1).rolling(w, min_periods=1).mean()
            df[f'{col}_rolling_std_{w}'] = df[col].shift(1).rolling(w, min_periods=1).std()

    # === Bước 7: Interaction features (chỉ với lag 1 và rolling 7) ===


    # === Bước 8: Tạo target y+1 đến y+5 từ 'temp' ===
    Y_dict = {}
    for h in [1, 2, 3, 4, 5]:
        Y_dict[f'temp_t+{h}'] = df['temp'].shift(-h)  # dự báo tương lai
    
    Y_df = pd.DataFrame(Y_dict, index=df.index)

    # === Bước 9: Ghép và dropna đồng bộ ===
    combined = pd.concat([df, Y_df], axis=1)
    combined = combined.dropna(axis=0)
    target_cols = [f'temp_t+{h}' for h in [1,2,3,4,5]]

    # === Bước 10: Tách X_final (không có 'temp' và target) và Y_final ===
    X_final = combined.drop(columns= target_cols + base_fea , errors='ignore')
    Y_final = combined[target_cols].reset_index(drop=True)

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


def get_model(best_params):
    """Trả về LightGBM model với tham số đã tune"""
    return LGBMRegressor(
        n_estimators=best_params.get('n_estimators', 400),
        max_depth=best_params.get('max_depth', 5),
        num_leaves=best_params.get('num_leaves', 63),
        min_data_in_leaf=best_params.get('min_data_in_leaf', 20),
        learning_rate=best_params.get('learning_rate', 0.01),
        feature_fraction=best_params.get('feature_fraction', 0.3),
        bagging_fraction=best_params.get('bagging_fraction', 0.7),
        bagging_freq=best_params.get('bagging_freq', 5),
        lambda_l1=best_params.get('lambda_l1', 0.1),
        lambda_l2=best_params.get('lambda_l2', 0.1),
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
def get_model_params(trial,  target_name):
    """Lấy parameters cho từng model type"""
    horizon = int(target_name.split('+')[-1])
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 800),

        # GIẢM SỨC MẠNH CÂY
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'num_leaves': trial.suggest_int('num_leaves', 10, 50),

        # REGULARIZATION TỰ NHIÊN
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 50, 100),

        # LEARNING RATE THẤP
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.03, log=True),

        # SUBSAMPLING MẠNH HƠN
        'feature_fraction': trial.suggest_float('feature_fraction', 0.1, 0.5),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 0.7),
        'bagging_freq': trial.suggest_int('bagging_freq', 2, 7),

        # REGULARIZATION L1 / L2
        'lambda_l1': trial.suggest_float('lambda_l1', 10, 50.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 10, 50.0, log=True),
    }

    return params
def train_final_model_all_features(X_train, y_train, best_params):
    """Train final model trên toàn bộ training data với tất cả features"""
    
    X_train_final, y_train_final = X_train.copy() , y_train.copy()
    val_size = int(len(X_train_final) * 0.1)
    X_tr = X_train_final.iloc[:-val_size]
    y_tr = y_train_final.iloc[:-val_size]
    X_val = X_train_final.iloc[-val_size:]
    y_val = y_train_final.iloc[-val_size:]
    
    model = get_model( best_params)
    
    # Train model
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        callbacks=[
            early_stopping(stopping_rounds=200, verbose=False),
            log_evaluation(0)
        ])
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
def objective_all_features(trial, X_train, y_train, target_name):
    """Objective function cho Optuna với tất cả features và model type"""
    
    params = get_model_params(trial, target_name)
    
    # Sử dụng tất cả features
    X_final, y_final = X_train.copy(), y_train.copy()

    tscv = TimeSeriesSplit(n_splits=4)
    all_metrics = {'mae': [], 'rmse': [], 'r2': [], 'mse': []}

    for train_idx, val_idx in tscv.split(X_final):
        X_tr, X_val = X_final.iloc[train_idx], X_final.iloc[val_idx]
        y_tr, y_val = y_final.iloc[train_idx], y_final.iloc[val_idx]

        model = get_model( params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
            eval_metric='rmse',
            callbacks=[
                early_stopping(stopping_rounds=200, verbose=False),
                log_evaluation(0)  # Không hiển thị log
            ])
            
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
    X_final, y_final = X_train.copy(), y_train.copy()
    
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    study = optuna.create_study(direction='minimize', sampler=sampler, pruner=pruner)


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
            optimization_params = {'n_trials': 1}

        
    X_numeric = X_train.select_dtypes(include=[np.number])
    # STEP 1: Model Training với Tất Cả Features
    models = {}
    best_params_dict = {}
    train_metrics_dict = {}
    cv_metrics_dict = {}
    
    target_columns = [c for c in y_train.columns if c.startswith("temp_t+")]
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
        
        # Kiểm tra overfitting
        train_test_gap = model_result['train_metrics']['r2'] - cv_metrics['r2']

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
    
    test_metrics_dict = {}
    
    # Duyệt qua tất cả targets
    for col, model_info in pipeline["models"].items():
        if model_info is None:
            continue
        
        rf_model = model_info['model']
        
        # Chuẩn bị test data với tất cả features
        X_test_final, y_test_final = X_test.copy(), y_test[col].copy()
        # Predict và tính metrics
        y_pred_test = rf_model.predict(X_test_final)
        test_metrics = calculate_all_metrics(y_test_final, y_pred_test)
        test_metrics_dict[col] = test_metrics
        
    # Trung bình metrics cho tất cả target
    avg_rmse = np.mean([m['rmse'] for m in test_metrics_dict.values()])
    avg_mae = np.mean([m['mae'] for m in test_metrics_dict.values()])
    avg_r2 = np.mean([m['r2'] for m in test_metrics_dict.values()])

    
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
        X_train_final, y_train_final = X_train.copy(), y_train[col].copy()
        y_pred_train = model.predict(X_train_final)
        train_metrics = calculate_all_metrics(y_train_final, y_pred_train)
        
        # 2. VALIDATION SET (CV metrics from pipeline)
        cv_metrics = pipeline["cv_metrics"][col]
        
        # 3. TEST SET EVALUATION
        X_test_final, y_test_final = X_test.copy(), y_test[col].copy()
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


def analyze_overfitting(final_results):
    """Phân tích overfitting và đề xuất cải tiến"""
    print("\n🔍 OVERFITTING ANALYSIS & RECOMMENDATIONS")
    print("=" * 80)
    
    for target, results in final_results.items():
        train_rmse = results['train']['rmse']
        test_rmse = results['test']['rmse']
        gap = train_rmse - test_rmse
        
        print(f"\n{target}:")
        print(f"  Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}, Gap: {gap:.4f}")

# ============================================================================
# 9. MAIN EXECUTION FUNCTION
# ============================================================================

def main_all_features_pipeline(data):
    """Main function sử dụng TẤT CẢ FEATURES cho từng target"""
    data = basic_cleaning(data)
    logger.report_table(
        title="Data Overview After Cleaning",
        series="basic_cleaning",
        table_plot=data.describe().reset_index()
    )
    
    train_x, train_y, test_x, test_y = split_data(data)
    print(f"📊 Train data: {train_x.shape}, Test data: {test_x.shape}")
    logger.report_scalar(
        title="Data Split", series="Train Samples", value=len(train_x), iteration=0
    )
    logger.report_scalar(
        title="Data Split", series="Test Samples", value=len(test_x), iteration=0
    )
    
    preprocessor = Preprocessor(threshold=50, var_threshold=0.0)
    X_train_processed = preprocessor.fit_transform(train_x)
    X_test_processed = preprocessor.transform(test_x)
    print(f"✅ Processed train: {X_train_processed.shape}, test: {X_test_processed.shape}")

    logger.report_scalar(
        title="Preprocessing", series="Features After Preprocessing", 
        value=X_train_processed.shape[1], iteration=0
    )

    X_train_fe, y_train_fe = feature_engineer(X_train_processed, train_y)
    X_test_fe, y_test_fe = feature_engineer(X_test_processed, test_y)
    print(f"✅ Engineered features - Train: {X_train_fe.shape}, Test: {X_test_fe.shape}")

    logger.report_scalar(
        title="Feature Engineering", series="Total Features", 
        value=X_train_fe.shape[1], iteration=0
    )
    logger.report_scalar(
        title="Feature Engineering", series="Training Samples", 
        value=X_train_fe.shape[0], iteration=0
    )
    X_train_fe.to_csv("train_X_after_feature_engineering.csv", index=False)
    y_train_fe.to_csv("train_Y_after_feature_engineering.csv", index=False)
    # TRAINING PIPELINE với tất cả features
    pipeline = complete_model_pipeline_all_features(
        X_train=X_train_fe,
        y_train=y_train_fe,
        optimization_params={'n_trials': 30}
    )
    
    # Đánh giá trên test set (metrics riêng từng target + trung bình)
    test_metrics_dict = evaluate_on_test_set_summary(pipeline, X_test_fe, y_test_fe)
    
    # Comprehensive evaluation và tính metrics trung bình trên tất cả target
    final_results, avg_metrics = comprehensive_final_evaluation_with_avg(
        pipeline, X_train_fe, y_train_fe, X_test_fe, y_test_fe
    )
    
    for iteration_variable, (target_name, metrics) in enumerate(final_results.items()):
        for dataset_type, metric_dict in metrics.items():
            logger.report_scalar(
                title=f"Metrics/{target_name}", series=f"R2_{dataset_type}", 
                value=metric_dict['r2'], iteration=iteration_variable
            )
            logger.report_scalar(
                title=f"Metrics/{target_name}", series=f"RMSE_{dataset_type}", 
                value=metric_dict['rmse'], iteration=iteration_variable
            )
            logger.report_scalar(
                title=f"Metrics/{target_name}", series=f"MAE_{dataset_type}", 
                value=metric_dict['mae'], iteration=iteration_variable
            )

    
    # Log average metrics
    logger.report_scalar(
        title="Average Metrics", series="R2_Train", value=avg_metrics['train']['r2'], iteration=0
    )
    logger.report_scalar(
        title="Average Metrics", series="R2_Test", value=avg_metrics['test']['r2'], iteration=0
    )
    logger.report_scalar(
        title="Average Metrics", series="RMSE_Test", value=avg_metrics['test']['rmse'], iteration=0
    )

    # Update pipeline
    pipeline['test_metrics'] = avg_metrics['test']
    pipeline['final_results'] = final_results
    
    # Overfitting analysis
    analyze_overfitting(final_results)
    
    return pipeline, final_results

# ============================================================================
# 10.onnx
# ============================================================================
def save_models_to_onnx(pipeline, save_dir="models_onnx/"):
    """
    Lưu LightGBM models sang ONNX với workaround cho isinstance bug
    """
    import os
    import sys
    import tempfile
    import numpy as np
    
    print(f"🔍 Python version: {sys.version}")
    
    import onnx
    import lightgbm as lgb
    
    print(f"📦 Package versions:")
    print(f"   - onnx: {onnx.__version__}")
    print(f"   - lightgbm: {lgb.__version__}")
    
    os.makedirs(save_dir, exist_ok=True)
    conversion_results = {"success": 0, "failed": 0}

    for target_name, model_info in pipeline["models"].items():
        model = model_info["model"]
        feature_names = model_info["feature_names"]
        n_features = len(feature_names)
        
        print(f"\n💾 Converting {target_name} to ONNX format...")
        print(f"   Model type: {type(model)}")
        print(f"   Number of features: {n_features}")
        
        try:
            # Get booster
            booster = model.booster_
            
            # Save booster to temporary file
            temp_model_path = tempfile.mktemp(suffix='.txt')
            booster.save_model(temp_model_path)
            print(f"   ✓ Saved booster to temp file")
            
            # Load back as pure Booster (not sklearn wrapper)
            pure_booster = lgb.Booster(model_file=temp_model_path)
            print(f"   ✓ Loaded as pure Booster: {type(pure_booster)}")
            
            # Create test input (dummy data with correct shape)
            test_input = np.random.randn(1, n_features).astype(np.float32)
            print(f"   ✓ Created test input with shape: {test_input.shape}")
            
            # Now convert using hummingbird-ml
            try:
                from hummingbird.ml import convert as hb_convert
                print(f"   Using Hummingbird converter...")
                
                # Convert using Hummingbird with test input
                onnx_model = hb_convert(
                    model, 
                    'onnx',
                    test_input=test_input,  # Add test input here!
                    extra_config={
                        'onnx_target_opset': 12,
                    }
                ).model
                
            except ImportError:
                print(f"   Hummingbird not found, installing...")
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "hummingbird-ml[onnx]"])
                from hummingbird.ml import convert as hb_convert
                
                onnx_model = hb_convert(
                    model, 
                    'onnx',
                    test_input=test_input,  # Add test input here!
                    extra_config={
                        'onnx_target_opset': 12,
                    }
                ).model
            
            file_path = os.path.join(save_dir, f"{target_name}.onnx")
            
            # Save ONNX model
            with open(file_path, 'wb') as f:
                f.write(onnx_model.SerializeToString())
            
            print(f"   ✅ Saved: {file_path}")
            conversion_results["success"] += 1
            
            # Verify
            try:
                onnx_model_check = onnx.load(file_path)
                onnx.checker.check_model(onnx_model_check)
                print(f"   ✓ Model verification passed")
            except Exception as ve:
                print(f"   ⚠️  Verification warning: {ve}")
            
            # Log file size
            file_size = os.path.getsize(file_path) / 1024
            logger.report_scalar(
                title="Model Export", series=f"{target_name}_Size_KB", 
                value=file_size, iteration=0
            )
            
            # Clean up temp file
            try:
                os.remove(temp_model_path)
            except:
                pass
            
        except Exception as e:
            print(f"   ❌ Failed to convert {target_name}")
            print(f"   Error: {str(e)}")
            import traceback
            traceback.print_exc()
            conversion_results["failed"] += 1
            continue

    print(f"\n{'='*80}")
    print(f"🎉 ONNX Export Summary: {conversion_results['success']} successful, {conversion_results['failed']} failed")
    print(f"{'='*80}")
    
    return conversion_results
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
    save_models_to_onnx(pipeline, save_dir="models_onnx/")
    for target_name in pipeline['models'].keys():
        model_path = f"models_onnx/{target_name}.onnx"
        task.upload_artifact(name=f"{target_name}_onnx_model", artifact_object=model_path)

    avg_test_r2 = pipeline['test_metrics']['r2']
    avg_test_rmse = pipeline['test_metrics']['rmse']
    logger.report_scalar(
            title="Final Performance", series="Average_Test_R2", value=avg_test_r2, iteration=0
        )
    logger.report_scalar(
            title="Final Performance", series="Average_Test_RMSE", value=avg_test_rmse, iteration=0
        )