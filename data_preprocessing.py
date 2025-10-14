import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

import os  

def get_data(path):
    """ get data from downloaded datasets and also update real time data"""
    # get old dataset
    data = pd.read_csv(path)
    # get new dataset
    return data
def clean_data(data):
    # delete unnecessary columns
    uni_value = data.nunique()
    col = uni_value[uni_value == 1].index
    data.drop(col, axis=1, inplace=True)
    data.drop(['description', 'sunrise', 'sunset'], axis=1, inplace=True)

    # normalize data dtypes
    data['datetime'] = pd.to_datetime(data['datetime'])


    # handle missing data in time series
    percentage_missing = data.isnull().sum() * 100 / len(data)
    cols = percentage_missing[percentage_missing > 0.5].index
    data.drop(cols, axis=1, inplace=True)

    # using interpolate
    # Numerical columns
    num_cols = data.select_dtypes(include=['float64', 'int64']).columns
    if len(num_cols) > 0:
        data_temp = data.set_index('datetime')
        data_temp[num_cols] = data_temp[num_cols].interpolate(method='time', limit_direction='both')
        data = data_temp.reset_index()
    # Categorical columns
    cat_cols = data.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        data[cat_cols] = data[cat_cols].ffill().bfill()

    # drop duplicates
    data_temp = data_temp.reset_index()
    data.drop_duplicates(inplace=True)
    
    return data

def winsorize_by_quantile(data, columns=None, lower_q=0.05, upper_q=0.95):
    # winsorize by quantile: convert outliers value into min/max by using clip
    dfw = clean_data(data)
    if columns is None:
        columns = dfw.select_dtypes(include='number').columns
    for col in columns:
        low, high = dfw[col].quantile(lower_q), dfw[col].quantile(upper_q)
        dfw[col] = dfw[col].clip(lower=low, upper=high)
    return dfw


def basic_transform(data):
    """
    encoding categorical data"""
    df = winsorize_by_quantile(data)
    # encoding vars
    categorical_cols = ['icon', 'conditions']
    ohe = OneHotEncoder(sparse_output=False, drop=None) 

    # Fit và transform
    encoded = ohe.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(categorical_cols))

    data = pd.concat([data.drop(columns=categorical_cols), encoded_df], axis=1)

    return data


def extract_datetime_features(df):
    """
    Extract components from datetime for seasonality.
    - Year, month, day, day of week, quarter.
    This helps capture annual, monthly, and weekly patterns in temperature.
    """
    df_copy = df.copy()
    df_copy['year'] = df_copy['datetime'].dt.year
    df_copy['month'] = df_copy['datetime'].dt.month
    df_copy['day'] = df_copy['datetime'].dt.day
    df_copy['dayofweek'] = df_copy['datetime'].dt.dayofweek
    df_copy['quarter'] = df_copy['datetime'].dt.quarter
    df_copy['is_weekend'] = df_copy['dayofweek'].isin([5, 6]).astype(int)
    return df_copy

def create_lag_features(df, columns, lags=[1, 2, 3, 7, 14, 30]):
    """
    Create lag features for specified columns.
    - Lags: Past values shifted by 1,2,3,7,14,30 days to capture short-term and longer-term dependencies.
    - Uses pd.concat to avoid PerformanceWarning.
    """
    df_copy = df.copy()
    new_columns = {}
    for col in columns:
        for lag in lags:
            new_columns[f'{col}_lag_{lag}'] = df_copy[col].shift(lag)
    lag_df = pd.DataFrame(new_columns, index=df_copy.index)
    return pd.concat([df_copy, lag_df], axis=1)

def create_rolling_features(df, columns, windows=[3, 7, 14, 30]):
    """
    Create rolling window features.
    - Mean and std over windows of 3,7,14,30 days, shifted by 1 to avoid data leakage.
    - Uses pd.concat to avoid PerformanceWarning.
    """
    df_copy = df.copy()
    new_columns = {}
    for col in columns:
        for window in windows:
            new_columns[f'{col}_roll_mean_{window}'] = df_copy[col].rolling(window=window).mean().shift(1)
            new_columns[f'{col}_roll_std_{window}'] = df_copy[col].rolling(window=window).std().shift(1)
    rolling_df = pd.DataFrame(new_columns, index=df_copy.index)
    return pd.concat([df_copy, rolling_df], axis=1)

def create_interactive_features(df):
    """
    Optional: Create interaction features.
    - Based on insights from data understanding, e.g., temp * humidity, solarradiation / (cloudcover + 1).
    - Uses copy to avoid fragmentation.
    """
    df_copy = df.copy()
    df_copy['temp_humidity_interact'] = df_copy['temp'] * df_copy['humidity']
    df_copy['effective_solar'] = df_copy['solarradiation'] / (df_copy['cloudcover'] + 1)
    df_copy['dew_wind_interact'] = df_copy['dew'] * df_copy['windspeedmean']
    return df_copy

def select_features_with_rf_xgb_plot(X, y,max_estimators=300, cumulative_threshold=0.91):
    """
    Kết hợp RandomForest & XGBoost để chọn feature + vẽ biểu đồ importance.
    """
    # Lọc giá trị hợp lệ
    valid_idx = y.notna()
    X, y = X.loc[valid_idx], y.loc[valid_idx]

    # Train RandomForest
    rf = RandomForestRegressor(n_estimators=max_estimators,random_state=42,n_jobs=-1)
    rf.fit(X, y)
    rf_imp = pd.Series(rf.feature_importances_, index=X.columns)

    #  Train XGBoost
    xgb = XGBRegressor(
        n_estimators=max_estimators,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        eval_metric='rmse'
    )
    xgb.fit(X, y)
    xgb_imp = pd.Series(xgb.feature_importances_, index=X.columns)

    #  Kết hợp RF + XGB
    combined_imp = (rf_imp + xgb_imp) / 2
    combined_imp = combined_imp.sort_values(ascending=False)

    #  Tính cumulative importance
    cum_imp = combined_imp.cumsum() / combined_imp.sum()

    # Chọn feature cho đến khi cumulative importance đạt ngưỡng
    selected_features = cum_imp[cum_imp <= cumulative_threshold].index
    threshold = combined_imp[selected_features].min()

    return X[selected_features], combined_imp, threshold



def extract_feature(data):
    data = data.sort_values('datetime').reset_index(drop=True)

    data = basic_transform(data)
    data = extract_datetime_features(data)
    data.set_index('datetime', inplace=True)
    data.sort_index(inplace=True)

    key_columns = ['temp', 'tempmax', 'tempmin', 'feelslike', 'humidity', 'dew', 'solarradiation', 'cloudcover', 'windspeedmean']
    encoded_cols = [col for col in data.columns if col.startswith(('icon_', 'conditions_'))]
    key_columns.extend(encoded_cols)
    data = create_lag_features(data, key_columns)
    data = create_rolling_features(data, key_columns)
    data = create_interactive_features(data)
    
    data = data.dropna().reset_index()

    X = data.drop(['temp', 'datetime'], axis = 1, errors='ignore')
    y = data['temp']
    
    X_selected, importance, threshold = select_features_with_rf_xgb_plot(X, y)
    X = X[X_selected.columns]
    def remove_highly_correlated_features(X, importance, threshold=0.9):
        """
        Xóa bớt feature tương quan cao, GIỮ lại feature có importance cao hơn.
        """
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Lấy toàn bộ cặp feature có correlation > threshold
        high_corr_pairs = (
            upper.stack()
            .reset_index()
            .rename(columns={0: "corr"})
            .query("corr > @threshold")
        )

        to_drop = set()

        for _, (feat1, feat2, corr_val) in high_corr_pairs.iterrows():
            # Bỏ qua nếu 1 trong 2 đã bị loại
            if feat1 in to_drop or feat2 in to_drop:
                continue

            imp1 = importance.get(feat1, 0)
            imp2 = importance.get(feat2, 0)

            # GIỮ feature có importance cao hơn
            if imp1 < imp2:
                to_drop.add(feat1)
            else:
                to_drop.add(feat2)

        X_reduced = X.drop(columns=list(to_drop), errors="ignore")

        print(f" Đã loại {len(to_drop)} feature tương quan cao:")
        print(list(to_drop))
        print(f" Còn lại {X_reduced.shape[1]} features sau khi lọc.")
        
        return X_reduced
    X = remove_highly_correlated_features(X, importance)

    X.dropna(inplace=True)
    y = y.loc[X.index]  # giữ đồng bộ index
    return X, y

def main():
    #  Load data
    path = "https://raw.githubusercontent.com/7hufofbun/DSEB--Machine-Learning_Group5/refs/heads/main/data/weather_hcm_daily.csv"
    data = pd.read_csv(path)
    
    #  Gọi hàm extract_feature đã viết sẵn
    X, y = extract_feature(data)
    
    #  In ra thông tin cơ bản
    print("X shape:", X.shape)
    print("Y shape:", y.shape)
    print("\nX head:")
    print(X.head())
    print("\nY head:")
    print(y.head())
    
    #  Export nếu muốn
    os.makedirs('outputs', exist_ok=True)
    X.to_csv('outputs/X_features.csv', index=True)
    y.to_csv('outputs/Y_target.csv', index=True)
    print("\nFeature engineering complete. Files exported to outputs/")

if __name__ == "__main__":
    main()
