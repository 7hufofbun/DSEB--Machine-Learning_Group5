import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

import os  

def get_data(path):
    """ get data from downloaded datasets and also update real time data"""
    # get old dataset
    data = pd.read_csv(path)
    # get new dataset
    return data

def split_data(data, train_ratio = 0.6, val_ratio = 0.2):
    if 'datetime' in data.columns:
        data = data.sort_values('datetime').reset_index(drop=True)
    n = len(data)
    n_train = int(n*train_ratio)
    n_val = int(n*val_ratio)
    n_test = n - n_train - n_val
    
    # Chia theo thứ tự thời gian
    train_data = data.iloc[:n_train]
    val_data = data.iloc[n_train:n_train + n_val]
    test_data = data.iloc[n_train + n_val:]

    # Divide X and y
    X_train = train_data.drop(['temp'], axis=1, errors='ignore')
    y_train = train_data['temp']

    X_val = val_data.drop(['temp'], axis=1, errors='ignore')
    y_val = val_data['temp']

    X_test = test_data.drop(['temp'], axis=1, errors='ignore')
    y_test = test_data['temp']

    return X_train, y_train, X_val, y_val, X_test, y_test

def basic_cleaning(data):
    data = data.drop_duplicates()
    # Drop unecessary columns
    uni_value = data.nunique()
    col = uni_value[uni_value == 1].index
    data.drop(col, axis=1, inplace=True)
    data.drop(['description'], axis=1, inplace=True)

    # Normalize dtypes
    data['datetime'] = pd.to_datetime(data['datetime'])
    data['sunrise'] = pd.to_datetime(data['sunrise'])
    data['sunset'] = pd.to_datetime(data['sunset'])

    return data

        
def feature_engineer_X(X, y = None,  lags=[1, 2, 3, 7, 14, 30], roll_windows=[3, 7, 14, 30]):
    df = X.copy()
    if y is not None:
        df['temp'] = y
        has_target = True
    else:
        has_target = False
    df = df.sort_values('datetime').reset_index(drop=True)

    # Extract datetime features
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['quarter'] = df['datetime'].dt.quarter
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

    #set datetime into index
    df.set_index('datetime', inplace=True)
    # Create new feature from sunset and sunrise 
    df['sunrise'] = df['sunrise'].dt.time
    df['sunset'] = df['sunset'].dt.time
    df['sunrise_hour'] = df['sunrise'].apply(lambda x: x.hour + x.minute/60 if pd.notnull(x) else np.nan)
    df['sunset_hour'] = df['sunset'].apply(lambda x: x.hour + x.minute/60 if pd.notnull(x) else np.nan)
    df['day_length_hours'] = df['sunset_hour'] - df['sunrise_hour']
    df.drop(['sunrise', 'sunset', 'sunrise_hour', 'sunset_hour'], axis=1, inplace=True)

    # create features for x
    columns = ['tempmax', 'tempmin', 'feelslike', 'dew', 'humidity', 'windspeed', 'windgust', 'sealevelpressure', 'precip', 'solarradiation', 'solarenergy', 'cloudcover']
    # create lag features
    for col in columns:
        if col in df.columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    # create rolling features 
    for col in columns:
        if col in df.columns:
            for window in roll_windows:
                df[f'{col}_roll_mean_{window}'] = df[col].shift(1).rolling(window=window).mean()
                df[f'{col}_roll_std_{window}'] = df[col].shift(1).rolling(window=window).std()

    # create lag and rolling features for temp(y)
    if has_target:
        for lag in lags:
            df[f'temp_lag_{lag}'] = df['temp'].shift(lag)
        for window in roll_windows:
            df[f'temp_roll_mean_{window}'] = df['temp'].shift(1).rolling(window=window).mean()
            df[f'temp_roll_std_{window}'] = df['temp'].shift(1).rolling(window=window).std()

    #create interaction features
    df['temp_humidity_interact'] = df['tempmax'] * df['humidity']
    df['effective_solar'] = df['solarradiation'] / (df['cloudcover'] + 1)
    df['dew_wind_interact'] = df['dew'] * df['windspeedmean']

    if has_target:
        X_processed = df.drop(['temp'], axis=1)
        y_processed = df['temp']
        return X_processed, y_processed
    else:
        return df



class Preprocessing(BaseEstimator, TransformerMixin):
    def __init__(self, lower = 0.05, upper = 0.95, threshold = 50, datetime = 'datetime'):
        self.lower = lower
        self.upper = upper  
        self.threshold = threshold
        self.datetime= datetime
    def fit(self, X, y=None):
        
        df = X.copy()
        
        # identify numerical cols
        self.numeric_cols = df.select_dtypes(include='number').columns.tolist()
        # identify categorical cols
        self.categorical_cols = df.select_dtypes(include='object').columns.tolist()

        # identify  missing to drop
        percentage_missing = df.isnull().sum() * 100 / len(df)
        self.cols_to_keep = percentage_missing[percentage_missing < self.threshold].index.tolist()
        self.numeric_cols_to_keep = [col for col in self.numeric_cols if col in self.cols_to_keep]
        self.categorical_cols_to_keep = [col for col in self.categorical_cols if col in self.cols_to_keep]
        # Compute quantiles from train set
        self.quantiles = {}
        for col in self.numeric_cols_to_keep:
            self.quantiles[col] = (df[col].quantile(self.lower), df[col].quantile(self.upper)) 
        
        # Fit OneHotEncoder once here — to avoid data leakage
        if len(self.categorical_cols_to_keep) > 0:
            self.ohe = OneHotEncoder(sparse_output=False, drop=None, handle_unknown='ignore')
            self.ohe.fit(df[self.categorical_cols_to_keep])
        else:
            self.ohe = None
        
        return self
    def transform(self, X, y=None):
        df = X.copy()

        if self.datetime in df.columns:
            keep_cols = [self.datetime] + [c for c in self.cols_to_keep if c != self.datetime]
            df = df.set_index(self.datetime)
        else:
            keep_cols = self.cols_to_keep
        df = df[keep_cols]

        # using interpolate for numerical columns
        for col in self.numeric_cols_to_keep:
            df[col] = df[col].interpolate(method='time').ffill().bfill()
        
        # clip data by quantile
        for col in self.numeric_cols_to_keep:
            low, high = self.quantiles[col]
            df[col] = df[col].clip(lower=low, upper=high)

        # encoding categorical data
        if self.ohe is not None and len(self.categorical_cols_to_keep) > 0:
            encoded = self.ohe.transform(df[self.categorical_cols_to_keep])
            encoded_df = pd.DataFrame(
                encoded, 
                columns=self.ohe.get_feature_names_out(self.categorical_cols_to_keep),
                index=df.index
            )
            df = pd.concat([df.drop(columns=self.categorical_cols_to_keep), encoded_df], axis=1)
        
        # for lag/rolling feature, causing nan in first n lag/rolling rows -> drop nan
        max_lag = max([int(s.split('_')[-1]) for s in df.columns if '_lag_' in s] or [0])
        max_roll = max([int(s.split('_')[-1]) for s in df.columns if '_roll_' in s] or [0])
        if max(max_lag, max_roll) > 0 and len(df) > max(max_lag, max_roll):
            df = df.iloc[max(max_lag, max_roll):]

        return df

class CorrelationReducer(BaseEstimator, TransformerMixin):
    """
    Removes features that are highly correlated with others
    to reduce redundancy and multicollinearity.
    """
    def __init__(self, threshold=0.9):
        self.threshold = threshold
        self.to_drop_ = None

    def fit(self, X, y=None):
        # Work only with numeric columns
        numeric_df = X.select_dtypes(include=[np.number])

        # Compute correlation matrix (absolute values)
        corr_matrix = numeric_df.corr().abs()

        # Upper triangle to avoid duplicate pairs
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Identify columns to drop
        self.to_drop_ = [col for col in upper.columns if any(upper[col] > self.threshold)]
        print(f"[CorrelationReducer] {len(self.to_drop_)} features dropped (threshold={self.threshold})")
        return self

    def transform(self, X, y=None):
        if self.to_drop_ is None:
            raise RuntimeError("CorrelationReducer must be fitted before calling transform().")
        X_reduced = X.drop(columns=self.to_drop_, errors='ignore')
        return X_reduced


class SafeAligner(BaseEstimator, TransformerMixin):
    """Ensure X and y have same time index after lag/rolling drops"""
    def fit(self, X, y=None):
        if y is not None:
            self.valid_index_ = X.index.intersection(y.index)
        else:
            self.valid_index_ = X.index
        return self

    def transform(self, X, y=None):
        X = X.loc[self.valid_index_]
        if y is not None:
            y = y.loc[self.valid_index_]
            return X, y
        return X
    
class PipelineWithY(Pipeline):
    def fit(self, X, y=None, **fit_params):
        Xt, yt = X, y
        for name, step in self.steps:
            if hasattr(step, "fit_transform"):
                out = step.fit_transform(Xt, yt)
            elif hasattr(step, "fit"):
                step.fit(Xt, yt)
                out = Xt
            else:
                out = Xt

            # 🔹 Kiểm tra output: có phải tuple (X, y) không?
            if isinstance(out, tuple) and len(out) == 2:
                Xt, yt = out
            else:
                Xt = out
        return self

    def transform(self, X, y=None):
        Xt, yt = X, y
        for name, step in self.steps:
            if hasattr(step, "transform"):
                out = step.transform(Xt, yt)
                if isinstance(out, tuple) and len(out) == 2:
                    Xt, yt = out
                else:
                    Xt = out
        return Xt, yt

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X, y)






    
def main():
    # Load data
    path = "https://raw.githubusercontent.com/7hufofbun/DSEB--Machine-Learning_Group5/refs/heads/main/data/weather_hcm_daily.csv"
    data = get_data(path)
    
    data = basic_cleaning(data)
    

    #  Gọi hàm extract_feature đã viết sẵn
    train_x, train_y, val_x, val_y, test_x, test_y = split_data(data)

    # Create feature outside pipeline
    train_x, train_y = feature_engineer_X(train_x, train_y)
    val_x, val_y = feature_engineer_X(val_x, val_y)
    test_x, test_y = feature_engineer_X(test_x, test_y)

    # Run Pipeline model

    pipe = PipelineWithY([
        ('num_pre', Preprocessing()),
        ('reduce_corr', CorrelationReducer(threshold=0.9)),
        ('align', SafeAligner())
    ])
    train_x, train_y = pipe.fit_transform(train_x, train_y)

    # 🔹 Print column info
    print("\n===== FEATURE INFORMATION =====")
    print(f"Total number of features after preprocessing & correlation reduction: {train_x.shape[1]}")
    print("\nColumn names:")
    print(train_x.columns.tolist())
    print("=================================")

    # 🔹 Optional: check dropped columns
    reducer_step = dict(pipe.named_steps).get('reduce_corr')
    if reducer_step and reducer_step.to_drop_:
        print(f"\nDropped {len(reducer_step.to_drop_)} correlated features:")
        print(reducer_step.to_drop_)

    #  In ra thông tin cơ bản
    print("Train X shape:", train_x.shape)
    print("Train y shape:", train_y.shape)
    print("\nTrain X head:")
    print(train_x.head())
    print("\nTrain y head:")
    print(train_y.head())
    print(train_x.isnull().sum())
    # Export 
    os.makedirs('outputs', exist_ok=True)
    train_x.to_csv('outputs/X_features.csv', index=True)
    train_y.to_csv('outputs/Y_target.csv', index=True)
    print("\nFeature engineering complete. Files exported to outputs/")

if __name__ == "__main__":
    main()
