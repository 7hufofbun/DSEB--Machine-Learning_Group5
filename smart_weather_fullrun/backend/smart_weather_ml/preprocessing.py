
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

def basic_cleaning(data: pd.DataFrame) -> pd.DataFrame:
    data = data.drop_duplicates()
    for col in ["datetime", "sunrise", "sunset"]:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors="coerce")
    data.drop(["description", "name", "address", "resolvedAddress", "latitude", "longitude", "source"], axis=1, inplace=True, errors="ignore")
    return data

def split_data(data: pd.DataFrame):
    n = len(data)
    n_train = int(n * 0.85)
    train_data = data.iloc[:n_train]
    test_data  = data.iloc[n_train:]
    X_train = train_data.drop(["temp"], axis=1)
    y_train = train_data["temp"]
    X_test  = test_data.drop(["temp"], axis=1)
    y_test  = test_data["temp"]
    return X_train, y_train, X_test, y_test

class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=50, var_threshold=0.0, smoothing=10):
        self.threshold = threshold
        self.var_threshold = var_threshold
        self.datetime_cols = ["datetime", "sunrise", "sunset"]
        self.smoothing = smoothing
        self.label_encoders_ = {}

    def fit(self, X, y=None):
        df = X.copy()
        temp_df = df.drop(columns=self.datetime_cols, errors="ignore")
        self.numeric_cols_ = temp_df.select_dtypes(include="number").columns.tolist()
        self.categorical_cols_ = temp_df.select_dtypes(include="object").columns.tolist()

        unique_value = temp_df[self.categorical_cols_].nunique() if self.categorical_cols_ else pd.Series(dtype=int)
        self.cat_cols_to_drop_ = unique_value[unique_value == 1].index.tolist()

        percentage_missing = temp_df.isnull().sum() * 100 / len(temp_df)
        self.missing_cols_to_drop_ = percentage_missing[percentage_missing > self.threshold].index.tolist()

        self.numeric_cols_to_keep_ = [c for c in self.numeric_cols_ if c not in self.missing_cols_to_drop_]
        self.categorical_cols_to_keep_ = [
            c for c in self.categorical_cols_ if c not in self.cat_cols_to_drop_ and c not in self.missing_cols_to_drop_
        ]

        if len(self.numeric_cols_to_keep_) > 0:
            selector = VarianceThreshold(threshold=self.var_threshold)
            selector.fit(temp_df[self.numeric_cols_to_keep_].fillna(0))
            self.numeric_cols_to_keep_ = [
                self.numeric_cols_to_keep_[i] for i, keep in enumerate(selector.get_support()) if keep
            ]
            variances = temp_df[self.numeric_cols_to_keep_].var()
            var_groups = {}
            for col, var in variances.items():
                var = round(var, 6)
                var_groups.setdefault(var, []).append(col)
            cols_to_remove = []
            for _, cols in var_groups.items():
                if len(cols) > 1:
                    cols_to_remove.extend(cols[1:])
            self.numeric_cols_to_keep_ = [c for c in self.numeric_cols_to_keep_ if c not in cols_to_remove]

        for col in self.categorical_cols_to_keep_:
            le = LabelEncoder()
            le.fit(temp_df[col].astype(str))
            self.label_encoders_[col] = le

        self.keep_cols_ = self.numeric_cols_to_keep_ + self.categorical_cols_to_keep_
        return self

    def transform(self, X, y=None):
        df = X.copy()
        datetime_values = df[self.datetime_cols].copy() if all(col in df.columns for col in self.datetime_cols) else pd.DataFrame()
        temp_df = df.drop(columns=self.datetime_cols, errors="ignore")
        keep_cols = [c for c in self.keep_cols_ if c in temp_df.columns]
        temp_df = temp_df[keep_cols]

        for col in self.numeric_cols_to_keep_:
            if col in temp_df.columns:
                temp_df[col] = temp_df[col].interpolate(method="linear").ffill().bfill()

        for col in self.categorical_cols_to_keep_:
            if col in temp_df.columns:
                temp_df[col] = self.label_encoders_[col].transform(temp_df[col].astype(str))

        if not datetime_values.empty:
            clean = pd.concat([datetime_values, temp_df], axis=1)
        else:
            clean = temp_df
        return clean
