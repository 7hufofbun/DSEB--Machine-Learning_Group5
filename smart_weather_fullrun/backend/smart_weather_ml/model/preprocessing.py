
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

def basic_cleaning(data: pd.DataFrame) -> pd.DataFrame:
    data = data.drop_duplicates()
    for col in ["datetime", "sunrise", "sunset"]:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors="coerce")
    data.drop(['description','name','address','resolvedAddress','latitude','longitude', 'source'], axis=1, inplace=True, errors="ignore")
    return data

def split_data(data: pd.DataFrame):
    n = len(data)
    n_train = int(n * 0.8)
    train_data = data.iloc[:n_train]
    test_data  = data.iloc[n_train:]
    X_train = train_data.drop(["temp"], axis=1)
    y_train = train_data["temp"]
    X_test  = test_data.drop(["temp"], axis=1)
    y_test  = test_data["temp"]
    return X_train, y_train, X_test, y_test

class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=50, var_threshold=0.0):
        self.threshold = threshold               
        self.var_threshold = var_threshold       
        self.datetime_cols = ['datetime', 'sunrise', 'sunset']
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
        # Fit OneHotEncoder on selected categorical columns
        if len(self.categorical_cols_to_keep_) > 0:
            self.ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            self.ohe.fit(temp_df[self.categorical_cols_to_keep_])
            self.ohe_cols_ =  [col.lower().replace(' ', r'_').replace(',', r'_').replace('.', r'_').replace('-', r'_') for col in list(self.ohe.get_feature_names_out(self.categorical_cols_to_keep_))]
        else:
            self.ohe = None
            self.ohe_cols_ = []
        # keep cols finally
        self.keep_cols_ = self.numeric_cols_to_keep_ + self.categorical_cols_to_keep_

        return self

    def transform(self, X, y=None):
        df = X.copy()

        # Preserve any datetime-like columns that are available so downstream feature engineering keeps temporal context
        datetime_present = [col for col in self.datetime_cols if col in df.columns]
        datetime_values = df[datetime_present].copy()

        temp_df = df.drop(columns=self.datetime_cols, errors='ignore')

        keep_cols = [c for c in self.numeric_cols_to_keep_ if c in temp_df.columns]
        temp_df = temp_df[keep_cols]


        # Impute numerical data
        for col in self.numeric_cols_to_keep_:
            if col in temp_df.columns:
                temp_df[col] = temp_df[col].interpolate(method='linear').ffill().bfill()

        # One-hot encode the kept categorical columns
        valid_cat_cols = [col for col in self.categorical_cols_to_keep_ if col in temp_df.columns]

        # One-hot encode categorical columns
        valid_cat_cols = [col for col in self.categorical_cols_to_keep_ if col in df.columns]
        if self.ohe is not None and valid_cat_cols:
            categorical_raw = df[valid_cat_cols].astype(str)
            categorical_data = pd.DataFrame(
                self.ohe.transform(categorical_raw),
                columns=self.ohe_cols_,
                index=df.index
            )
        else:
            categorical_data = pd.DataFrame(index=df.index)


        # Concatenate all processed data
        clean = pd.concat([datetime_values, temp_df, categorical_data], axis=1)

        return clean


