import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

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
    data.drop(['description'], axis=1, inplace=True)

    # normalize data dtypes
    data['datetime'] = pd.to_datetime(data['datetime'])
    data['sunrise'] = pd.to_datetime(data['sunrise'])
    data['sunset'] = pd.to_datetime(data['sunset'])

    # handle missing data
    percentage_missing = data.isnull().sum() * 100 / len(data)
    cols = percentage_missing[percentage_missing > 0.5].index
    data.drop(cols, axis=1, inplace=True)

    num_cols = data.select_dtypes(include=['float64', 'int64']).columns
    if len(num_cols) > 0:
        num_imputer = SimpleImputer(strategy='median')
        data[num_cols] = num_imputer.fit_transform(data[num_cols])
    
    # Categorical columns
    cat_cols = data.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        data[cat_cols] = cat_imputer.fit_transform(data[cat_cols])

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
    """ contains 
    encoding categorical data,
    feature selection"""
    df = winsorize_by_quantile(data)
    # encoding vars
    categorical_cols = ['icon', 'conditions']
    ohe = OneHotEncoder(sparse_output=False, drop=None) 

    # Fit và transform
    encoded = ohe.fit_transform(data[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(categorical_cols))

    data = pd.concat([data.drop(columns=categorical_cols), encoded_df], axis=1)

    # feature selection
    numerical = data.select_dtypes('number').columns.tolist()
    corr_temp = data[numerical].corr()['temp'].sort_values(ascending=True)

    weak_relationship = corr_temp[abs(corr_temp) < 0.1].index.tolist()
    data.drop(weak_relationship, axis=1, inplace=True)

    return data

data = get_data(r"https://raw.githubusercontent.com/7hufofbun/DSEB--Machine-Learning_Group5/refs/heads/main/data/weather_hcm_daily.csv")
data = basic_transform(data)
print(data.head())
