
import numpy as np
import pandas as pd

def feature_engineer(X: pd.DataFrame, y: pd.Series):
    df = X.copy()
    df['temp'] = y.values
    
    # === Bước 1: Đảm bảo index là datetime và chuẩn hóa tần suất ===
    if 'datetime' not in df.columns:
        raise ValueError("X must contain 'datetime' column.")
    
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime').sort_index()
    

    # === Bước 2: Loại bỏ biến thừa ===
    cols_to_drop = ['feelslikemax', 'feelslikemin',  'moonphase', 'visibility']
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
    
    for h in [1, 2, 7, 14]:
        df[f'temp_lag_{h}'] = df['temp'].shift(h)
        df[f'temp_d{h}'] = df['temp'].diff(h)

    for j in [3,7, 14, 30]:
        df[f'temp_rolling_mean_{j}'] = df['temp'].shift(1).rolling(j, min_periods = 1).mean()
        df[f'temp_rolling_std_{j}'] = df['temp'].shift(1).rolling(j, min_periods = 1).std()


    # === Bước 6: Tạo lag/diff/rolling cho base_fea (không bao gồm sunrise/sunset) ===
    for slag in [1, 3, 4, 5]:
        for col in base_fea:
            if col in ['sunrise', 'sunset']:
                continue
            df[f'{col}_lag_{slag}'] = df[col].shift(slag)
            df[f'{col}_d{slag}'] = df[col].diff(slag)

    for w in [3,  7, 14, 21]:
        for col in base_fea:
            if col in ['sunrise', 'sunset']:
                continue
            # DÙNG .shift(1) để tránh leakage!
            df[f'{col}_rolling_mean_{w}'] = df[col].shift(1).rolling(w, min_periods=1).mean()
            df[f'{col}_rolling_std_{w}'] = df[col].shift(1).rolling(w, min_periods=1).std()

    # df['humid_temp'] = df['humidity_rolling_std_3']*df['temp_rolling_std_3']
    df['sea_cloud'] = df['sealevelpressure_rolling_std_3']*df['cloudcover_rolling_std_3']

    # === Bước 8: Tạo target y+1 đến y+5 từ 'temp' ===
    Y_dict = {}
    for h in [1, 2, 3, 4, 5]:
        Y_dict[f'temp_t+{h}'] = df['temp'].shift(-h)  # dự báo tương lai
    
    Y_df = pd.DataFrame(Y_dict, index=df.index)

    # === Bước 9: Ghép và dropna đồng bộ ===
    combined = pd.concat([df, Y_df], axis=1)
    combined = combined.dropna(axis=0)
    target_cols = [f'temp_t+{h}' for h in [1,2,3,4,5]]
    combined = combined.reset_index(drop=True)

    # === Bước 10: Tách X_final (không có 'temp' và target) và Y_final ===
    X_final = combined.drop(columns= target_cols + base_fea , errors='ignore')
    Y_final = combined[target_cols].reset_index(drop=True)

    return X_final, Y_final