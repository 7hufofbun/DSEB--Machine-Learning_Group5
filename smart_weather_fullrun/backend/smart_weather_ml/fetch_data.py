import os
from datetime import datetime, timedelta, timezone
import pandas as pd
import requests
import pandas as pd
from io import StringIO
import time

BASE_DIR = os.path.dirname(__file__)
DAILY_PATH =  os.environ.get(
    "WEATHER_DAILY_CSV", 
    os.path.join(BASE_DIR, "data", "weather_hcm_daily.csv")
)
HOURLY_PATH = os.environ.get(
    "WEATHER_HOURLY_CSV", 
    os.path.join(BASE_DIR, "data", "weather_hcm_hourly.csv")
)
if not os.path.exists(DAILY_PATH) and os.path.exists("/mnt/data/weather_hcm_daily.csv"):
    DAILY_PATH = "/mnt/data/weather_hcm_daily.csv"
if not os.path.exists(HOURLY_PATH) and os.path.exists("/mnt/data/weather_hcm_hourly.csv"):
    HOURLY_PATH = "/mnt/data/weather_hcm_hourly.csv"

LOCATION = "Ho Chi Minh City, Vietnam"
API_KEY = os.getenv("WEATHER_API_KEY", "")
DAILY_FEATURES = [
    'name','address','resolvedAddress','latitude','longitude','datetime',
    'tempmax','tempmin','temp','feelslikemax','feelslikemin','feelslike',
    'dew','humidity','precip','precipprob','precipcover','preciptype',
    'windgust','windspeed','windspeedmax','windspeedmean','windspeedmin',
    'winddir','pressure','cloudcover','visibility','solarradiation','solarenergy',
    'uvindex','severerisk','sunrise','sunset','moonphase','conditions',
    'description','icon','source'
]
HOURLY_FEATURES = ['name','address','resolvedAddress','latitude','longitude',
                'datetime','temp','feelslike','dew','humidity','precip','precipprob','preciptype',
                'windgust','windspeed','winddir','pressure','cloudcover','visibility','solarradiation','solarenergy',
                'uvindex','severerisk','conditions','icon','source']

def download_weather_data(location: str,start_date: str,end_date: str = None,api_key: str = "",unit: str = "metric",content_type: str = "csv",include: str = "days",features: list = None, save_path: str = None,delay: float = 1.0):
    time.sleep(delay)
    base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
    if end_date:
        url = f"{base_url}/{location}/{start_date}/{end_date}"
    else:
        url = f"{base_url}/{location}/{start_date}"
    params = {
        "unitGroup": unit,
        "include": include,
        "key": api_key,
        "contentType": content_type
    }
    if features:
        params["elements"] = ",".join(features)
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"Lỗi {response.status_code}: {response.text}")
    if content_type == "csv":
        df = pd.read_csv(StringIO(response.text))
        if save_path:
            df.to_csv(save_path, index=False, encoding='utf-8')
        return df
    else:
        data = response.json()
        if save_path:
            import json
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        return data

def append_directly(new_data, csv_file_path, reorder_columns=True):
    if isinstance(new_data, str):
        new_df = pd.read_csv(new_data)
    else:
        new_df = new_data
    file_exists = os.path.exists(csv_file_path)
    if file_exists:
        old_df = pd.read_csv(csv_file_path)
        old_columns = old_df.columns.tolist()
        new_columns = new_df.columns.tolist()
        new_df.columns = new_df.columns.str.strip()
        old_columns_clean = [col.strip() for col in old_columns]
        if set(new_df.columns) != set(old_columns_clean):
            column_mapping = {}
            for new_col in new_df.columns:
                new_col_clean = new_col.strip()
                # Tìm cột tương ứng trong file gốc
                for old_col in old_columns:
                    if old_col.strip() == new_col_clean:
                        column_mapping[new_col] = old_col
                        break
                # Nếu không tìm thấy, giữ nguyên tên
                if new_col not in column_mapping:
                    column_mapping[new_col] = new_col
            
            new_df = new_df.rename(columns=column_mapping)
        if reorder_columns:
            common_columns = [col for col in old_columns if col in new_df.columns]
            extra_in_new = [col for col in new_df.columns if col not in old_columns]
            missing_in_new = [col for col in old_columns if col not in new_df.columns]
            # Sắp xếp cột file mới theo thứ tự file gốc + thêm cột mới ở cuối
            final_columns = common_columns + extra_in_new
            new_df = new_df.reindex(columns=final_columns)
            
    header = not file_exists
    new_df.to_csv(csv_file_path, mode='a', index=False, header=header)
def get_last_date(csv_file_path):
    """Trả về ngày cuối cùng trong CSV, hoặc None nếu chưa có dữ liệu"""
    if not os.path.exists(csv_file_path):
        return None
    df = pd.read_csv(csv_file_path, usecols=['datetime'])
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    return df['datetime'].max().date() if not df.empty else None

if __name__ == "__main__":
    # === DAILY DATA ===
    last_date = get_last_date(DAILY_PATH)
    start_date = (last_date + timedelta(days=1)) if last_date else (datetime.utcnow().date() - timedelta(days=30))
    end_date = datetime.utcnow().date() - timedelta(days=1)
    daily = download_weather_data(
        location=LOCATION,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        api_key=API_KEY,
        include="days",
        features=DAILY_FEATURES
    )
    if not daily.empty:
        append_directly(daily, DAILY_PATH)

    # === HOURLY DATA ===
    if HOURLY_PATH:
        last_day_hour = get_last_date(HOURLY_PATH)
        start_day_hour = (last_day_hour + timedelta(days=1)) if last_day_hour else (datetime.utcnow().date() - timedelta(days=30))
        end_day_hour = datetime.utcnow().date() - timedelta(days=1)
        hourly = download_weather_data(
            location=LOCATION,
            start_date=start_day_hour.strftime("%Y-%m-%d"),
            end_date=end_day_hour.strftime("%Y-%m-%d"),
            api_key=API_KEY,
            include="hours",
            features=HOURLY_FEATURES
        )

        if not hourly.empty:
            append_directly(hourly, HOURLY_PATH)
