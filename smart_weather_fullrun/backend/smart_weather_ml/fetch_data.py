import os
from datetime import datetime, timezone, timedelta
import pandas as pd
import requests
from io import StringIO
import time

# === Cấu hình ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DAILY_PATH = os.path.join(DATA_DIR, "weather_hcm_daily.csv")
HOURLY_PATH = os.path.join(DATA_DIR, "weather_hcm_hourly.csv")

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
# === Tải dữ liệu từ API ===
def download_weather_data(location, start_date, end_date=None, api_key="", unit="metric", content_type="csv", include="days", features=None, save_path=None, delay=1.0):
    time.sleep(delay)
    base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
    url = f"{base_url}/{location}/{start_date}" + (f"/{end_date}" if end_date else "")
    params = {
        "unitGroup": unit,
        "include": include,
        "key": api_key,
        "contentType": content_type
    }
    if features:
        params["elements"] = ",".join(features)

    print(f"🔗 Gọi API: {url}")
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

# === Ghi dữ liệu vào file CSV ===
def append_directly(new_data, csv_file_path, reorder_columns=True):
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

    if isinstance(new_data, str):
        new_df = pd.read_csv(new_data)
    else:
        new_df = new_data

    file_exists = os.path.exists(csv_file_path)
    if file_exists:
        old_df = pd.read_csv(csv_file_path)
        old_columns = old_df.columns.tolist()
        new_df.columns = new_df.columns.str.strip()
        old_columns_clean = [col.strip() for col in old_columns]

        if set(new_df.columns) != set(old_columns_clean):
            column_mapping = {}
            for new_col in new_df.columns:
                new_col_clean = new_col.strip()
                for old_col in old_columns:
                    if old_col.strip() == new_col_clean:
                        column_mapping[new_col] = old_col
                        break
                if new_col not in column_mapping:
                    column_mapping[new_col] = new_col
            new_df = new_df.rename(columns=column_mapping)

        if reorder_columns:
            common_columns = [col for col in old_columns if col in new_df.columns]
            extra_in_new = [col for col in new_df.columns if col not in old_columns]
            final_columns = common_columns + extra_in_new
            new_df = new_df.reindex(columns=final_columns)

    header = not file_exists
    new_df.to_csv(csv_file_path, mode='a', index=False, header=header)

# === Chạy chính ===
if __name__ == "__main__":
    VN_TZ = timezone(timedelta(hours=7))
    today = datetime.now(VN_TZ).date()

    try:
        # === DAILY ===
        daily = download_weather_data(
            location=LOCATION,
            start_date=today.strftime("%Y-%m-%d"),
            end_date=today.strftime("%Y-%m-%d"),
            api_key=API_KEY,
            include="days",
            features=DAILY_FEATURES
        )
        if not daily.empty:
            append_directly(daily, DAILY_PATH)

        # === HOURLY ===
        hourly = download_weather_data(
            location=LOCATION,
            start_date=today.strftime("%Y-%m-%d"),
            end_date=today.strftime("%Y-%m-%d"),
            api_key=API_KEY,
            include="hours",
            features=HOURLY_FEATURES
        )
        if not hourly.empty:
            append_directly(hourly, HOURLY_PATH)

    except Exception as e:
        print(f"❌ Lỗi khi tải dữ liệu: {e}")
