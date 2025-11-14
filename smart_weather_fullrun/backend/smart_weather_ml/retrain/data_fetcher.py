"""
Data fetcher for Visual Crossing Weather API
Location: backend/smart_weather_ml/retrain/data_fetcher.py
"""

import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Optional
import os


class VisualCrossingDataFetcher:
    """Fetch weather data from Visual Crossing API"""
    
    def __init__(self, api_key: str, location: str = "Ho Chi Minh City, Vietnam"):
        """
        Initialize data fetcher
        
        Args:
            api_key: Visual Crossing API key
            location: Location string for weather data
        """
        self.api_key = api_key
        self.location = location
        self.base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
    
    def fetch_historical_data(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Fetch historical weather data
        
        Args:
            start_date: Start date in format 'YYYY-MM-DD'
            end_date: End date in format 'YYYY-MM-DD'
        
        Returns:
            DataFrame with weather data or None if failed
        """
        url = f"{self.base_url}/{self.location}/{start_date}/{end_date}"
        
        params = {
            'unitGroup': 'metric',
            'key': self.api_key,
            'include': 'days',
            'elements': (
                'name,address,resolvedAddress,latitude,longitude,datetime,tempmax,tempmin,temp,feelslikemax,feelslikemin,feelslike,'
                'dew,humidity,precip,precipprob,precipcover,snow,snowdepth,'
                'windgust,windspeed,winddir,sealevelpressure,cloudcover,visibility,'
                'solarradiation,solarenergy,uvindex,conditions,description,icon,'
                'sunrise,sunset,moonphase'
            )
        }
        
        try:
            print(f"🌐 Fetching data from {start_date} to {end_date}...")
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            df = pd.DataFrame(data['days'])
            
            print(f"✅ Successfully fetched {len(df)} days of data")
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching data: {e}")
            return None
    
    def fetch_latest_data(self, days_back: int = 30) -> Optional[pd.DataFrame]:
        """
        Fetch most recent weather data
        
        Args:
            days_back: Number of days to fetch from today
        
        Returns:
            DataFrame with weather data or None if failed
        """
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        return self.fetch_historical_data(start_date, end_date)
    
    def fetch_since_date(self, since_date: str) -> Optional[pd.DataFrame]:
        """
        Fetch data from a specific date until today
        
        Args:
            since_date: Start date in format 'YYYY-MM-DD'
        
        Returns:
            DataFrame with weather data or None if failed
        """
        end_date = datetime.now().strftime('%Y-%m-%d')
        return self.fetch_historical_data(since_date, end_date)


def load_local_data(data_path: str = "data/weather_hcm_daily.csv") -> pd.DataFrame:
    """
    Load local weather data
    
    Args:
        data_path: Path to CSV file
    
    Returns:
        DataFrame with weather data
    """
    if data_path.startswith('http'):
        df = pd.read_csv(data_path)
    else:
        # Handle relative path from backend root
        if not os.path.isabs(data_path):
            backend_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            data_path = os.path.join(backend_root, data_path)
        
        if not os.path.exists(data_path):
            print(f"📂 No local data found at {data_path}, returning empty DataFrame")
            return pd.DataFrame(columns=['datetime'])  # Empty DF with datetime column
    
        df = pd.read_csv(data_path)
    
    print(f"📂 Loaded {len(df)} rows from local data")
    return df


def merge_data(original_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge new data with original data, avoiding duplicates
    
    Args:
        original_df: Original training data
        new_df: New data from API
    
    Returns:
        Combined DataFrame
    """
    # Ensure datetime columns
    original_df['datetime'] = pd.to_datetime(original_df['datetime'])
    new_df['datetime'] = pd.to_datetime(new_df['datetime'])
    
    # Remove duplicates based on datetime
    combined = pd.concat([original_df, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=['datetime'], keep='last')
    combined = combined.sort_values('datetime').reset_index(drop=True)
    
    print(f"📊 Original data: {len(original_df)} rows")
    print(f"📊 New data: {len(new_df)} rows")
    print(f"📊 Combined data: {len(combined)} rows (after deduplication)")
    
    return combined


def save_combined_data(df: pd.DataFrame, output_path: str = "data/combined_weather_data.csv"):
    """
    Save combined data to CSV
    
    Args:
        df: DataFrame to save
        output_path: Output file path
    """
    # Handle relative path from backend root
    if not os.path.isabs(output_path):
        backend_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        output_path = os.path.join(backend_root, output_path)
    
    # Create directory if not exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"💾 Saved combined data to {output_path}")


def update_local_data(
    api_key: str,
    data_path: str = "data/weather_hcm_daily.csv",
    default_start_date: str = "2025-01-01"  # Fallback if no local data
) -> None:
    """
    Update local weather data by fetching new data from the last recorded date until today,
    merging it, and saving back to the same file.
    
    Args:
        api_key: Visual Crossing API key
        data_path: Path to the local CSV file
        default_start_date: Default start date if no local data exists
    """
    df = load_local_data(data_path)
    
    if len(df) == 0:
        start_date = default_start_date
        print(f"📅 No local data, fetching from default start date: {start_date}")
    else:
        last_date = df['datetime'].max()
        start_date_obj = pd.to_datetime(last_date) + timedelta(days=1)
        start_date = start_date_obj.strftime('%Y-%m-%d')
        print(f"📅 Last local data date: {last_date.date()}, fetching from {start_date}")
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    if start_date > end_date:
        print("✅ Data is already up to date")
        return
    
    fetcher = VisualCrossingDataFetcher(api_key)
    new_df = fetcher.fetch_historical_data(start_date, end_date)
    
    if new_df is not None:
        combined = merge_data(df, new_df)
        save_combined_data(combined, data_path)
    else:
        print("❌ No new data fetched, local data unchanged")