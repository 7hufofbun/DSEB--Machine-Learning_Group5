"""
Retraining trigger logic
Location: backend/smart_weather_ml/retrain/triggers.py
"""

import pandas as pd
from typing import Tuple, Dict, Any
from datetime import timedelta


def check_retrain_trigger(
    original_df: pd.DataFrame,
    new_df: pd.DataFrame,
    min_new_samples: int = 30,
    date_threshold_days: int = 7
) -> Tuple[bool, str]:
    """
    Check if retraining should be triggered
    
    Triggers:
    1. Minimum number of new samples
    2. Data freshness (recent data available)
    3. Data quality checks
    
    Args:
        original_df: Original training data
        new_df: New data from API
        min_new_samples: Minimum new samples required
        date_threshold_days: Minimum days difference required
    
    Returns:
        (should_retrain, reason)
    """
    if new_df is None or len(new_df) == 0:
        return False, "No new data available"
    
    # Handle empty original data (first time training)
    if len(original_df) == 0:
        if len(new_df) >= min_new_samples:
            return True, f"✅ Initial training with {len(new_df)} samples"
        else:
            return False, f"Insufficient data for initial training: {len(new_df)} < {min_new_samples}"
    
    # Ensure datetime columns
    original_df['datetime'] = pd.to_datetime(original_df['datetime'])
    new_df['datetime'] = pd.to_datetime(new_df['datetime'])
    
    # Count truly new samples (not in original data)
    original_dates = set(original_df['datetime'].dt.date)
    new_dates = set(new_df['datetime'].dt.date)
    truly_new_dates = new_dates - original_dates
    num_truly_new = len(truly_new_dates)
    
    # Check 1: Minimum samples
    if num_truly_new < min_new_samples:
        return False, f"Insufficient new data: {num_truly_new} new samples < {min_new_samples} required"
    
    # Check 2: Data freshness
    latest_original = original_df['datetime'].max()
    latest_new = new_df['datetime'].max()
    days_difference = (latest_new - latest_original).days
    
    if days_difference < date_threshold_days:
        return False, f"New data not recent enough: {days_difference} days newer < {date_threshold_days} days required"
    
    # Check 3: Data quality (basic checks)
    required_cols = ['datetime', 'temp', 'tempmax', 'tempmin']
    missing_cols = [col for col in required_cols if col not in new_df.columns]
    if missing_cols:
        return False, f"Missing required columns: {missing_cols}"
    
    # Check for excessive missing values
    missing_pct = new_df[required_cols].isnull().mean() * 100
    max_missing = missing_pct.max()
    if max_missing > 50:
        return False, f"Too many missing values in new data: {max_missing:.1f}% (max 50%)"
    
    # All checks passed - trigger retraining
    reason = (
        f"✅ Retraining triggered: "
        f"{num_truly_new} new samples, "
        f"{days_difference} days newer than existing data"
    )
    return True, reason


def check_data_drift(
    original_df: pd.DataFrame, 
    new_df: pd.DataFrame,
    drift_threshold: float = 2.0
) -> Tuple[bool, Dict[str, Any]]:
    """
    Check for data drift between original and new data
    
    Args:
        original_df: Original training data
        new_df: New data from API
        drift_threshold: Number of standard deviations for drift detection
    
    Returns:
        (has_drift, drift_details)
    """
    drift_details = {}
    
    # Skip if not enough data
    if len(original_df) == 0 or len(new_df) == 0:
        return False, {'message': 'Insufficient data for drift detection'}
    
    # Compare temperature statistics
    numeric_cols = ['temp', 'tempmax', 'tempmin', 'humidity', 'precip']
    
    for col in numeric_cols:
        if col in original_df.columns and col in new_df.columns:
            # Skip if too many missing values
            if original_df[col].isnull().mean() > 0.5 or new_df[col].isnull().mean() > 0.5:
                continue
            
            orig_mean = original_df[col].mean()
            new_mean = new_df[col].mean()
            
            orig_std = original_df[col].std()
            new_std = new_df[col].std()
            
            if orig_std == 0:  # Avoid division by zero
                continue
            
            # Calculate z-score for drift detection
            mean_diff = abs(new_mean - orig_mean)
            z_score = mean_diff / orig_std
            
            # Check if mean has shifted significantly
            if z_score > drift_threshold:
                drift_level = 'HIGH' if z_score > 3 else 'MODERATE'
                drift_details[col] = {
                    'original_mean': round(orig_mean, 2),
                    'new_mean': round(new_mean, 2),
                    'difference': round(new_mean - orig_mean, 2),
                    'z_score': round(z_score, 2),
                    'drift': drift_level
                }
    
    has_drift = len(drift_details) > 0
    
    if has_drift:
        print("\n⚠️  Data drift detected:")
        for col, details in drift_details.items():
            print(f"   {col}: {details['drift']} drift (z-score: {details['z_score']})")
            print(f"      Original mean: {details['original_mean']}, New mean: {details['new_mean']}")
    
    return has_drift, drift_details


def should_force_retrain(
    last_retrain_date: pd.Timestamp, 
    max_days_without_retrain: int = 90
) -> Tuple[bool, str]:
    """
    Check if retraining should be forced based on time elapsed
    
    Args:
        last_retrain_date: Date of last retraining
        max_days_without_retrain: Maximum days without retraining
    
    Returns:
        (should_force, reason)
    """
    if pd.isna(last_retrain_date):
        return True, "No previous retraining record found"
    
    days_since_retrain = (pd.Timestamp.now() - last_retrain_date).days
    
    if days_since_retrain >= max_days_without_retrain:
        return True, f"⏰ Force retrain: {days_since_retrain} days since last retrain (max: {max_days_without_retrain})"
    
    return False, f"Last retrain was {days_since_retrain} days ago (within {max_days_without_retrain} day threshold)"


def validate_data_quality(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate data quality before retraining
    
    Args:
        df: DataFrame to validate
    
    Returns:
        (is_valid, message)
    """
    if len(df) == 0:
        return False, "Dataset is empty"
    
    # Check required columns
    required_cols = ['datetime', 'temp', 'tempmax', 'tempmin', 'humidity']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return False, f"Missing required columns: {missing_cols}"
    
    # Check date range
    df['datetime'] = pd.to_datetime(df['datetime'])
    date_range = (df['datetime'].max() - df['datetime'].min()).days
    if date_range < 365:
        return False, f"Insufficient date range: {date_range} days (minimum 365 days required)"
    
    # Check for excessive missing values
    missing_pct = df[required_cols].isnull().mean() * 100
    max_missing = missing_pct.max()
    if max_missing > 80:
        return False, f"Too many missing values: {max_missing:.1f}% (max 80%)"
    
    # Check for duplicate dates
    duplicate_dates = df['datetime'].duplicated().sum()
    if duplicate_dates > 0:
        return False, f"Found {duplicate_dates} duplicate dates in dataset"
    
    # All validation passed
    return True, f"✅ Data quality validation passed ({len(df)} samples, {date_range} days)"


def analyze_data_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze data distribution for monitoring
    
    Args:
        df: DataFrame to analyze
    
    Returns:
        Dictionary with distribution statistics
    """
    if len(df) == 0:
        return {'error': 'Empty dataset'}
    
    stats = {
        'total_samples': len(df),
        'date_range': {
            'start': df['datetime'].min().strftime('%Y-%m-%d'),
            'end': df['datetime'].max().strftime('%Y-%m-%d'),
            'days': (df['datetime'].max() - df['datetime'].min()).days
        },
        'missing_values': {},
        'temperature_stats': {}
    }
    
    # Missing value analysis
    for col in df.columns:
        missing_pct = df[col].isnull().mean() * 100
        if missing_pct > 0:
            stats['missing_values'][col] = round(missing_pct, 2)
    
    # Temperature statistics
    temp_cols = ['temp', 'tempmax', 'tempmin']
    for col in temp_cols:
        if col in df.columns:
            stats['temperature_stats'][col] = {
                'mean': round(df[col].mean(), 2),
                'std': round(df[col].std(), 2),
                'min': round(df[col].min(), 2),
                'max': round(df[col].max(), 2)
            }
    
    return stats