"""
Retraining module for weather forecast models

This module provides functionality for:
- Fetching new weather data from Visual Crossing API
- Checking retraining triggers and data quality
- Orchestrating the complete retraining pipeline
- Updating models with new data

Usage:
    from smart_weather_ml.retrain import ModelRetrainingPipeline
    
    pipeline = ModelRetrainingPipeline(api_key="your_key")
    result = pipeline.run_retrain_pipeline()
"""

from .data_fetcher import (
    VisualCrossingDataFetcher,
    load_local_data,
    merge_data,
    save_combined_data,
    update_local_data
)

from .triggers import (
    check_retrain_trigger,
    check_data_drift,
    should_force_retrain,
    validate_data_quality,
    analyze_data_distribution
)

from .retrain_pipeline import ModelRetrainingPipeline

__all__ = [
    # Data fetcher
    'VisualCrossingDataFetcher',
    'load_local_data',
    'merge_data',
    'save_combined_data',
    'update_local_data',
    
    # Triggers
    'check_retrain_trigger',
    'check_data_drift',
    'should_force_retrain',
    'validate_data_quality',
    'analyze_data_distribution',
    
    # Main pipeline
    'ModelRetrainingPipeline'
]

__version__ = '1.0.0'
