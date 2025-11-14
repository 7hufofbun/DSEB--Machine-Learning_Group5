#!/usr/bin/env python3
"""
Manual retraining script
Location: backend/smart_weather_ml/scripts/run_retrain.py

Cach dung` : Paste vao terminal (o duoi de set trial optuna  con cai force la force no retrain du k du data)
    python -m smart_weather_fullrun.backend.smart_weather_ml.scripts.run_retrain
    python -m smart_weather_fullrun.backend.smart_weather_ml.scripts.run_retrain --trial 1 --force
"""

import os
import sys
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
# API key
# FLPRTVQTM65DDXU2R6JQR2UUT
# 8TP79HQ3WEJ5K3FFUFMTZ8GBT
# set api key : [System.Environment]::SetEnvironmentVariable("VISUAL_CROSSING_API_KEY", "your_key", "User") (cai nay tuy mng dung api key cua t hoac cua mng)

from smart_weather_ml.retrain.retrain_pipeline import ModelRetrainingPipeline


def load_config():
    """Load configuration from environment variables"""
    api_key = os.getenv('VISUAL_CROSSING_API_KEY')
    
    if not api_key:
        print("❌ ERROR: API key not found!")
        print("   Set VISUAL_CROSSING_API_KEY environment variable")
        print("   Example: export VISUAL_CROSSING_API_KEY='your_key_here'")
        sys.exit(1)
    
    return {
        'api_key': api_key,
        'original_data_path': os.getenv('DATA_PATH', 'data/weather_hcm_daily.csv'),
        'models_dir': os.getenv('MODELS_DIR', 'models_onnx/'),
        'days_back': int(os.getenv('DAYS_BACK', '90')),
        'min_new_samples': int(os.getenv('MIN_NEW_SAMPLES', '30')),
        'date_threshold_days': int(os.getenv('DATE_THRESHOLD_DAYS', '7')),
        'n_trials': int(os.getenv('N_TRIALS', '30')),
        'use_clearml': os.getenv('USE_CLEARML', 'false').lower() == 'true'
    }


def main():
    parser = argparse.ArgumentParser(
        description='Retrain weather forecast models with new data from Visual Crossing API'
    )
    parser.add_argument(
        '--force', 
        action='store_true',
        help='Force retraining without trigger checks'
    )
    parser.add_argument(
        '--trials', 
        type=int, 
        default=None,
        help='Number of Optuna trials (default: from config/env)'
    )
    parser.add_argument(
        '--days-back', 
        type=int, 
        default=None,
        help='Days of historical data to fetch (default: from config/env)'
    )
    parser.add_argument(
        '--min-samples',
        type=int,
        default=None,
        help='Minimum new samples required for retraining'
    )
    parser.add_argument(
        '--use-clearml',
        action='store_true',
        help='Enable ClearML tracking'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    # Override with command line arguments
    if args.trials:
        config['n_trials'] = args.trials
    if args.days_back:
        config['days_back'] = args.days_back
    if args.min_samples:
        config['min_new_samples'] = args.min_samples
    if args.use_clearml:
        config['use_clearml'] = True
    
    # Print header
    print("\n" + "="*80)
    print("🌤️  WEATHER FORECAST MODEL RETRAINING")
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Print configuration
    print("\n📋 Configuration:")
    print(f"   API Key: {'*' * 20}{config['api_key'][-8:]}")
    print(f"   Data path: {config['original_data_path']}")
    print(f"   Models dir: {config['models_dir']}")
    print(f"   Days back: {config['days_back']}")
    print(f"   Min new samples: {config['min_new_samples']}")
    print(f"   Date threshold: {config['date_threshold_days']} days")
    print(f"   Optuna trials: {config['n_trials']}")
    print(f"   Use ClearML: {config['use_clearml']}")
    print(f"   Force retrain: {args.force}")
    
    # Initialize pipeline
    retrain_pipeline = ModelRetrainingPipeline(
        api_key=config['api_key'],
        original_data_path=config['original_data_path'],
        models_dir=config['models_dir']
    )
    
    try:
        # Run retraining
        result = retrain_pipeline.run_retrain_pipeline(
            days_back=config['days_back'],
            min_new_samples=config['min_new_samples'],
            date_threshold_days=config['date_threshold_days'],
            n_trials=config['n_trials'],
            use_clearml=config['use_clearml'],
            force=args.force
        )
        
        # Print results
        print("\n" + "="*80)
        print(f"📊 RETRAINING RESULT: {result['status'].upper()}")
        print("="*80)
        print(f"Message: {result['message']}")
        
        if 'new_data_count' in result:
            print(f"New data samples: {result['new_data_count']}")
        if 'total_data_count' in result:
            print(f"Total data samples: {result['total_data_count']}")
        
        if result['status'] == 'success':
            print(f"\n✅ Models successfully retrained!")
            print(f"   📁 Location: {config['models_dir']}")
            print(f"   📈 Average Test R²: {result.get('avg_test_r2', 'N/A'):.4f}")
            print(f"   📉 Average Test RMSE: {result.get('avg_test_rmse', 'N/A'):.4f}")
            print(f"   🎯 Models: {', '.join(result.get('models_saved', []))}")
            
            if config['use_clearml']:
                print(f"   🔗 ClearML Task ID: {result.get('clearml_task_id', 'N/A')}")
            
            sys.exit(0)
            
        elif result['status'] == 'skipped':
            print("\n⏭️  Retraining skipped - trigger conditions not met")
            print("   Use --force to retrain anyway")
            sys.exit(0)
            
        else:
            print(f"\n❌ Retraining failed")
            if 'traceback' in result:
                print("\nError details:")
                print(result['traceback'])
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Retraining cancelled by user")
        sys.exit(130)
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
