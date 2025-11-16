"""
Main retraining pipeline
Location: backend/smart_weather_ml/retrain/retrain_pipeline.py
"""

import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_root = os.path.dirname(os.path.dirname(current_dir))
if backend_root not in sys.path:
    sys.path.insert(0, backend_root)

from smart_weather_ml.retrain.data_fetcher import (
    VisualCrossingDataFetcher,
    load_local_data,
    merge_data,
    save_combined_data,
    update_local_data
)
from smart_weather_ml.retrain.triggers import (
    check_retrain_trigger,
    check_data_drift,
    validate_data_quality,
    analyze_data_distribution
)

# Import model components
from smart_weather_ml.model.utils import set_seed
from smart_weather_ml.model.preprocessing import Preprocessor, basic_cleaning, split_data
from smart_weather_ml.model.features import feature_engineer
from smart_weather_ml.model.pipeline import complete_model_pipeline
from smart_weather_ml.model.evaluation import comprehensive_final_evaluation_with_avg
from smart_weather_ml.model.export import save_models_to_onnx


class ModelRetrainingPipeline:
    """Manage model retraining with new data"""
    
    def __init__(
        self,
        api_key: str,
        original_data_path: str = "data/weather_hcm_daily.csv",
        models_dir: str = "models_onnx/",
        backend_root: Optional[str] = None
    ):
        """
        Initialize retraining pipeline
        
        Args:
            api_key: Visual Crossing API key
            original_data_path: Path to original training data
            models_dir: Directory to save ONNX models
            backend_root: Backend root directory (auto-detected if None)
        """
        self.api_key = api_key
        self.original_data_path = original_data_path
        self.models_dir = models_dir
        
        # Get backend root directory
        if backend_root is None:
            current_file = os.path.abspath(__file__)
            self.backend_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
        else:
            self.backend_root = backend_root
        
        # Resolve paths relative to backend root
        if not os.path.isabs(self.original_data_path):
            self.original_data_path = os.path.join(self.backend_root, self.original_data_path)
        if not os.path.isabs(self.models_dir):
            self.models_dir = os.path.join(self.backend_root, self.models_dir)
        
        # Create models directory if not exists
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize data fetcher
        self.fetcher = VisualCrossingDataFetcher(api_key)
    
    def load_original_data(self) -> pd.DataFrame:
        """Load original training data"""
        return load_local_data(self.original_data_path)
    
    def check_trigger(
        self,
        original_df: pd.DataFrame,
        new_df: pd.DataFrame,
        min_new_samples: int = 30,
        date_threshold_days: int = 7
    ) -> Tuple[bool, str]:
        """
        Check if retraining should be triggered
        
        Returns:
            (should_retrain, reason)
        """
        return check_retrain_trigger(
            original_df, new_df, min_new_samples, date_threshold_days
        )
    
    def retrain_models(
        self,
        combined_data: pd.DataFrame,
        n_trials: int = 30,
        use_clearml: bool = False
    ) -> Tuple[Dict, Dict, Optional[Any]]:
        """
        Complete retraining pipeline
        
        Args:
            combined_data: Combined DataFrame with all data
            n_trials: Number of Optuna trials for hyperparameter tuning
            use_clearml: Whether to use ClearML for tracking
        
        Returns:
            (pipeline, final_results, clearml_task)
        """
        print("\n" + "="*80)
        print("🔄 STARTING MODEL RETRAINING PIPELINE")
        print("="*80)
        
        # Initialize ClearML task if requested
        task = None
        logger = None
        if use_clearml:
            try:
                from clearml import Task
                task = Task.init(
                    project_name="Weather Forecast HCM",
                    task_name=f"LGBM Retrain - {datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    task_type=Task.TaskTypes.training,
                    output_uri=True
                )
                logger = task.get_logger()
                print("✅ ClearML tracking enabled")
            except ImportError:
                print("⚠️  ClearML not available, skipping tracking")
            except Exception as e:
                print(f"⚠️  ClearML initialization failed: {e}")
        
        # Set seed for reproducibility
        set_seed(42)
        
        # Validate data quality
        is_valid, validation_msg = validate_data_quality(combined_data)
        print(f"\n📊 Data validation: {validation_msg}")
        
        if not is_valid:
            raise ValueError(f"Data quality check failed: {validation_msg}")
        
        # Analyze data distribution
        data_stats = analyze_data_distribution(combined_data)
        print(f"\n📈 Data statistics:")
        print(f"   Total samples: {data_stats['total_samples']}")
        print(f"   Date range: {data_stats['date_range']['start']} to {data_stats['date_range']['end']}")
        print(f"   Duration: {data_stats['date_range']['days']} days")
        
        # Data cleaning
        data = basic_cleaning(combined_data)
        if logger:
            logger.report_table(
                title="Data Overview After Cleaning",
                series="basic_cleaning",
                table_plot=data.describe().reset_index()
            )
        
        # Split data
        train_x, train_y, test_x, test_y = split_data(data)
        print(f"\n📊 Data split:")
        print(f"   Train: {train_x.shape}")
        print(f"   Test: {test_x.shape}")
        
        if logger:
            logger.report_scalar(
                title="Data Split", series="Train Samples", value=len(train_x), iteration=0
            )
            logger.report_scalar(
                title="Data Split", series="Test Samples", value=len(test_x), iteration=0
            )
        
        # Preprocessing
        print("\n🔧 Preprocessing data...")
        preprocessor = Preprocessor(threshold=50, var_threshold=0.0)
        X_train_processed = preprocessor.fit_transform(train_x)
        X_test_processed = preprocessor.transform(test_x)
        print(f"   Train: {X_train_processed.shape}")
        print(f"   Test: {X_test_processed.shape}")
        
        if logger:
            logger.report_scalar(
                title="Preprocessing", series="Features After Preprocessing", 
                value=X_train_processed.shape[1], iteration=0
            )
        
        # Feature engineering
        print("\n🎨 Engineering features...")
        X_train_fe, y_train_fe = feature_engineer(X_train_processed, train_y)
        X_test_fe, y_test_fe = feature_engineer(X_test_processed, test_y)
        print(f"   Train: {X_train_fe.shape} ({X_train_fe.shape[1]} features)")
        print(f"   Test: {X_test_fe.shape}")
        
        if logger:
            logger.report_scalar(
                title="Feature Engineering", series="Total Features", 
                value=X_train_fe.shape[1], iteration=0
            )
            logger.report_scalar(
                title="Feature Engineering", series="Training Samples", 
                value=X_train_fe.shape[0], iteration=0
            )
        
        # Training pipeline
        print(f"\n🚀 Training models with {n_trials} Optuna trials...")
        print("   This may take a while...\n")
        
        pipeline = complete_model_pipeline(
            X_train=X_train_fe,
            y_train=y_train_fe,
            preprocessor=preprocessor,  # Add this
            optimization_params={'n_trials': n_trials}
        )
        
        # Comprehensive evaluation
        print("\n📊 Evaluating models...")
        final_results, avg_metrics = comprehensive_final_evaluation_with_avg(
            pipeline['models'],  # ← Pass pipeline['models'], not pipeline
            X_train_fe, 
            y_train_fe, 
            X_test_fe, 
            y_test_fe
    )
        
        # Log metrics if ClearML is available
        if logger:
            for iteration_variable, (target_name, metrics) in enumerate(final_results.items()):
                for dataset_type, metric_dict in metrics.items():
                    logger.report_scalar(
                        title=f"Metrics/{target_name}", series=f"R2_{dataset_type}", 
                        value=metric_dict['r2'], iteration=iteration_variable
                    )
                    logger.report_scalar(
                        title=f"Metrics/{target_name}", series=f"RMSE_{dataset_type}", 
                        value=metric_dict['rmse'], iteration=iteration_variable
                    )
                    logger.report_scalar(
                        title=f"Metrics/{target_name}", series=f"MAE_{dataset_type}", 
                        value=metric_dict['mae'], iteration=iteration_variable
                    )
            
            logger.report_scalar(
                title="Average Metrics", series="R2_Test", 
                value=avg_metrics['test']['r2'], iteration=0
            )
            logger.report_scalar(
                title="Average Metrics", series="RMSE_Test", 
                value=avg_metrics['test']['rmse'], iteration=0
            )
            logger.report_scalar(
                title="Average Metrics", series="MAE_Test", 
                value=avg_metrics['test']['mae'], iteration=0
            )
        
        # Save models to ONNX
        print(f"\n💾 Saving models to {self.models_dir}")
        save_models_to_onnx(pipeline, save_dir=self.models_dir)
        
        # Upload artifacts to ClearML if available
        if task:
            for target_name in pipeline['models'].keys():
                model_path = os.path.join(self.models_dir, f"{target_name}.onnx")
                if os.path.exists(model_path):
                    task.upload_artifact(
                        name=f"{target_name}_onnx_model", 
                        artifact_object=model_path
                    )
        
        print("\n" + "="*80)
        print("✅ RETRAINING COMPLETED SUCCESSFULLY")
        print("="*80)
        
        # Add test metrics to pipeline
        pipeline['test_metrics'] = avg_metrics['test']
        pipeline['final_results'] = final_results

        # Persist a lightweight metrics summary for the web UI
        try:
            import json
            metrics_out = {
                'train': avg_metrics.get('train', {}),
                'test': avg_metrics.get('test', {}),
                'per_target': final_results,
                'evaluated_at': datetime.now().isoformat(),
            }
            metrics_path = os.path.join(self.backend_root, 'model_metrics.json')
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics_out, f, default=lambda o: o if isinstance(o, (int, float, str, type(None), dict, list)) else str(o))
            print(f"Saved model metrics summary to {metrics_path}")
        except Exception as e:
            print(f"Warning: failed to persist model metrics summary: {e}")
        
        return pipeline, final_results, task
    
    def run_retrain_pipeline(
        self,
        days_back: int = 60,
        min_new_samples: int = 30,
        date_threshold_days: int = 7,
        n_trials: int = 30,
        use_clearml: bool = False,
        force: bool = False,
        check_drift: bool = True
    ) -> Dict[str, Any]:
        """
        Complete pipeline: fetch, check, and retrain
        
        Args:
            days_back: How many days of recent data to fetch
            min_new_samples: Minimum new samples to trigger retrain
            date_threshold_days: Minimum days difference to trigger retrain
            n_trials: Optuna trials for hyperparameter tuning
            use_clearml: Whether to use ClearML for tracking
            force: Force retraining without trigger checks
            check_drift: Check for data drift
        
        Returns:
            dict with status and results
        """
        try:
            # Load original data
            print("📂 Loading original data...")
            original_data = self.load_original_data()
            
            # Fetch new data
            print(f"🌐 Fetching latest {days_back} days of data from API...")
            new_data = self.fetcher.fetch_latest_data(days_back=days_back)
            
            if new_data is None:
                return {
                    'status': 'failed',
                    'message': 'Failed to fetch new data from API',
                    'error': 'API request failed'
                }
            
            # Check for data drift if requested
            if check_drift and len(original_data) > 0:
                has_drift, drift_details = check_data_drift(original_data, new_data)
                if has_drift:
                    print("\n⚠️  Data drift detected - retraining recommended")
            
            # Check if retraining should be triggered (unless forced)
            if not force:
                should_retrain, reason = self.check_trigger(
                    original_data, new_data,
                    min_new_samples=min_new_samples,
                    date_threshold_days=date_threshold_days
                )
                
                print(f"\n🔍 Retrain Check: {reason}")
                
                if not should_retrain:
                    return {
                        'status': 'skipped',
                        'message': reason,
                        'new_data_count': len(new_data),
                        'original_data_count': len(original_data)
                    }
            else:
                print("\n⚠️  FORCE MODE: Skipping trigger checks")
            
            # Merge data
            print("\n📊 Merging data...")
            combined_data = merge_data(original_data, new_data)
            
            # Save combined data (for future retraining)
            combined_path = os.path.join(self.backend_root, "data/combined_weather_data.csv")
            save_combined_data(combined_data, combined_path)
            
            # Also update the original data file
            save_combined_data(combined_data, self.original_data_path)
            print(f"   Updated original data file: {self.original_data_path}")
            
            # Retrain models
            pipeline, final_results, task = self.retrain_models(
                combined_data, n_trials=n_trials, use_clearml=use_clearml
            )
            
            result = {
                'status': 'success',
                'message': 'Models retrained successfully',
                'new_data_count': len(new_data),
                'original_data_count': len(original_data),
                'total_data_count': len(combined_data),
                'avg_test_r2': pipeline['test_metrics']['r2'],
                'avg_test_rmse': pipeline['test_metrics']['rmse'],
                'avg_test_mae': pipeline['test_metrics']['mae'],
                'models_saved': list(pipeline['models'].keys()),
                'models_dir': self.models_dir
            }
            
            if task:
                result['clearml_task_id'] = task.id
            
            return result
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"\n❌ Error during retraining: {str(e)}")
            print(error_trace)
            
            return {
                'status': 'error',
                'message': str(e),
                'traceback': error_trace
            }
    
    def update_data_only(self, default_start_date: str = "2023-01-01") -> Dict[str, Any]:
        """
        Update local data without retraining models
        
        Args:
            default_start_date: Default start date if no local data exists
        
        Returns:
            dict with status and results
        """
        try:
            print("📂 Updating local data...")
            
            update_local_data(
                api_key=self.api_key,
                data_path=self.original_data_path,
                default_start_date=default_start_date
            )
            
            # Load updated data to get stats
            updated_data = self.load_original_data()
            
            return {
                'status': 'success',
                'message': 'Data updated successfully',
                'total_samples': len(updated_data),
                'date_range': {
                    'start': updated_data['datetime'].min().strftime('%Y-%m-%d'),
                    'end': updated_data['datetime'].max().strftime('%Y-%m-%d')
                }
            }
            
        except Exception as e:
            import traceback
            return {
                'status': 'error',
                'message': str(e),
                'traceback': traceback.format_exc()
            }