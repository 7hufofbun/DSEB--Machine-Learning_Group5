from .utils import set_seed, init_clearml
from .io import get_data
from .preprocessing import basic_cleaning, split_data, Preprocessor
from .features import feature_engineer
from .pipeline import complete_model_pipeline
from .tuning import *
from .evaluation import evaluate_on_test_set_summary, comprehensive_final_evaluation_with_avg
from .export import save_models_to_onnx
import argparse
from pathlib import Path
from .lgbm import train_final_model_all_features
import os
import json

FEATURE_COLS_PATH = Path(__file__).with_name("feature_cols.json")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
DEFAULT_DATA = os.path.join(DATA_DIR, "weather_hcm_daily.csv")


def _save_feature_columns(feature_columns):
    try:
        FEATURE_COLS_PATH.parent.mkdir(parents=True, exist_ok=True)
        FEATURE_COLS_PATH.write_text(json.dumps(feature_columns, ensure_ascii=False, indent=2))
        print(f"Saved feature column list to {FEATURE_COLS_PATH}")
    except Exception as exc:
        print(f"Failed to persist feature columns: {exc}")

def main_all_features_pipeline(data, logger):
    data = basic_cleaning(data)
    try:
        logger.report_table(title="Data Overview After Cleaning", series="basic_cleaning", table_plot=data.describe().reset_index())
    except Exception:
        pass

    train_x, train_y, test_x, test_y = split_data(data)
    print(f"📊 Train data: {train_x.shape}, Test data: {test_x.shape}")
    logger.report_scalar(
        title="Data Split", series="Train Samples", value=len(train_x), iteration=0
    )
    logger.report_scalar(
        title="Data Split", series="Test Samples", value=len(test_x), iteration=0
    )
    preprocessor = Preprocessor(threshold=50, var_threshold=0.0)
    pipeline = complete_model_pipeline(
        train_x, train_y, preprocessor,
        optimization_params={'n_trials': 35}
    )
    final_models = {}
    for col in [f"temp_t+{i}" for i in range(1, 6)]:
        print("\n" + "="*60)
        print(f"🚀 Fitting FINAL model on full train set for: {col}")
        print("="*60)
        best_params = pipeline['best_params'][col]
        model_result = train_final_model_all_features(
            train_x, train_y, best_params, preprocessor, col
        )
        final_models[col] = model_result

    pipeline['models'] = final_models

    X_train_processed = preprocessor.transform(train_x)
    X_test_processed  = preprocessor.transform(test_x)

    X_train_fe, y_train_fe = feature_engineer(X_train_processed, train_y)
    X_test_fe, y_test_fe   = feature_engineer(X_test_processed, test_y)

    feature_columns = X_train_fe.columns.tolist()
    pipeline['feature_columns'] = feature_columns
    _save_feature_columns(feature_columns)

    final_results, avg_metrics = comprehensive_final_evaluation_with_avg(
        pipeline['models'], X_train_fe, y_train_fe, X_test_fe, y_test_fe
    )

    for iteration_variable, (target_name, metrics) in enumerate(final_results.items()):
        for dataset_type in ['train', 'test']:
            metric_dict = metrics[dataset_type]
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
    logger.report_scalar(title="Average Metrics", series="R2_Train", value=avg_metrics['train']['r2'], iteration=0)
    logger.report_scalar(title="Average Metrics", series="R2_Test",  value=avg_metrics['test']['r2'], iteration=0)
    logger.report_scalar(title="Average Metrics", series="RMSE_Train", value=avg_metrics['train']['rmse'], iteration=0)
    logger.report_scalar(title="Average Metrics", series="RMSE_Test",  value=avg_metrics['test']['rmse'], iteration=0)

    pipeline['final_results'] = final_results
    pipeline['test_metrics'] = avg_metrics['test']
    pipeline['train_metrics'] = avg_metrics['train']

    return pipeline, final_results



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=DEFAULT_DATA, help="CSV path or URL for training data")
    p.add_argument("--save_dir", default="models_onnx", help="Directory to write ONNX models")
    p.add_argument("--overwrite", action="store_true", help="Force overwrite existing ONNX files")
    return p.parse_args()

if __name__ == "__main__":
    set_seed(42)
    args = parse_args()

    # ClearML
    task, logger = init_clearml()

    # Data
    data = get_data(args.data)

    # Train pipeline
    pipeline, _ = main_all_features_pipeline(data, logger)

    # Export ONNX (truyền overwrite xuống export.py)
    export_stats = save_models_to_onnx(
        pipeline,
        save_dir=args.save_dir,
        logger=logger,
        overwrite=args.overwrite,   # <-- thêm dòng này
    )

    # Upload artifact + log metric (giữ nguyên như cũ)
    if task is not None:
        try:
            for target_name in pipeline["models"].keys():
                model_path = f"{Path(args.save_dir) / (target_name + '.onnx')}"
                task.upload_artifact(name=f"{target_name}_onnx_model", artifact_object=model_path)
            logger.report_scalar(title="Final Performance", series="Average_Test_R2",   value=pipeline["test_metrics"]["r2"],   iteration=0)
            logger.report_scalar(title="Final Performance", series="Average_Test_RMSE", value=pipeline["test_metrics"]["rmse"], iteration=0)
            logger.report_scalar(title="Final Performance", series="Average_Test_MSE", value=pipeline["test_metrics"]["mse"], iteration=0)
            logger.report_scalar(title="Final Performance", series="Average_Test_MAE", value=pipeline["test_metrics"]["mae"], iteration=0)
        except Exception as e:
            print(f"ClearML artifact upload failed: {e}")
