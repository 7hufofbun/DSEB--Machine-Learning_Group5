from .utils import set_seed, init_clearml
from .io import get_data
from .preprocessing import basic_cleaning, split_data, Preprocessor
from .features import feature_engineer
from .model.pipeline import complete_model_pipeline_all_features
from .evaluation import evaluate_on_test_set_summary, comprehensive_final_evaluation_with_avg, analyze_overfitting
from .export import save_models_to_onnx
import argparse
from pathlib import Path

DEFAULT_DATA = "https://raw.githubusercontent.com/7hufofbun/DSEB--Machine-Learning_Group5/refs/heads/main/data/weather_hcm_daily.csv"

def main_all_features_pipeline(data, logger):
    data = basic_cleaning(data)
    try:
        logger.report_table(title="Data Overview After Cleaning", series="basic_cleaning", table_plot=data.describe().reset_index())
    except Exception:
        pass

    train_x, train_y, test_x, test_y = split_data(data)
    pre = Preprocessor(threshold=50, var_threshold=0.0)
    X_train_p = pre.fit_transform(train_x)
    X_test_p  = pre.transform(test_x)

    X_tr_fe, y_tr_fe = feature_engineer(X_train_p, train_y)
    X_te_fe, y_te_fe = feature_engineer(X_test_p,  test_y)

    pipeline = complete_model_pipeline_all_features(
        X_train=X_tr_fe, y_train=y_tr_fe, optimization_params={"n_trials": 30}
    )
    _ = evaluate_on_test_set_summary(pipeline, X_te_fe, y_te_fe)
    final_results, avg_metrics = comprehensive_final_evaluation_with_avg(
        pipeline, X_tr_fe, y_tr_fe, X_te_fe, y_te_fe
    )
    pipeline["test_metrics"] = avg_metrics["test"]
    pipeline["final_results"] = final_results
    pipeline["input_dim"] = X_tr_fe.shape[1]

    analyze_overfitting(final_results)
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
        except Exception as e:
            print(f"ClearML artifact upload failed: {e}")
