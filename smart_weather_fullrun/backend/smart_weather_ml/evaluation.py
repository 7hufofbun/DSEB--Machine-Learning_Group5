
import numpy as np
from .metrics import calculate_all_metrics
def evaluate_on_test_set_summary(pipeline, X_test, y_test):
    test_metrics_dict = {}
    for col, model_info in pipeline["models"].items():
        if model_info is None: continue
        model = model_info["model"]
        y_pred_test = model.predict(X_test)
        test_metrics_dict[col] = calculate_all_metrics(y_test[col], y_pred_test)
    return test_metrics_dict
def comprehensive_final_evaluation_with_avg(pipeline, X_train, y_train, X_test, y_test):
    final_results = {}
    metrics_acc = {"train": [], "validation": [], "test": []}
    for col, model_info in pipeline["models"].items():
        if model_info is None: continue
        model = model_info["model"]
        y_pred_tr  = model.predict(X_train)
        train_m    = calculate_all_metrics(y_train[col], y_pred_tr)
        cv_metrics = pipeline["cv_metrics"][col]
        y_pred_te  = model.predict(X_test)
        test_m     = calculate_all_metrics(y_test[col], y_pred_te)
        final_results[col] = {"train": train_m, "validation": cv_metrics, "test": test_m}
        metrics_acc["train"].append(train_m)
        metrics_acc["validation"].append(cv_metrics)
        metrics_acc["test"].append(test_m)
    def avg(ms): return {k: np.mean([m[k] for m in ms]) for k in ["rmse", "mae", "r2", "mse"]}
    return final_results, {"train": avg(metrics_acc["train"]), "validation": avg(metrics_acc["validation"]), "test": avg(metrics_acc["test"])}
def analyze_overfitting(final_results):
    for target, results in final_results.items():
        tr, te = results["train"], results["test"]
        print(f"\n{target}: Train RMSE={tr['rmse']:.4f} vs Test RMSE={te['rmse']:.4f} | Train R2={tr['r2']:.4f} vs Test R2={te['r2']:.4f}")
