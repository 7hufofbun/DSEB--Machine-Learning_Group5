
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
def comprehensive_final_evaluation_with_avg(final_models, X_train, y_train,X_test, y_test):
    results_per_target = {}
    train_metrics_accumulator = []
    test_metrics_accumulator = []
    for col, model_info in final_models.items():
        if model_info is None:
            continue
        model = model_info['model']
        X_train_final, y_train_final = X_train.copy(), y_train[col].copy()
        y_pred_train = model.predict(X_train_final)
        train_metrics = calculate_all_metrics(y_train_final, y_pred_train)
        train_metrics_accumulator.append(train_metrics)
        X_test_final, y_test_final = X_test.copy(), y_test[col].copy()
        y_pred_test = model.predict(X_test_final)
        test_metrics = calculate_all_metrics(y_test_final, y_pred_test)
        test_metrics_accumulator.append(test_metrics)
        results_per_target[col] = {
            'train': train_metrics,
            'test': test_metrics
        }
    avg_train = {k: np.mean([m[k] for m in train_metrics_accumulator]) for k in ['rmse', 'mae', 'r2', 'mse']}
    avg_test  = {k: np.mean([m[k] for m in test_metrics_accumulator]) for k in ['rmse', 'mae', 'r2', 'mse']}
    return results_per_target, {'train': avg_train, 'test': avg_test}

