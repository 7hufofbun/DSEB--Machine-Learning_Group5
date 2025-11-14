from .tuning import optimize_model_all_features

import numpy as np
def complete_model_pipeline(X_train, y_train,  preprocessor, optimization_params=None):
    if optimization_params is None:
        optimization_params = {'n_trials': 20}
    best_params_dict = {}
    cv_metrics_dict = {}
    train_metrics_dict = {}
    
    target_columns = [f"temp_t+{i}" for i in range(1, 6)]
    for col in target_columns:
        best_params, cv_metrics, train_metrics = optimize_model_all_features(
            X_train,
            y_train,
            col,
            preprocessor,
            n_trials=optimization_params['n_trials']
        )
        best_params_dict[col] = best_params
        cv_metrics_dict[col] = cv_metrics
        train_metrics_dict[col] = train_metrics
    pipeline = {
            "best_params": best_params_dict,
            "cv_metrics": cv_metrics_dict,
            "train_metrics": train_metrics_dict,
            "features": X_train.columns.tolist(),
            "feature_selection_method": "ALL FEATURES",
            "pipeline_type": "all_features",
        }
    return pipeline
