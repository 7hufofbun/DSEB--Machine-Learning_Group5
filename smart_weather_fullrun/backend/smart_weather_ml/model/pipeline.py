
def complete_model_pipeline_all_features(X_train, y_train, optimization_params=None):
    from .tuning import optimize_model_all_features
    from .lgbm import train_final_model_all_features
    if optimization_params is None:
        optimization_params = {"n_trials": 1}
    models, best_params_dict, train_metrics_dict, cv_metrics_dict = {}, {}, {}, {}
    target_columns = [c for c in y_train.columns if c.startswith("temp_t+")]
    for col in target_columns:
        best_params, cv_metrics = optimize_model_all_features(
            X_train=X_train, y_train=y_train[col], target_name=col, n_trials=optimization_params["n_trials"]
        )
        model_result = train_final_model_all_features(X_train=X_train, y_train=y_train[col], best_params=best_params)
        models[col] = model_result
        best_params_dict[col] = best_params
        train_metrics_dict[col] = model_result["train_metrics"]
        cv_metrics_dict[col] = cv_metrics
    return {
        "models": models,
        "features": {col: model_info["feature_names"] for col, model_info in models.items()},
        "best_params": best_params_dict,
        "train_metrics": train_metrics_dict,
        "cv_metrics": cv_metrics_dict,
        "feature_selection_method": "ALL FEATURES",
        "pipeline_type": "all_features",
    }
