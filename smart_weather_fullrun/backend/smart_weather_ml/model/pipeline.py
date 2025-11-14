from .tuning import optimize_model_all_features

import numpy as np
def complete_model_pipeline(X_train, y_train, preprocessor, optimization_params=None):
    if optimization_params is None:
        optimization_params = {'n_trials': 20}
    
    models = {}  # ← Add this
    best_params_dict = {}
    cv_metrics_dict = {}
    train_metrics_dict = {}
    
    target_columns = [f"temp_t+{i}" for i in range(1, 6)]
    for col in target_columns:
        best_params, cv_metrics, train_metrics = optimize_model_all_features(
            X_train=X_train,
            y_train=y_train[col],
            target_name=col,
            preprocessor=preprocessor,
            n_trials=optimization_params['n_trials']
        )
        
        # ← ADD: Train final model with best params
        from .lgbm import get_model
        final_model = get_model(best_params)
        
        # Drop datetime columns before training
        X_train_clean = X_train.copy()
        datetime_cols = ['datetime', 'sunrise', 'sunset']
        for dc in datetime_cols:
            if dc in X_train_clean.columns:
                X_train_clean = X_train_clean.drop(dc, axis=1)
        
        final_model.fit(X_train_clean, y_train[col])
        
        # ← ADD: Store model in correct structure
        models[col] = {
            'model': final_model,
            'best_params': best_params,
            'cv_metrics': cv_metrics,
            'train_metrics': train_metrics,
            'feature_names': X_train_clean.columns.tolist()
        }
        
        best_params_dict[col] = best_params
        cv_metrics_dict[col] = cv_metrics
        train_metrics_dict[col] = train_metrics
    
    pipeline = {
        "models": models,  # ← ADD this line
        "best_params": best_params_dict,
        "cv_metrics": cv_metrics_dict,
        "train_metrics": train_metrics_dict,
        "features": X_train.columns.tolist(),
        "feature_selection_method": "ALL FEATURES",
        "pipeline_type": "all_features",
    }
    return pipeline