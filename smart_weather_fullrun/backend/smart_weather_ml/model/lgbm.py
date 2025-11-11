
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from ..metrics import calculate_all_metrics
def get_model(best_params):
    return LGBMRegressor(
        n_estimators=best_params.get("n_estimators", 400),
        max_depth=best_params.get("max_depth", 5),
        num_leaves=best_params.get("num_leaves", 63),
        min_data_in_leaf=best_params.get("min_data_in_leaf", 20),
        learning_rate=best_params.get("learning_rate", 0.01),
        feature_fraction=best_params.get("feature_fraction", 0.3),
        bagging_fraction=best_params.get("bagging_fraction", 0.7),
        bagging_freq=best_params.get("bagging_freq", 5),
        lambda_l1=best_params.get("lambda_l1", 0.1),
        lambda_l2=best_params.get("lambda_l2", 0.1),
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
def get_model_params(trial, target_name):
    _ = int(target_name.split("+")[-1])
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "num_leaves": trial.suggest_int("num_leaves", 10, 50),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.03, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 0.4),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 0.7),
        "bagging_freq": trial.suggest_int("bagging_freq", 2, 7),
        "lambda_l1": trial.suggest_float("lambda_l1", 12, 50.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 10, 50.0, log=True),
    }
def train_final_model_all_features(X_train, y_train, best_params):
    X_train_final, y_train_final = X_train.copy(), y_train.copy()
    val_size = max(1, int(len(X_train_final) * 0.1))
    X_tr = X_train_final.iloc[:-val_size]
    y_tr = y_train_final.iloc[:-val_size]
    X_val = X_train_final.iloc[-val_size:]
    y_val = y_train_final.iloc[-val_size:]
    model = get_model(best_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[early_stopping(stopping_rounds=200, verbose=False), log_evaluation(0)],
    )
    y_pred_train = model.predict(X_train_final)
    train_metrics = calculate_all_metrics(y_train_final, y_pred_train)
    return {
        "model": model,
        "train_metrics": train_metrics,
        "feature_names": list(X_train_final.columns),
        "best_params": best_params,
        "n_features": X_train_final.shape[1],
    }
