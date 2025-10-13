import optuna
from sklearn.metrics import mean_squared_error

def tune_xgb_model(X_train, Y_train, X_val, Y_val, train_func, n_trials=50):
    """
    Tune XGBoost hyperparameters using Optuna.
    Args:
        X_train, Y_train, X_val, Y_val: Training and validation data.
        train_func: Function to train and evaluate model (from model_training.py).
        n_trials (int): Number of Optuna trials.
    Returns:
        dict: Best hyperparameters.
        float: Best RMSE value.
    """
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'random_state': 42,
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 2.0)
        }
        # Handle three return values from train_func
        model, metrics, _ = train_func(X_train, Y_train, X_val, Y_val, log_to_clearml=False, **params)
        rmse = metrics['rmse']
        return rmse

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    print(f"Best params: {study.best_params}, Best RMSE: {study.best_value:.4f}")
    return study.best_params, study.best_value