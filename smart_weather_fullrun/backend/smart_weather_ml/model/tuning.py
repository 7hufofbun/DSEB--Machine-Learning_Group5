
import numpy as np
import optuna
from sklearn.model_selection import TimeSeriesSplit
from ..metrics import calculate_all_metrics
from .lgbm import get_model, get_model_params
from ..features import feature_engineer
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

def objective_all_features(trial, X_train, y_train, target_name, preprocessor):
    params = get_model_params(trial)
    all_train_metrics = {'mae': [], 'rmse': [], 'r2': [], 'mse': []}
    X_final, y_final = X_train.copy(), y_train.copy()
    tscv = TimeSeriesSplit(n_splits=4)
    all_metrics = {'mae': [], 'rmse': [], 'r2': [], 'mse': []}
    for train_idx, val_idx in tscv.split(X_final):
        X_tr_r, X_val_r = X_final.iloc[train_idx], X_final.iloc[val_idx]
        y_tr_r, y_val_r = y_final.iloc[train_idx], y_final.iloc[val_idx]
        X_tr_p = preprocessor.fit_transform(X_tr_r)
        X_val_p = preprocessor.transform(X_val_r)
        X_tr, y_tr_fe = feature_engineer(X_tr_p, y_tr_r)
        X_val, y_val_fe = feature_engineer(X_val_p, y_val_r)
        y_tr = y_tr_fe[target_name]
        y_val = y_val_fe[target_name]
        model = get_model( params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
            eval_metric='rmse',
            callbacks=[
                early_stopping(stopping_rounds=200, verbose=False),
                log_evaluation(0)  # Không hiển thị log
            ])
        
        y_pred_train = model.predict(X_tr)
        train_fold_metrics = calculate_all_metrics(y_tr, y_pred_train)
        for k, v in train_fold_metrics.items():
            all_train_metrics[k].append(v)

        y_pred = model.predict(X_val)
        metrics = calculate_all_metrics(y_val, y_pred)
        for k, v in metrics.items():
            all_metrics[k].append(v)

    mean_train_metrics = {k: np.mean(v) for k, v in all_train_metrics.items()}
    trial.set_user_attr('train_metrics', mean_train_metrics)
    mean_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    trial.set_user_attr('metrics', mean_metrics)

    score = mean_metrics['rmse'] * (1.0 + 0.3 * (1 - mean_metrics['r2']))
    return score

def optimize_model_all_features(X_train, y_train, target_name,  preprocessor, n_trials=50):
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    study = optuna.create_study(direction='minimize', sampler=sampler, pruner=pruner)
    study.optimize(
        lambda trial: objective_all_features(trial, X_train, y_train, target_name, preprocessor),
        n_trials=n_trials,
        show_progress_bar=True
    )
    best_metrics = study.best_trial.user_attrs['metrics']
    train_metrics = study.best_trial.user_attrs.get('train_metrics', None)
    return study.best_params, best_metrics, train_metrics

