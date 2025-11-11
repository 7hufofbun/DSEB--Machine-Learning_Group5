
import numpy as np
import optuna
from sklearn.model_selection import TimeSeriesSplit
from ..metrics import calculate_all_metrics
from .lgbm import get_model, get_model_params
def objective_all_features(trial, X_train, y_train, target_name):
    params = get_model_params(trial, target_name)
    X_final, y_final = X_train.copy(), y_train.copy()
    tscv = TimeSeriesSplit(n_splits=4)
    all_metrics = {"mae": [], "rmse": [], "r2": [], "mse": []}
    for tr_idx, val_idx in tscv.split(X_final):
        X_tr, X_val = X_final.iloc[tr_idx], X_final.iloc[val_idx]
        y_tr, y_val = y_final.iloc[tr_idx], y_final.iloc[val_idx]
        model = get_model(params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="rmse", callbacks=[])
        y_pred = model.predict(X_val)
        metrics = calculate_all_metrics(y_val, y_pred)
        for k, v in metrics.items(): all_metrics[k].append(v)
    mean_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    trial.set_user_attr("metrics", mean_metrics)
    score = mean_metrics["rmse"] * (1.0 + 0.3 * (1 - mean_metrics["r2"]))
    return score
def optimize_model_all_features(X_train, y_train, target_name, n_trials=30):
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(lambda t: objective_all_features(t, X_train, y_train, target_name), n_trials=n_trials, show_progress_bar=True)
    best_metrics = study.best_trial.user_attrs["metrics"]
    return study.best_params, best_metrics
