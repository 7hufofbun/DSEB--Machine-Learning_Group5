import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from tuning_utils import tune_xgb_model
from logging_utils import init_clearml_task, log_metrics, log_model_artifact, log_prediction_plot, close_clearml_task
import os
import joblib
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Load X and Y (from feature engineering, Step 4)
X = pd.read_csv('outputs/X_features.csv', index_col=0, parse_dates=[0])
Y = pd.read_csv('outputs/Y_target.csv', index_col=0, parse_dates=[0])

# Adjust Y for 1-day ahead forecasting
Y = Y.shift(-1).dropna()  # Next day's mean temperature
X = X.loc[Y.index]  # Align X with Y

# Chronological split: 70% train, 15% val, 15% test
n = len(X)
train_size = int(0.7 * n)
val_size = int(0.15 * n)
X_train = X.iloc[:train_size]
Y_train = Y.iloc[:train_size]
X_val = X.iloc[train_size:train_size + val_size]
Y_val = Y.iloc[train_size:train_size + val_size]
X_test = X.iloc[train_size + val_size:]
Y_test = Y.iloc[train_size + val_size:]

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Compute metrics function
def compute_metrics(y_true, y_pred):
    """
    Compute evaluation metrics for regression.
    Args:
        y_true: Actual values.
        y_pred: Predicted values.
    Returns:
        dict: RMSE, MAPE, R2 metrics.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'rmse': rmse, 'mape': mape, 'r2': r2}

# Train XGB model
def train_xgb_model(X_train, Y_train, X_val, Y_val, task_name='Daily_XGB_1Day', log_to_clearml=True, **xgb_kwargs):
    """
    Train XGBoost model with given parameters.
    Args:
        X_train, Y_train, X_val, Y_val: Training and validation data.
        task_name (str): ClearML task name for logging.
        log_to_clearml (bool): Whether to log to ClearML.
        xgb_kwargs: XGBoost hyperparameters.
    Returns:
        Pipeline: Trained model pipeline.
        dict: Validation metrics.
        array: Validation predictions.
    """
    if log_to_clearml:
        task = init_clearml_task(project_name='HCM_Temp_Forecast', task_name=task_name, hyperparams=xgb_kwargs)
    
    # Create pipeline with scaling and model
    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('model', XGBRegressor(**xgb_kwargs))
    ])
    pipeline.fit(X_train, Y_train)
    
    # Predict and compute metrics
    preds_val = pipeline.predict(X_val)
    metrics = compute_metrics(Y_val, preds_val)
    
    if log_to_clearml:
        # Log metrics, model, and prediction plot to ClearML
        log_metrics(metrics)
        log_model_artifact(pipeline, f'xgb_model_{task_name.lower()}.joblib')
        log_prediction_plot(Y_val.values, preds_val, f'xgb_val_plot_{task_name.lower()}.png', 
                          title=f"{task_name} Predictions", series="Temp Forecast")
        close_clearml_task()
    
    return pipeline, metrics, preds_val

# Compare pre- and post-tuning metrics
def compare_models(pre_metrics, post_metrics, set_name="Validation"):
    """
    Print comparison of pre- and post-tuning metrics (R2, RMSE, MAPE).
    Args:
        pre_metrics (dict): Metrics for pre-tuning model.
        post_metrics (dict): Metrics for post-tuning model.
        set_name (str): Name of dataset (e.g., Validation, Test).
    """
    print(f"\n{set_name} Metrics Comparison (Pre-Tuning vs Post-Tuning):")
    print(f"{'Metric':<10} {'Pre-Tuning':<12} {'Post-Tuning':<12} {'Improvement':<12}")
    print("-" * 46)
    for metric in ['rmse', 'mape', 'r2']:
        pre_value = pre_metrics[metric]
        post_value = post_metrics[metric]
        # Improvement: negative for RMSE/MAPE (lower is better), positive for R2 (higher is better)
        improvement = (pre_value - post_value) / pre_value * 100 if metric != 'r2' else (post_value - pre_value) / pre_value * 100
        print(f"{metric.upper():<10} {pre_value:.4f} {'':<6} {post_value:.4f} {'':<6} {improvement:.2f}%")

# Plot metrics comparison
def plot_metrics_comparison(pre_metrics_val, post_metrics_val, pre_metrics_test, post_metrics_test):
    """
    Plot bar chart comparing R2, RMSE, MAPE for pre- and post-tuning models (validation and test sets).
    Args:
        pre_metrics_val, post_metrics_val: Validation metrics for pre- and post-tuning.
        pre_metrics_test, post_metrics_test: Test metrics for pre- and post-tuning.
    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    metrics = ['RMSE', 'MAPE', 'R2']
    pre_values_val = [pre_metrics_val['rmse'], pre_metrics_val['mape'], pre_metrics_val['r2']]
    post_values_val = [post_metrics_val['rmse'], post_metrics_val['mape'], post_metrics_val['r2']]
    pre_values_test = [pre_metrics_test['rmse'], pre_metrics_test['mape'], pre_metrics_test['r2']]
    post_values_test = [post_metrics_test['rmse'], post_metrics_test['mape'], post_metrics_test['r2']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Validation set
    ax1.bar(x - width/2, pre_values_val, width, label='Pre-Tuning', color='#ff7f0e')
    ax1.bar(x + width/2, post_values_val, width, label='Post-Tuning', color='#2ca02c')
    ax1.set_title('Validation Set Metrics')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.set_ylabel('Value')
    ax1.legend()
    
    # Test set
    ax2.bar(x - width/2, pre_values_test, width, label='Pre-Tuning', color='#ff7f0e')
    ax2.bar(x + width/2, post_values_test, width, label='Post-Tuning', color='#2ca02c')
    ax2.set_title('Test Set Metrics')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.set_ylabel('Value')
    ax2.legend()
    
    plt.tight_layout()
    return fig

if __name__ == '__main__':
    print('\n--- Training Pre-Tuning Model (Default Params) ---')
    # Default parameters for pre-tuning
    default_params = {
        'n_estimators': 100,
        'learning_rate': 0.05,
        'max_depth': 5,
        'random_state': 42
    }
    pre_model, pre_metrics, pre_preds_val = train_xgb_model(
        X_train, Y_train, X_val, Y_val, 
        task_name='Daily_XGB_PreTuning', 
        log_to_clearml=True, 
        **default_params
    )
    
    print('\n--- Hyperparameter Tuning with Optuna ---')
    # Run Optuna tuning
    best_params, best_rmse = tune_xgb_model(X_train, Y_train, X_val, Y_val, train_xgb_model, n_trials=20)
    
    print('\n--- Training Post-Tuning Model (Best Params) ---')
    # Train post-tuning model with best params
    best_params['random_state'] = 42
    post_model, post_metrics, post_preds_val = train_xgb_model(
        X_train, Y_train, X_val, Y_val, 
        task_name='Daily_XGB_PostTuning', 
        log_to_clearml=True, 
        **best_params
    )
    
    # Compare metrics on validation set
    compare_models(pre_metrics, post_metrics, set_name="Validation")
    
    # Evaluate both models on test set
    pre_preds_test = pre_model.predict(X_test)
    post_preds_test = post_model.predict(X_test)
    pre_test_metrics = compute_metrics(Y_test, pre_preds_test)
    post_test_metrics = compute_metrics(Y_test, post_preds_test)
    
    # Compare metrics on test set
    compare_models(pre_test_metrics, post_test_metrics, set_name="Test")
    # --- Log Comparison Table to ClearML ---

    # Create a summary DataFrame for validation and test metrics
    comparison_data = {
        'Metric': ['RMSE', 'MAPE', 'R2'],
        'Pre-Tuning (Val)': [pre_metrics['rmse'], pre_metrics['mape'], pre_metrics['r2']],
        'Post-Tuning (Val)': [post_metrics['rmse'], post_metrics['mape'], post_metrics['r2']],
        'Pre-Tuning (Test)': [pre_test_metrics['rmse'], pre_test_metrics['mape'], pre_test_metrics['r2']],
        'Post-Tuning (Test)': [post_test_metrics['rmse'], post_test_metrics['mape'], post_test_metrics['r2']],
    }

    comparison_df = pd.DataFrame(comparison_data)

    # Compute improvement percentages
    comparison_df['Val_Improvement (%)'] = [
        (pre_metrics['rmse'] - post_metrics['rmse']) / pre_metrics['rmse'] * 100,
        (pre_metrics['mape'] - post_metrics['mape']) / pre_metrics['mape'] * 100,
        (post_metrics['r2'] - pre_metrics['r2']) / pre_metrics['r2'] * 100
    ]
    comparison_df['Test_Improvement (%)'] = [
        (pre_test_metrics['rmse'] - post_test_metrics['rmse']) / pre_test_metrics['rmse'] * 100,
        (pre_test_metrics['mape'] - post_test_metrics['mape']) / pre_test_metrics['mape'] * 100,
        (post_test_metrics['r2'] - pre_test_metrics['r2']) / pre_test_metrics['r2'] * 100
    ]

    # Start a new ClearML task for the table
    table_task = init_clearml_task(project_name='HCM_Temp_Forecast', task_name='Metrics_Comparison_Table')

    # Log table as an artifact (CSV)
    comparison_csv_path = 'outputs/metrics_comparison_table.csv'
    comparison_df.to_csv(comparison_csv_path, index=False)
    log_model_artifact(comparison_csv_path, 'metrics_comparison_table.csv')

    # Or log directly as a table visualization
    logger = table_task.get_logger()
    logger.report_table(
        title='Pre vs Post Tuning Metrics Comparison',
        series='Metrics Summary',
        iteration=0,
        table_plot=comparison_df
    )

    close_clearml_task()
    
    # Save final model (post-tuning)
    os.makedirs('outputs/models', exist_ok=True)
    joblib.dump(post_model, 'outputs/models/xgb_final.joblib')

    print('\nAll done. Models saved to outputs/models/. View logs at: https://app.clear.ml')