### API for ClearML (đầu tiên chạy clearml-init trong terminal xong paste đoạn dưới này vào)
# api {
#  web_server: https://app.clear.ml/
#  api_server: https://api.clear.ml
#  files_server: https://files.clear.ml
#  credentials {
#    "access_key" = "NU3OBBA64349VUUFSSIMQKZCYCBLWF"
#    "secret_key" = "ulpmq-JS2iHoB7TIvkfRaTQsVkaAwiwm7iyweH7GqpVZ_2xeU9jaHnmQO1ppoKh0h_c"
#  }
# }

from clearml import Task, Logger
import matplotlib.pyplot as plt
import os
import joblib

def init_clearml_task(project_name, task_name, hyperparams=None):
    """
    Initialize a ClearML task and connect hyperparameters.
    Args:
        project_name (str): ClearML project name (e.g., 'HCM_Temp_Forecast').
        task_name (str): Task name (e.g., 'Daily_XGB_1Day').
        hyperparams (dict): Model hyperparameters to log.
    Returns:
        task: Initialized ClearML task object.
    """
    task = Task.init(project_name=project_name, task_name=task_name)
    task.set_base_docker("python:3.9")  # For reproducibility
    if hyperparams:
        task.connect(hyperparams)  # Log hyperparameters
    return task

def log_metrics(metrics, iteration=0):
    """
    Log metrics (e.g., RMSE, MAPE, R2) to ClearML.
    Args:
        metrics (dict): Dictionary of metric names and values (e.g., {'rmse': 1.2, 'mape': 0.03}).
        iteration (int): Training iteration or epoch.
    """
    logger = Logger.current_logger()
    for metric_name, value in metrics.items():
        logger.report_scalar(title="Validation Metrics", series=metric_name.upper(), value=value, iteration=iteration)

def log_model_artifact(model, model_name, output_dir='outputs/models'):
    """
    Save and log model as a ClearML artifact.
    Args:
        model: Trained model object.
        model_name (str): Name for model file (e.g., 'xgb_model.joblib').
        output_dir (str): Directory to save model.
    """
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, model_name)
    joblib.dump(model, model_path)
    task = Task.current_task()
    task.upload_artifact(model_name, model_path)

def log_prediction_plot(y_true, y_pred, plot_name='val_plot.png', title="Val Predictions", series="Temp Forecast"):
    """
    Log a plot of actual vs predicted values to ClearML.
    Args:
        y_true: Actual values (array-like).
        y_pred: Predicted values (array-like).
        plot_name (str): Filename for plot.
        title (str): Plot title in ClearML.
        series (str): Plot series name in ClearML.
    """
    plt.figure()
    plt.plot(y_true, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.legend()
    plt.title("Actual vs Predicted Temperature")
    plt.savefig(plot_name)
    logger = Logger.current_logger()
    logger.report_matplotlib_figure(title=title, series=series, figure=plt.gcf(), iteration=0)
    plt.close()

def close_clearml_task():
    """Close the current ClearML task to save logs."""
    task = Task.current_task()
    if task:
        task.close()