
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
def calculate_all_metrics(y_true, y_pred):
    return {
        "mae":  mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2":   r2_score(y_true, y_pred),
        "mse":  mean_squared_error(y_true, y_pred),
    }
