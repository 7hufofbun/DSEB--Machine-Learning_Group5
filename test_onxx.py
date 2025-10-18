import onnxruntime as rt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

# Load your data
X_test = pd.read_csv('outputs/X_features.csv', parse_dates=['datetime'])
y_test = pd.read_csv('outputs/Y_target.csv', parse_dates=['datetime'])

# Align and process the same way as before
data = pd.merge(X_test, y_test, on='datetime', how='inner')
data = data.sort_values('datetime').set_index('datetime')
data['temp_next_day'] = data['temp'].shift(-1)
data = data.dropna()
features = data.columns.drop(['temp', 'temp_next_day'])
X = data[features]
y_shifted = data['temp_next_day']
split_idx = int(0.8 * len(X))
X_test = X.iloc[split_idx:]
y_test = y_shifted.iloc[split_idx:]

# Load ONNX model
sess = rt.InferenceSession("model.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

# Predict
onnx_pred = sess.run([label_name], {input_name: X_test.values.astype(np.float32)})[0]

# Metrics
rmse = np.sqrt(mean_squared_error(y_test, onnx_pred))
mape = mean_absolute_percentage_error(y_test, onnx_pred)
r2 = r2_score(y_test, onnx_pred)

print(f"ONNX model metrics:\nRMSE: {rmse:.2f}, MAPE: {mape:.2%}, R2: {r2:.2f}")
