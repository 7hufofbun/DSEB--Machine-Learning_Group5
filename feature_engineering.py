# để ý đoạn này là tạo file preprocess  (mọi người để ý path)
# from data_preprocessing import basic_transform, get_data
# data = get_data(r"https://raw.githubusercontent.com/7hufofbun/DSEB--Machine-Learning_Group5/refs/heads/main/data/weather_hcm_daily.csv")
# data = basic_transform(data)
# data.to_csv(r"F:\DSEB\Semester5\ML\Project\DSEB--Machine-Learning_Group5\data\preprocessed_data.csv", index=False)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt  # Optional for visualization

# Load preprocessed data (trỏ vào folder data không nó bị lỗi (k dùng link git được do chưa merge với main))
df = pd.read_csv("data\preprocessed_data.csv", parse_dates=["datetime"])

# Set index là datetime
df = df.set_index("datetime").sort_index()

# Create 5 future-day targets
for i in range(1, 6):
    df[f"temp_target_t+{i}"] = df["temp"].shift(-i)

# Drop rows where any of the 5 targets are NaN (last 5 rows)
df = df.dropna(subset=[f"temp_target_t+{i}" for i in range(1, 6)])

# Define y as all 5 targets
target_cols = [f"temp_target_t+{i}" for i in range(1, 6)]
y = df[target_cols]

print("Multi-output target variables defined:")
print(y.head())
print("\nShape of y:", y.shape)

# =====================================
# Step 1: Dimensionality Reduction
# Explanation: Dropping non-numeric or redundant columns is a good start, but I've 
# refined the list based on typical weather data insights. For example, 'precipprob' 
# and 'precipcover' might still be useful if precipitation affects temperature indirectly, 
# but we'll let feature selection decide later. We keep datetime for now to extract 
# features, then drop it.
# =====================================

# Keep a copy before dropping
df_reduced = df.copy()

# Drop text/non-numeric columns
text_cols = ["conditions", "description", "icon", "source"]
df_reduced = df_reduced.drop(columns=[c for c in text_cols if c in df_reduced.columns])

# Drop potentially redundant columns (based on domain knowledge and correlation)
redundant_cols = ["solarenergy", "uvindex", "moonphase"]  # Redundant or irrelevant
df_reduced = df_reduced.drop(columns=[c for c in redundant_cols if c in df_reduced.columns])

print("\nAfter initial dimensionality reduction:")
print("Remaining columns:", df_reduced.columns.tolist())
print("Shape:", df_reduced.shape)

# Separate X (features) from y (targets)
X = df_reduced.drop(columns=target_cols + ["datetime"])  # Drop targets and datetime for now

# =====================================
# Step 2: Add Lag Features
# Explanation: Lags capture autocorrelation in time series (e.g., yesterday's temp 
# influences today's). We've limited max_lag to 7, but you could optimize this via 
# autocorrelation plots (e.g., using pd.plotting.lag_plot or statsmodels.acf). 
# Engineered features like lags often outperform originals because they explicitly 
# model temporal dependencies, which models like RF might not capture otherwise. 
# This is generally good for the model as it provides more relevant signals, but 
# watch for multicollinearity (high correlation between lags).
# =====================================

lag_features = ["temp", "humidity", "solarradiation", "cloudcover"]
max_lag = 7

for col in lag_features:
    for lag in range(1, max_lag + 1):
        X[f"{col}_lag{lag}"] = df_reduced[col].shift(lag)

# Drop NaNs introduced by shifts
X = X.dropna().reset_index(drop=True)
y = y.iloc[X.index].reset_index(drop=True)  # Align y with X

print(f"\nAdded lag features up to {max_lag} days for: {lag_features}")
print("New X shape:", X.shape)

# =====================================
# Step 3: Add Rolling Statistics
# Explanation: Rolling means/std capture trends and volatility over windows. 
# We've kept windows [3,7] as short-term trends are useful for weather. These 
# can be "better" than originals because they smooth noise and highlight patterns 
# (e.g., weekly averages). Good for models, but larger windows might leak future 
# info if not careful—here, since we're using past data only, it's fine.
# =====================================

rolling_features = ["temp", "humidity", "solarradiation"]
windows = [3, 7]

for col in rolling_features:
    for w in windows:
        X[f"{col}_rollmean_{w}"] = df_reduced[col].rolling(window=w).mean().shift(1) 
        X[f"{col}_rollstd_{w}"] = df_reduced[col].rolling(window=w).std().shift(1)

# Drop NaNs from rolling
X = X.dropna().reset_index(drop=True)
y = y.iloc[X.index].reset_index(drop=True)

print("\nAdded rolling mean/std features for:", rolling_features)
print("New X shape:", X.shape)

# =====================================
# Step 4: Add Time-Based Features
# Explanation: These capture seasonality (e.g., month affects temp). We've added 
# cyclical encoding for month/dayofweek to handle periodicity (e.g., Dec close to Jan). 
# This is an optimization: sine/cosine transformations make models understand cycles better.
# =====================================

# Align datetime with current X index
datetime_aligned = df["datetime"].iloc[X.index]

X["month"] = datetime_aligned.dt.month
X["day"] = datetime_aligned.dt.day
X["dayofweek"] = datetime_aligned.dt.dayofweek

# Cyclical encoding for better modeling of periodicity
X["month_sin"] = np.sin(2 * np.pi * X["month"] / 12)
X["month_cos"] = np.cos(2 * np.pi * X["month"] / 12)
X["dayofweek_sin"] = np.sin(2 * np.pi * X["dayofweek"] / 7)
X["dayofweek_cos"] = np.cos(2 * np.pi * X["dayofweek"] / 7)

# Drop original non-cyclical if desired (but keep for interpretability)
# X = X.drop(columns=["month", "dayofweek"])

print("\nAdded time-based features with cyclical encoding: month, day, dayofweek, sins/cos")

# =====================================
# Step 5: Add Interaction Features
# Explanation: Interactions capture combined effects (e.g., high humidity + high temp = discomfort, 
# but for temp prediction, solar * (1-cloud) models effective sunlight). These can boost 
# performance if non-linear relationships exist. They're often "better" because base features 
# alone might not capture synergies.
# =====================================

X["solar_cloud_interact"] = X["solarradiation"] * (1 - X["cloudcover"] / 100)
X["humidity_temp_interact"] = X["humidity"] * X["temp"]

X = X.dropna().reset_index(drop=True)
y = y.iloc[X.index].reset_index(drop=True)

print("\nAdded interaction features: solar_cloud_interact, humidity_temp_interact")
print("Final X shape before selection:", X.shape)
print("y shape:", y.shape)

# =====================================
# Step 6: Feature Selection with Random Forest
# Explanation: RF feature importance is good, but we've improved by:
# - Using TimeSeriesSplit for CV to respect time order (prevents leakage).
# - Embedded selection with SelectFromModel (automatically selects based on importance).
# - Added cross-validation to evaluate if selected features improve model performance.
# - Why engineered > original? Engineered features encode domain-specific patterns 
#   (time, trends), making them more predictive. This is beneficial for the model 
#   as long as we avoid overfitting—use CV to check.
# =====================================

# Select only numeric columns
X = X.select_dtypes(include=[np.number])

# Time-based split (80% train, 20% test, chronological)
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Train RF for importance
rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Get average importances
feature_importances = np.mean([tree.feature_importances_ for tree in rf.estimators_], axis=0)

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": feature_importances
}).sort_values(by="Importance", ascending=False)

print("\nTop 15 Important Features:")
print(importance_df.head(15))

# Visualize (optional)
plt.figure(figsize=(10, 6))
plt.barh(importance_df["Feature"].head(15), importance_df["Importance"].head(15))
plt.gca().invert_yaxis()
plt.title("Top 15 Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.show()

# Embedded feature selection (auto-select based on mean importance)
selector = SelectFromModel(rf, prefit=True)
X_selected = selector.transform(X)
selected_features = X.columns[selector.get_support()].tolist()

print("\nSelected important features:")
print(selected_features)
print("New X_selected shape:", X_selected.shape)

# =====================================
# Step 7: Evaluate Optimization
# Explanation: To check if new features are "good," we use cross-validation with 
# TimeSeriesSplit. Compare MSE with/without engineered features. If engineered 
# lower MSE, they're beneficial. This prevents blindly adding features.
# =====================================

# Re-split selected features
X_selected_train, X_selected_test = pd.DataFrame(X_selected[:split_idx], columns=selected_features), pd.DataFrame(X_selected[split_idx:], columns=selected_features)

# Baseline: Train RF on selected features
rf_selected = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf_selected.fit(X_selected_train, y_train)

# Predict and evaluate
y_pred = rf_selected.predict(X_selected_test)
mse = mean_squared_error(y_test, y_pred)
print(f"\nMSE on test set with selected features: {mse:.4f}")

# Optional: Compare to original features only (no engineered)
original_cols = [col for col in X.columns if not any(s in col for s in ["lag", "roll", "interact", "sin", "cos"])]
X_original = X[original_cols]
X_orig_train, X_orig_test = X_original.iloc[:split_idx], X_original.iloc[split_idx:]

rf_orig = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf_orig.fit(X_orig_train, y_train)
y_pred_orig = rf_orig.predict(X_orig_test)
mse_orig = mean_squared_error(y_test, y_pred_orig)
print(f"MSE on test set with original features only: {mse_orig:.4f}")

# Cross-validation for robustness
tscv = TimeSeriesSplit(n_splits=5)
cv_scores = cross_val_score(rf_selected, X_selected_train, y_train, cv=tscv, scoring="neg_mean_squared_error")
print(f"\nCV MSE (selected features): {-cv_scores.mean():.4f}")

# =====================================
# Step 8: Save Results
# Explanation: Export as before, but now with selected features only.
# =====================================

importance_df.to_csv("outputs/feature_importance.csv", index=False)
pd.DataFrame(X_selected, columns=selected_features).to_csv("outputs/X_selected.csv", index=False)
y.to_csv("outputs/y_targets.csv", index=False)

print("\nExported:")
print(" - feature_importance.csv")
print(" - X_selected.csv")
print(" - y_targets.csv")


