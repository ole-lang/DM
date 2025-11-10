import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

from Data import fuel_intervals as data

# Drop NaNs 
# TODO; what else can i adjust in order to improve the algorithm
df_features = data

# Features and target variable
df_features["duration_s"] = (df_features["end_time"] - df_features["start_time"]).dt.total_seconds()
X = df_features.drop(columns=["start_time", "end_time", "fuel_diff_ml"])
y = df_features["fuel_diff_ml"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, min_samples_split=5, min_samples_leaf=2, max_features='sqrt')
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f"R² = {r2_score(y_test, y_pred):.3f}")
print(f"MAE = {mean_absolute_error(y_test, y_pred):.3f} liters")
print(f"RMSE = {np.sqrt(mean_squared_error(y_test, y_pred)):.3f} liters")

# simple Scatter-Plot
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="k", linestyle="--")
plt.xlabel("Actual fuel_diff_ml")
plt.ylabel("Predicted fuel_diff_ml")
plt.title("Predicted vs. Actual")
plt.grid(True)
plt.tight_layout()
plt.show()

'''
# plot fuel consumption over time
data['start_time'] = pd.to_datetime(data['start_time'])
data = data.sort_values('start_time')
plt.plot(data['start_time'], data['fuel_diff_ml'], marker='o')
plt.xlabel("Time")
plt.ylabel("Fuel Consumption (ml)")
plt.title("Fuel Consumption Over Time")
plt.show()'''
