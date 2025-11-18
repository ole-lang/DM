import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

from Data import fuel_intervals as data

class RandomForestModel:
    def __init__(self, n_estimators, max_depth, random_state, 
    min_samples_split, min_samples_leaf, max_features):
        self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, 
        min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)
    
    # TODO remove tensor flow
    
'''
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

# Make predictions
y_pred = model.predict(X_test)


# rebuild a dataframe to analyze predictions temporally
results = X_test.copy()
results["actual"] = y_test
results["predicted"] = y_pred
results["start_time"] = df_features.loc[X_test.index, "start_time"]
results["end_time"] = df_features.loc[X_test.index, "end_time"]
results["start_time"] = pd.to_datetime(results["start_time"])
results = results.sort_values("start_time")
results = results.set_index("start_time")

# Aggregate by 30-minute windows
agg_30min = results.resample("30T").sum(numeric_only=True)[["actual", "predicted"]]


# Evaluate the model
print(f"R² = {r2_score(y_test, y_pred):.3f}")
print(f"MAE = {mean_absolute_error(y_test, y_pred):.3f} liters")
print(f"RMSE = {np.sqrt(mean_squared_error(y_test, y_pred)):.3f} liters")

r2_agg = r2_score(agg_30min["actual"], agg_30min["predicted"])
mae_agg = mean_absolute_error(agg_30min["actual"], agg_30min["predicted"])
rmse_agg = np.sqrt(mean_squared_error(agg_30min["actual"], agg_30min["predicted"]))

print(f"Aggregated (30-min) R² = {r2_agg:.3f}")
print(f"Aggregated (30-min) MAE = {mae_agg:.3f} ml")
print(f"Aggregated (30-min) RMSE = {rmse_agg:.3f} ml")

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

# Time series plot of actual vs predicted fuel consumption in 30-min windows
plt.figure(figsize=(10,5))
plt.plot(agg_30min.index, agg_30min["actual"], label="Actual (30 min total)", marker='o')
plt.plot(agg_30min.index, agg_30min["predicted"], label="Predicted (30 min total)", marker='x')
plt.xlabel("Time")
plt.ylabel("Total fuel consumption (ml)")
plt.title("Actual vs Predicted Fuel Consumption (30-min windows)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# plot fuel consumption over time
data['start_time'] = pd.to_datetime(data['start_time'])
data = data.sort_values('start_time')
plt.plot(data['start_time'], data['fuel_diff_ml'], marker='o')
plt.xlabel("Time")
plt.ylabel("Fuel Consumption (ml)")
plt.title("Fuel Consumption Over Time")
plt.show()'''
