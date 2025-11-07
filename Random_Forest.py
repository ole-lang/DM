import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
