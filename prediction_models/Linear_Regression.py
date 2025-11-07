import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

from Data import df_model

X = df_model[["gps_speed"]].values.reshape(-1, 1)
y = df_model["fuel_rate"].values
model = LinearRegression()
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

# Modellparameters
a = model.coef_[0]
b = model.intercept_

print(f"Regression: fuel_rate = {a:.4f} * gps_speed + {b:.4f}")

# Scores
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

print(f"R² = {r2:.3f}")
print(f"MAE = {mae:.3f} ml/s")
print(f"RMSE = {rmse:.3f} ml/s")

plt.figure(figsize=(8,5))
plt.scatter(X, y, alpha=0.5, label="Datapoints")
plt.plot(X, y_pred, color="red", label="Linear Regression")
plt.xlabel("GPS-Speed (km/h)")
plt.ylabel("Fuel Rate (ml/s)")
plt.title("Linear Regression: GPS-Speed → Fuel Rate")
plt.legend()
plt.show()
