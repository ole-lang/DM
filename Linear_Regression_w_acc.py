from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
from Data import fuel_intervals


df = fuel_intervals.copy()

# only use rows without NaN values
df = df.dropna(subset=["mean_speed", "std_speed", "fuel_diff_ml"]).reset_index(drop=True)

X = df[["mean_speed", "std_speed","total_acc_m_s", "total_brake_m_s"]].values
y = df["fuel_diff_ml"].values

# Dividing the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions and metrics
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Used rows:", len(df))
print("Coefficients (mean_speed, std_speed, acc, brake):", model.coef_)
print("Intercept:", model.intercept_)
print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)
print("R²:", r2)


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
