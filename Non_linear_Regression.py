# python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from Data import fuel_intervals  # erwartet, dass `fuel_intervals` in Data.py erstellt wird

# Features / Ziel
features = ["mean_speed", "std_speed", "total_acc_m_s", "total_brake_m_s"]
target = "fuel_diff_ml"

df = fuel_intervals.copy()

# Nur vollständige Zeilen für Features + Ziel
df = df.dropna(subset=features + [target]).reset_index(drop=True)

X = df[features].values
y = df[target].values

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Sample weights: größere fuel_diff -> höhere Gewichtung (verringert Einfluss häufiger kleiner Werte)
y_shift = y_train - np.min(y_train) + 1.0
sample_weight = np.clip(y_shift, 1.0, None)

# Nicht-lineares Modell (Gradient Boosting)
model = GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=2)
model.fit(X_train, y_train, sample_weight=sample_weight)



y_pred = model.predict(X_test)

# Metriken
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

"""
print("Used rows:", len(df))
print("Features:", features)
print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)
print("R²:", r2)
"""

# Analyse für hohe Werte (z.B. >200 ml)
high_thresh = 200.0
high_mask = y_test > high_thresh
high_count = int(high_mask.sum())
high_mae = None
if high_count > 0:
    high_mae = mean_absolute_error(y_test[high_mask], y_pred[high_mask])
#print("High values (>{} ml): {}   High MAE: {}".format(int(high_thresh), high_count, high_mae))

"""
# Scatterplot: tatsächl. vs. vorhergesagt (gerundet)
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.7)
mn, mx = np.nanmin(y_test), np.nanmax(y_test)
plt.plot([mn, mx], [mn, mx], color="k", linestyle="--")
plt.xlabel("Actual fuel_diff_ml")
plt.ylabel("Predicted fuel_diff_ml (gerundet auf 10 ml)")
plt.title("Predicted vs Actual (GradientBoosting, gerundet)")
plt.grid(True)
plt.tight_layout()
plt.show()


# Feature Importances
importances = model.feature_importances_
plt.figure(figsize=(6,3))
plt.bar(features, importances, color="C2", alpha=0.8)
plt.ylabel("Feature importance")
plt.title("Feature Importances (GradientBoosting)")
plt.tight_layout()
plt.show()
"""