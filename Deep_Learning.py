# python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from Data import fuel_intervals

# Konfiguration
features = ["mean_speed", "std_speed", "total_acc_m_s", "total_brake_m_s"]
target = "fuel_diff_ml"
use_log_target = False   # True = log1p transformieren, dann inverse vor dem Runden
random_state = 42
test_size = 0.2

# Daten vorbereiten
df = fuel_intervals.copy().dropna(subset=features + [target]).reset_index(drop=True)
X = df[features].values.astype(np.float32)
y = df[target].values.astype(np.float32)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# Optionale Zieltransformation (stabilisiert bei starker Schiefe)
if use_log_target:
    y_train_t = np.log1p(np.clip(y_train, a_min=0.0, a_max=None))
else:
    y_train_t = y_train.copy()

# Sample weights: größere Verbrauchswerte stärker gewichten
shift = np.maximum(y_train, 0.0)  # nur nicht-negative
sample_weight = (shift - shift.min()) + 1.0
sample_weight = np.clip(sample_weight, 1.0, None)

# Normalisierung
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Modellaufbau (kleines MLP, Dropout + BatchNorm optional)
def build_model(input_dim):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="linear")(x)
    m = models.Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss="mse",
              metrics=["mae"])
    return m

model = build_model(X_train_s.shape[1])

# Callbacks
es = callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=0)
rlr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, verbose=0)

# Training
history = model.fit(
    X_train_s, y_train_t,
    sample_weight=sample_weight,
    validation_split=0.1,
    epochs=200,
    batch_size=32,
    callbacks=[es, rlr],
    verbose=1
)

# Vorhersage und Rücktransformation
y_pred_t = model.predict(X_test_s).reshape(-1)
if use_log_target:
    y_pred = np.expm1(np.clip(y_pred_t, a_min=None, a_max=None))
else:
    y_pred = y_pred_t



# Metriken
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Analyse hoher Werte
high_thresh = 200.0
high_mask = y_test > high_thresh
high_count = int(high_mask.sum())
high_mae = mean_absolute_error(y_test[high_mask], y_pred[high_mask]) if high_count > 0 else None

print("Used rows:", len(df))
print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)
print("R²:", r2)
print("High values (>{} ml): {}  High MAE: {}".format(int(high_thresh), high_count, high_mae))

# Scatterplot
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.7)
mn, mx = np.nanmin(y_test), np.nanmax(y_test)
plt.plot([mn, mx], [mn, mx], color="k", linestyle="--")
plt.xlabel("Actual fuel_diff_ml")
plt.ylabel("Predicted fuel_diff_ml")
plt.title("Deep Learning: Predicted vs Actual (gerundet)")
plt.grid(True)
plt.tight_layout()
plt.show()

"""
# Modell speichern + Scaler
model.save("deep_model.h5")
import joblib
joblib.dump(scaler, "scaler.save")
"""