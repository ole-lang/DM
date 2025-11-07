# create a mullti-layer perceptron model for fuel consumption prediction
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras import layers

from Data import fuel_intervals as data

# Drop NaN rows
# environment variable: TF_ENABLE_ONEDNN_OPTS=0
df_features = data.dropna()

# Define features and target
X = df_features.drop(columns=["fuel_diff_ml", "start_time", "end_time"], errors='ignore')
y = df_features["fuel_diff_ml"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features (important for neural networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define model
# Define model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)  # output = predicted fuel_used
])
# Compile model
model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mae']
)

model.summary()

history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=16,
    verbose=1
)

test_loss, test_mae = model.evaluate(X_test_scaled, y_test)

# Predictions
y_pred = model.predict(X_test_scaled).flatten()
print(f"Test MAE: {test_mae:.4f}")
print(f"Test Loss (MSE): {test_loss:.4f}")
# print(f"R2 Score: {keras.metrics.R2Score()(y_test, y_pred).numpy():.4f}")


plt.scatter(y_test, y_pred)
plt.xlabel("Actual fuel used")
plt.ylabel("Predicted fuel used")
plt.title("MLP predictions vs actual")
plt.show()
