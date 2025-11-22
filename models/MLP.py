# create a mullti-layer perceptron model for fuel consumption prediction

import keras
from keras import layers
from sklearn.preprocessing import StandardScaler

class MLPFuelModel():
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

    def build_model(self, input_dim):
        """Define and compile the Keras model."""
        model = keras.Sequential([
<<<<<<< HEAD
            keras.Input(shape=(input_dim,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
=======
            layers.Dense(64, activation='relu', input_shape=(input_dim,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)  # Output layer
>>>>>>> 2eb7e5b682a4028eae483acd4a99603b46789ddf
        ])
        model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['mae']
        )
        return model

    def train(self, X_train, y_train, epochs=100, batch_size=16, validation_split=0.2, verbose=1):
        """Train the MLP on scaled data."""
        # Scale the data
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Build model if not already built
        if self.model is None:
            self.model = self.build_model(X_train_scaled.shape[1])

        # Train model
        history = self.model.fit(
            X_train_scaled, y_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose
        )

        # return history

    def predict(self, X_test):
        """Make predictions on new data."""
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled).flatten()
        return y_pred

'''
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

# Define model - adjust amount of neurons per layers and layers
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
print(f"R² = {r2_score(y_test, y_pred):.3f}")


plt.scatter(y_test, y_pred)
plt.xlabel("Actual fuel used")
plt.ylabel("Predicted fuel used")
plt.title("MLP predictions vs actual")
plt.show()'''
