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
            keras.Input(shape=(input_dim,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
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
