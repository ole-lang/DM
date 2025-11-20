from sklearn.ensemble import RandomForestRegressor

class RandomForestModel:
    def __init__(self, n_estimators, max_depth, random_state, 
    min_samples_split, min_samples_leaf, max_features):
        self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, 
        min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)
