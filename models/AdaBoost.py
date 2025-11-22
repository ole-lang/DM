from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

class AdaBoostModel():

    def __init__(self, max_tree_depth, n_estimators, learning_rate, loss):
        base_estimator = DecisionTreeRegressor(max_depth=max_tree_depth)
        self.model = AdaBoostRegressor(
            estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            loss=loss,
            random_state=42
        )


    def train(self, X_train, y_train):
        '''Train the AdaBoost model'''
        self.model.fit(X_train, y_train)

    def predict(self, X):
        '''Make predictions using the trained AdaBoost model'''
        return self.model.predict(X)
