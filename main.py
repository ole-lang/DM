from sklearn.model_selection import train_test_split
import pandas as pd

from models.AdaBoost import AdaBoostModel
from models.Random_Forest import RandomForestModel
from models.MLP import MLPFuelModel
from Model_Evaluator import ModelEvaluator
from models.Linear_Regression import LinearRegressionModel

from Data import DataLoader

# TODO create a list, of all the files and run them on our models and evaluate them
# put it then into a table of some sort and try to find ut which files perform better or worse
# maybe try to find some patterns in the data that lead to better or worse performance

fuel_data = pd.read_csv("fuel_data/863609060584707.csv")
df_features = DataLoader(fuel_data).create_pd_dataframe()

# Features and target variable
df_features["duration_s"] = (df_features["end_time"] - df_features["start_time"]).dt.total_seconds()
X = df_features.drop(columns=["start_time", "end_time", "fuel_diff_ml"])
y = df_features["fuel_diff_ml"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
regression_model = LinearRegressionModel()
regression_model.train(X_train, y_train)

rf_model = RandomForestModel(
    n_estimators=300, max_depth=4, random_state=42, 
    min_samples_split=5, min_samples_leaf=2, max_features='sqrt'
)
rf_model.train(X_train, y_train)

ada_boost_model = AdaBoostModel(
    max_tree_depth=4, n_estimators=500, learning_rate=0.01, loss="exponential"
)
ada_boost_model.train(X_train, y_train)

mlp_model = MLPFuelModel()
mlp_model.build_model(input_dim=X_train.shape[1])
mlp_model.train(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2, verbose=0)


# Evaluate models
print("\n--- Linear Regression Model Evaluation ---")
regression_evaluator = ModelEvaluator(regression_model, df_features)
regression_evaluator.evaluate(X_test, y_test, aggregate_window="30min")
print("\n--- Random Forest Model Evaluation ---")
rf_evaluator = ModelEvaluator(rf_model, df_features)
rf_evaluator.evaluate(X_test, y_test, aggregate_window="30min")  
print("\n--- AdaBoost Model Evaluation ---")
ada_boost_evaluator = ModelEvaluator(ada_boost_model, df_features)
ada_boost_evaluator.evaluate(X_test, y_test, aggregate_window="30min")
print("\n--- MLP Model Evaluation ---")
mlp_evaluator = ModelEvaluator(mlp_model, df_features)
mlp_evaluator.evaluate(X_test, y_test, aggregate_window="30min")
