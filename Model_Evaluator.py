# evaluation/model_evaluator.py
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

class ModelEvaluator:
    def __init__(self, model, df_features):
        """
        model: trained model object (inherits BaseModel)
        df_features: dataframe including start_time, end_time, fuel_diff_ml, and features used in training
        """
        self.model = model
        self.df_features = df_features.copy()
        
        # X_test needs to be scaled in MLP case
    def evaluate(self, X_test, y_test, aggregate_window="30min"):
        """Evaluate model both normally and on aggregated 30-min windows."""
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test, columns=self.df_features.drop(columns=["start_time", "end_time", "fuel_diff_ml"]).columns)
        
        y_pred = self.model.predict(X_test)

        # normal metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f"\n Normal Evaluation:")
        print(f"R² = {r2:.3f}")
        print(f"MAE = {mae:.3f} ml")
        print(f"RMSE = {rmse:.3f} ml")

        # aggregated metrics
        results = X_test.copy()
        results["actual"] = y_test.values
        results["predicted"] = y_pred
        results["start_time"] = self.df_features.loc[X_test.index, "start_time"]

        results["start_time"] = pd.to_datetime(results["start_time"])
        results = results.sort_values("start_time").set_index("start_time")
        '''
        agg = results.resample(aggregate_window).sum(numeric_only=True)[["actual", "predicted"]]

        r2_agg = r2_score(agg["actual"], agg["predicted"])
        mae_agg = mean_absolute_error(agg["actual"], agg["predicted"])
        rmse_agg = np.sqrt(mean_squared_error(agg["actual"], agg["predicted"]))

        print(f"\n Aggregated ({aggregate_window}) Evaluation:")
        print(f"R² = {r2_agg:.3f}")
        print(f"MAE = {mae_agg:.3f} ml")
        print(f"RMSE = {rmse_agg:.3f} ml")

        '''
        # Plots:
        plt.figure(figsize=(6,6))
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Predicted vs Actual (interval level)")
        plt.show()
        '''
        plt.figure(figsize=(10,5))
        plt.plot(agg.index, agg["actual"], label="Actual", marker='o')
        plt.plot(agg.index, agg["predicted"], label="Predicted", marker='x')
        plt.title(f"Aggregated Fuel Consumption ({aggregate_window})")
        plt.legend()
        plt.show()
        '''
        return {
            "normal": {"r2": r2, "mae": mae, "rmse": rmse},
            # "aggregated": {"r2": r2_agg, "mae": mae_agg, "rmse": rmse_agg}
        }
        