import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np


# ensure predictions output dir exists
PRED_DIR = Path("predictions")
PRED_DIR.mkdir(exist_ok=True)

from data_handling.data_w_acc import DataLoader
from model_evaluator import ModelEvaluator
from models.AdaBoost import AdaBoostModel
from models.Linear_Regression import LinearRegressionModel
from models.MLP import MLPFuelModel
from models.Random_Forest import RandomForestModel

INPUT_PATH = Path("fuel_data/863609060735564.csv")
df = pd.read_csv(str(INPUT_PATH))
df_features = DataLoader(df).create_pd_dataframe()

for col in ("start_time", "end_time"):
    if col in df_features.columns:
        df_features[col] = pd.to_datetime(df_features[col], errors="coerce")

df_features["duration_s"] = (df_features["end_time"] - df_features["start_time"]).dt.total_seconds()
X = df_features.drop(
    columns=[c for c in ["start_time", "end_time", "fuel_diff_ml"] if c in df_features.columns])
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

# --- compute predictions for test set and save per-point predictions ---
def _clip_array(arr):
    return np.maximum(np.asarray(arr).ravel(), 0.0)

def _patch_model_predict_clipped(model):
    """Replace model.predict with a clipped wrapper and return the original predict callable."""
    orig = getattr(model, "predict")
    def _clipped(X):
        return _clip_array(orig(X))
    model.predict = _clipped
    return orig

# compute clipped predictions for saving
linear_preds = _clip_array(regression_model.predict(X_test))
rf_preds = _clip_array(rf_model.predict(X_test))
ada_preds = _clip_array(ada_boost_model.predict(X_test))
mlp_preds = _clip_array(mlp_model.predict(X_test))


def save_point_predictions_for_file(file_name: str, df_features: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.Series, preds: dict):
    """Save per-test-point actual and predicted values for all models into CSV and return path."""
    idx = X_test.index
    times = pd.DataFrame(index=idx)
    for col in ("start_time", "end_time"):
        if col in df_features.columns:
            times[col] = pd.to_datetime(df_features.loc[idx, col].values)
    # include mean_speed for each test point if available in df_features
    if "mean_speed" in df_features.columns:
        times["mean_speed"] = df_features.loc[idx, "mean_speed"].values
    df_out = times.reset_index(drop=True)
    df_out["actual"] = y_test.reset_index(drop=True).values

    n = len(df_out)
    for name, arr in preds.items():
        arr = np.asarray(arr).ravel()
        if arr.shape[0] != n:
            raise ValueError(f"Prediction length for model '{name}' ({arr.shape[0]}) does not match number of test points ({n}).")
        col_pred = f"pred_{name}"
        col_res = f"res_{name}"
        col_abs = f"abs_{name}"
        df_out[col_pred] = arr
        df_out[col_res] = df_out["actual"] - df_out[col_pred]
        df_out[col_abs] = df_out[col_res].abs()

    out_path = PRED_DIR / f"predictions_{Path(file_name).name}"
    df_out.to_csv(out_path, index=False, encoding="utf-8")
    return out_path


# save predictions and get path (don't fail hard on save)
try:
    preds_dict = {"linear": linear_preds, "rf": rf_preds, "ada": ada_preds, "mlp": mlp_preds}
    pred_file = save_point_predictions_for_file(INPUT_PATH.name, df_features, X_test, y_test, preds_dict)
    pred_file_name = pred_file.name if pred_file is not None else ""
except Exception as e:
    print(f"Warning: saving per-point predictions failed: {e}")
    pred_file_name = ""

# Evaluate models using a temporarily patched predict (clipped >=0) so metrics reflect non-negative preds
orig = None
try:
    orig = _patch_model_predict_clipped(regression_model)
    reg_eval_result = ModelEvaluator(regression_model, df_features).evaluate(X_test, y_test, aggregate_window="10min")
finally:
    if orig is not None:
        regression_model.predict = orig

linear_aggregated_r2 = reg_eval_result.get("aggregated", {}).get("r2") if isinstance(reg_eval_result, dict) else None
linear_r2 = reg_eval_result.get("normal", {}).get("r2") if isinstance(reg_eval_result, dict) else None
linear_mae = reg_eval_result.get("normal", {}).get("mae") if isinstance(reg_eval_result, dict) else None
linear_rmse = reg_eval_result.get("normal", {}).get("rmse") if isinstance(reg_eval_result, dict) else None
linear_aggregated_mae = reg_eval_result.get("aggregated", {}).get("mae") if isinstance(reg_eval_result, dict) else None
linear_aggregated_rmse = reg_eval_result.get("aggregated", {}).get("rmse") if isinstance(reg_eval_result,
                                                                                         dict) else None

try:
    orig = _patch_model_predict_clipped(rf_model)
    rf_eval_result = ModelEvaluator(rf_model, df_features).evaluate(X_test, y_test, aggregate_window="10min")
finally:
    if orig is not None:
        rf_model.predict = orig

rf_aggregated_r2 = rf_eval_result.get("aggregated", {}).get("r2") if isinstance(rf_eval_result, dict) else None
rf_r2 = rf_eval_result.get("normal", {}).get("r2") if isinstance(rf_eval_result, dict) else None
rf_mae = rf_eval_result.get("normal", {}).get("mae") if isinstance(rf_eval_result, dict) else None
rf_rmse = rf_eval_result.get("normal", {}).get("rmse") if isinstance(rf_eval_result, dict) else None
rf_aggregated_mae = rf_eval_result.get("aggregated", {}).get("mae") if isinstance(rf_eval_result, dict) else None
rf_aggregated_rmse = rf_eval_result.get("aggregated", {}).get("rmse") if isinstance(rf_eval_result, dict) else None

try:
    orig = _patch_model_predict_clipped(ada_boost_model)
    ada_eval_result = ModelEvaluator(ada_boost_model, df_features).evaluate(X_test, y_test, aggregate_window="10min")
finally:
    if orig is not None:
        ada_boost_model.predict = orig

ada_aggregated_r2 = ada_eval_result.get("aggregated", {}).get("r2") if isinstance(ada_eval_result, dict) else None
ada_r2 = ada_eval_result.get("normal", {}).get("r2") if isinstance(ada_eval_result, dict) else None
ada_mae = ada_eval_result.get("normal", {}).get("mae") if isinstance(ada_eval_result, dict) else None
ada_rmse = ada_eval_result.get("normal", {}).get("rmse") if isinstance(ada_eval_result, dict) else None
ada_aggregated_mae = ada_eval_result.get("aggregated", {}).get("mae") if isinstance(ada_eval_result, dict) else None
ada_aggregated_rmse = ada_eval_result.get("aggregated", {}).get("rmse") if isinstance(ada_eval_result, dict) else None

try:
    orig = _patch_model_predict_clipped(mlp_model)
    mlp_eval_result = ModelEvaluator(mlp_model, df_features).evaluate(X_test, y_test, aggregate_window="10min")
finally:
    if orig is not None:
        mlp_model.predict = orig

mlp_aggregated_r2 = mlp_eval_result.get("aggregated", {}).get("r2") if isinstance(mlp_eval_result, dict) else None
mlp_r2 = mlp_eval_result.get("normal", {}).get("r2") if isinstance(mlp_eval_result, dict) else None
mlp_mae = mlp_eval_result.get("normal", {}).get("mae") if isinstance(mlp_eval_result, dict) else None
mlp_rmse = mlp_eval_result.get("normal", {}).get("rmse") if isinstance(mlp_eval_result, dict) else None
mlp_aggregated_mae = mlp_eval_result.get("aggregated", {}).get("mae") if isinstance(mlp_eval_result, dict) else None
mlp_aggregated_rmse = mlp_eval_result.get("aggregated", {}).get("rmse") if isinstance(mlp_eval_result, dict) else None

mean_speed_total = df_features["mean_speed"].mean() if "mean_speed" in df_features.columns else 0
std_speed_total = df_features["std_speed"].std() if "std_speed" in df_features.columns else 0

file_results = {
    "linear_r2": linear_r2,
    "rf_r2": rf_r2,
    "ada_r2": ada_r2,
    "mlp_r2": mlp_r2,
    "linear_mae": linear_mae,
    "rf_mae": rf_mae,
    "ada_mae": ada_mae,
    "mlp_mae": mlp_mae,
    "linear_rmse": linear_rmse,
    "rf_rmse": rf_rmse,
    "ada_rmse": ada_rmse,
    "mlp_rmse": mlp_rmse,
    "linear_aggregated_r2": linear_aggregated_r2,
    "rf_aggregated_r2": rf_aggregated_r2,
    "ada_aggregated_r2": ada_aggregated_r2,
    "mlp_aggregated_r2": mlp_aggregated_r2,
    "linear_aggregated_mae": linear_aggregated_mae,
    "rf_aggregated_mae": rf_aggregated_mae,
    "ada_aggregated_mae": ada_aggregated_mae,
    "mlp_aggregated_mae": mlp_aggregated_mae,
    "linear_aggregated_rmse": linear_aggregated_rmse,
    "rf_aggregated_rmse": rf_aggregated_rmse,
    "ada_aggregated_rmse": ada_aggregated_rmse,
    "mlp_aggregated_rmse": mlp_aggregated_rmse,
    "num_rows": len(df_features),
    "mean_speed": mean_speed_total,
    "std_speed": std_speed_total,
    "duration_mean_s": df_features["duration_s"].mean() if "duration_s" in df_features.columns else 0,
    "speed_points_mean": df_features["n_speed_points"].mean() if "n_speed_points" in df_features.columns else 0
}
file_results["predictions_file"] = pred_file_name

pd.DataFrame([file_results]).to_csv("test_results.csv", index=False)
