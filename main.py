import traceback
from pathlib import Path

import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from Data import DataLoader
from Model_Evaluator import ModelEvaluator
from models.AdaBoost import AdaBoostModel
from models.Linear_Regression import LinearRegressionModel
from models.MLP import MLPFuelModel
from models.Random_Forest import RandomForestModel


DATA_DIR = Path("fuel_data")
CSV_GLOB = "*.csv"
OUT_CSV = Path("Results_no_acc.csv")

i = 1
results = []
for p in sorted(DATA_DIR.glob(CSV_GLOB)):
    print(i)
    i += 1
    try:
        fuel_data = pd.read_csv(p)
        df_features = DataLoader(fuel_data).create_pd_dataframe()
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as e:
        print(f"Reading Error in `{p}`: {e}")
        continue
    except Exception as e:
        print(f"Unexpected Reading Error in `{p}`: {e}")
        traceback.print_exc()
        continue

    try:
        for col in ("start_time", "end_time"):
            if col in df_features.columns:
                df_features[col] = pd.to_datetime(df_features[col], errors="coerce")

        # Drop rows mit ungültigen Zeitstempeln
        if {"start_time", "end_time"}.issubset(df_features.columns):
            df_features = df_features.dropna(subset=["start_time", "end_time"])
        if df_features.empty or "fuel_diff_ml" not in df_features.columns:
            print(f"Keine verwertbaren Daten in `{p}`, überspringe.")
            continue

        df_features["duration_s"] = (df_features["end_time"] - df_features["start_time"]).dt.total_seconds()
        X = df_features.drop(
            columns=[c for c in ["start_time", "end_time", "fuel_diff_ml"] if c in df_features.columns])
        y = df_features["fuel_diff_ml"]

        if X.empty or y.empty:
            print(f"Keine Features/Labels in `{p}`, überspringe.")
            continue

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


        # Hilfsfunktion für sichere R2-Berechnung
        def safe_r2(model, X_val, y_val):
            try:
                y_pred = model.predict(X_val)
                return float(r2_score(y_val, y_pred))
            except Exception as e:
                print(f"Fehler bei Vorhersage mit {model.__class__.__name__} für `{p}`: {e}")
                return None


        # Evaluationen durchführen und aggregierte Werte holen
        regression_evaluator = ModelEvaluator(regression_model, df_features)
        reg_eval_result = regression_evaluator.evaluate(X_test, y_test, aggregate_window="10min")
        linear_aggregated_r2 = reg_eval_result.get("aggregated", {}).get("r2") if isinstance(reg_eval_result,
                                                                                             dict) else None
        linear_r2 = reg_eval_result.get("r2") if isinstance(reg_eval_result, dict) and reg_eval_result.get(
            "r2") is not None else safe_r2(regression_model, X_test, y_test)

        rf_evaluator = ModelEvaluator(rf_model, df_features)
        rf_eval_result = rf_evaluator.evaluate(X_test, y_test, aggregate_window="10min")
        rf_aggregated_r2 = rf_eval_result.get("aggregated", {}).get("r2") if isinstance(rf_eval_result, dict) else None
        rf_r2 = rf_eval_result.get("r2") if isinstance(rf_eval_result, dict) and rf_eval_result.get(
            "r2") is not None else safe_r2(rf_model, X_test, y_test)

        ada_boost_evaluator = ModelEvaluator(ada_boost_model, df_features)
        ada_eval_result = ada_boost_evaluator.evaluate(X_test, y_test, aggregate_window="10min")
        ada_aggregated_r2 = ada_eval_result.get("aggregated", {}).get("r2") if isinstance(ada_eval_result,
                                                                                          dict) else None
        ada_r2 = ada_eval_result.get("r2") if isinstance(ada_eval_result, dict) and ada_eval_result.get(
            "r2") is not None else safe_r2(ada_boost_model, X_test, y_test)

        mlp_evaluator = ModelEvaluator(mlp_model, df_features)
        mlp_eval_result = mlp_evaluator.evaluate(X_test, y_test, aggregate_window="10min")
        mlp_aggregated_r2 = mlp_eval_result.get("aggregated", {}).get("r2") if isinstance(mlp_eval_result,
                                                                                          dict) else None
        mlp_r2 = mlp_eval_result.get("r2") if isinstance(mlp_eval_result, dict) and mlp_eval_result.get(
            "r2") is not None else safe_r2(mlp_model, X_test, y_test)

        mean_speed_total = df_features["mean_speed"].mean() if "mean_speed" in df_features.columns else 0
        std_speed_total = df_features["std_speed"].std() if "std_speed" in df_features.columns else 0

        file_results = {
            "file": p.name,
            "linear_r2": linear_r2,
            "rf_r2": rf_r2,
            "ada_r2": ada_r2,
            "mlp_r2": mlp_r2,
            "linear_aggregated_r2": linear_aggregated_r2,
            "rf_aggregated_r2": rf_aggregated_r2,
            "ada_aggregated_r2": ada_aggregated_r2,
            "mlp_aggregated_r2": mlp_aggregated_r2,
            "num_rows": len(df_features),
            "mean_speed": mean_speed_total,
            "std_speed": std_speed_total,
            "duration_mean_s": df_features["duration_s"].mean() if "duration_s" in df_features.columns else 0,
            "speed_points_mean": df_features["n_speed_points"].mean() if "n_speed_points" in df_features.columns else 0
        }

        results.append(file_results)

    except Exception as e:
        print(f"Fehler beim Verarbeiten der Datei `{p}`: {e}")
        traceback.print_exc()
        continue

# Schreibe aggregierte Ergebnisse weg
if results:
    pd.DataFrame(results).to_csv(OUT_CSV, index=False)
    print(f"Ergebnisse geschrieben nach `{OUT_CSV}`")

