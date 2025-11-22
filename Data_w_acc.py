import numpy as np
import pandas as pd
import os
from Acceleration import _acc_brake_totals


class DataLoader():
    def __init__(self, data_folder):
        self.data_folder = data_folder

    def create_pd_dataframe(self):

        df = self.data_folder
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time")

        # Extract MDI speeds
        mdi_speed_df = df.dropna(subset=["TRACKS.MUNIC.MDI_OBD_SPEED (km/h)"])[
            ["time", "TRACKS.MUNIC.MDI_OBD_SPEED (km/h)"]].copy()
        mdi_speed_df = mdi_speed_df.rename(columns={"TRACKS.MUNIC.MDI_OBD_SPEED (km/h)": "speed"})
        mdi_speed_df["source"] = "mdi"

        # Calculate fuel differences
        fuel_df = df.dropna(subset=["TRACKS.MUNIC.MDI_OBD_FUEL (ml)"]).sort_values("time").copy()
        fuel_df["fuel_diff_ml"] = fuel_df["TRACKS.MUNIC.MDI_OBD_FUEL (ml)"].diff()

        fuel_df = fuel_df.sort_values("time").reset_index(drop=True)
        speed_all = mdi_speed_df.sort_values("time").reset_index(drop=True)

        results = []
        for i in range(1, len(fuel_df)):
            start = fuel_df.loc[i - 1, "time"]
            end = fuel_df.loc[i, "time"]
            fuel_diff = fuel_df.loc[i, "fuel_diff_ml"]
            duration = (end - start).total_seconds()
            if duration <= 0 or duration > 600:
                continue  # Skip invalid or too long intervals:

            mask = (speed_all["time"] > start) & (speed_all["time"] <= end)
            speeds = speed_all.loc[mask, "speed"].dropna()
            n_points = int(speeds.size)
            mean_speed = speeds.mean() if n_points > 0 else np.nan
            std_speed = speeds.std(ddof=1) if n_points > 1 else (0.0 if n_points == 1 else np.nan)

            if n_points >= 2:
                speed_times = speed_all.loc[mask, "time"].reset_index(drop=True)
                speed_values = speeds.reset_index(drop=True)
                acc_metrics = _acc_brake_totals(speed_times, speed_values)
            else:
                # Keine ausreichenden Speed-Punkte -> sichere Defaults für alle erwarteten Keys
                acc_metrics = {}

            results.append({
                "start_time": start,
                "end_time": end,
                "fuel_diff_ml": fuel_diff,
                "n_speed_points": n_points,
                "mean_speed": mean_speed,
                "std_speed": std_speed,
                "total_acc_m_s": acc_metrics.get("total_acc_m_s", 0.0) / ((end - start).total_seconds()) if (end - start).total_seconds() > 0 else np.nan,
                "total_brake_m_s": acc_metrics.get("total_brake_m_s", 0.0) / ((end - start).total_seconds()) if (end - start).total_seconds() > 0 else np.nan,
                "total_acc_duration_s": acc_metrics.get("total_acc_duration_s", 0.0),
                "total_brake_duration_s": acc_metrics.get("total_brake_duration_s", 0.0),
                "peak_acc_m_s2": acc_metrics.get("peak_acc_m_s2", 0.0),
                "peak_brake_m_s2": acc_metrics.get("peak_brake_m_s2", 0.0),
                "mean_acc_m_s2": acc_metrics.get("mean_acc_m_s2", 0.0),
                "mean_brake_m_s2": acc_metrics.get("mean_brake_m_s2", 0.0),
                "acc_event_count": acc_metrics.get("acc_event_count", 0),
                "brake_event_count": acc_metrics.get("brake_event_count", 0),
            })

        fuel_intervals = pd.DataFrame(results)
        # print(fuel_intervals["total_acc_m_s"].to_string(index=False))

        # Amount of NaN values analysis
        # rows_with_nan = fuel_intervals.isna().any(axis=1).sum()
        # print("Rows with ≥1 NaN:", rows_with_nan)
        # print("NaN per column:\n", fuel_intervals.isna().sum())
        # total_nans = fuel_intervals.isna().sum().sum()
        # print("Total NaNs:", total_nans)

        # rows_all_nan = fuel_intervals.isna().all(axis=1).sum()
        # print("Column with only NaNs:", rows_all_nan)

        # Drop rows with NaN in allen relevanten Metriken
        metric_cols = [
            "fuel_diff_ml", "n_speed_points", "mean_speed", "std_speed",
            "total_acc_m_s", "total_brake_m_s",
            "total_acc_duration_s", "total_brake_duration_s",
            "peak_acc_m_s2", "peak_brake_m_s2",
            "mean_acc_m_s2", "mean_brake_m_s2",
            "acc_event_count", "brake_event_count",
        ]
        # rows_with_nan = fuel_intervals[metric_cols].isna().any(axis=1).sum()
        # print(f"Rows with NaN in metrics: {rows_with_nan}")
        fuel_intervals = fuel_intervals.dropna(subset=metric_cols).reset_index(drop=True)

        # 95 quantile analysis for fuel_diff_ml
        s = fuel_intervals["fuel_diff_ml"]
        valid = s.dropna()

        if len(valid) < 10:
            print("To few values in fuel_diff:", len(valid))
        else:
            q_low = valid.quantile(0.025)
            q_high = valid.quantile(0.975)

            is_outlier = s.notna() & ((s < q_low) | (s > q_high))
            removed = int(is_outlier.sum())
            before = len(fuel_intervals)

            fuel_intervals = fuel_intervals.loc[~is_outlier].reset_index(drop=True)

            # after = len(fuel_intervals)
            # print(f"Quantile: low={q_low}, high={q_high}")
            # print(f"Outliers removed: {removed} / {before} -> Rows left: {after}")

            # Optional: als CSV speichern
            # fuel_intervals.to_csv('fuel_intervals_filtered.csv', index=False, encoding='utf-8')

            # Save in CSV
            fuel_intervals.to_csv('fuel_intervals.csv', index=False, encoding='utf-8')
            print("Geschrieben:", os.path.abspath('fuel_intervals.csv'))

        return fuel_intervals
