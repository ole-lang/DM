import os

import pandas as pd
import numpy as np
from Acceleration import _acc_brake_totals
from pathlib import Path

DATA_DIR = Path("fuel_data")
CSV_GLOB = "*.csv"
OUT_CSV = Path("regression_r2_per_file.csv")


for p in sorted(DATA_DIR.glob(CSV_GLOB)):
    try:
        csv = pd.read_csv(p)
        df = pd.read_csv("fuel_data/" + csv)
        # df2 = pd.read_csv("fuel_data/863609060549064.csv")
        # df = pd.concat([df, df2], axis=0, ignore_index=True, sort=False)

        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time")

        # Extract GPS speeds
        gps_speed_df = df.dropna(subset=["TRACKS.MUNIC.GPS_SPEED (km/h)"])[["time", "TRACKS.MUNIC.GPS_SPEED (km/h)"]].copy()
        gps_speed_df = gps_speed_df.rename(columns={"TRACKS.MUNIC.GPS_SPEED (km/h)": "speed"})
        gps_speed_df["source"] = "gps"

        # Extract MDI speeds
        mdi_speed_df = df.dropna(subset=["TRACKS.MUNIC.MDI_OBD_SPEED (km/h)"])[
            ["time", "TRACKS.MUNIC.MDI_OBD_SPEED (km/h)"]].copy()
        mdi_speed_df = mdi_speed_df.rename(columns={"TRACKS.MUNIC.MDI_OBD_SPEED (km/h)": "speed"})
        mdi_speed_df["source"] = "mdi"

        # Combine GPS and MDI speeds
        speed_all = pd.concat([gps_speed_df, mdi_speed_df], ignore_index=True).sort_values("time").reset_index(drop=True)

        # Calculate fuel differences
        fuel_df = df.dropna(subset=["TRACKS.MUNIC.MDI_OBD_FUEL (ml)"]).sort_values("time").copy()
        fuel_df["fuel_diff_ml"] = fuel_df["TRACKS.MUNIC.MDI_OBD_FUEL (ml)"].diff()

        fuel_df = fuel_df.sort_values("time").reset_index(drop=True)
        speed_all = speed_all.sort_values("time").reset_index(drop=True)

        results = []
        for i in range(1, len(fuel_df)):
            start = fuel_df.loc[i - 1, "time"]
            end = fuel_df.loc[i, "time"]
            fuel_diff_raw = fuel_df.loc[i, "fuel_diff_ml"]
            fuel_diff = fuel_diff_raw if (pd.notna(fuel_diff_raw) and fuel_diff_raw >= 0) else np.nan

            duration_s = (end - start).total_seconds()
            # only use time intervals between 0 and 600 seconds

            if duration_s <= 0 or duration_s >= 600:
                continue

            mask = (mdi_speed_df["time"] > start) & (mdi_speed_df["time"] <= end)
            speeds = mdi_speed_df.loc[mask, "speed"].dropna()

            n_points = int(speeds.size)
            mean_speed = speeds.mean() if n_points > 0 else np.nan
            std_speed = speeds.std(ddof=1) if n_points > 1 else (0.0 if n_points == 1 else np.nan)

            if n_points >= 2:
                speed_times = mdi_speed_df.loc[mask, "time"].reset_index(drop=True)
                speed_values = speeds.reset_index(drop=True)
                acc_metrics = _acc_brake_totals(speed_times, speed_values)
            else:
                acc_metrics = {"total_acc_m_s": 0.0, "total_brake_m_s": 0.0}

            results.append({
                "start_time": start,
                "end_time": end,
                "fuel_diff_ml": fuel_diff,
                "n_speed_points": n_points,
                "mean_speed": mean_speed,
                "std_speed": std_speed,
                "total_acc_m_s": acc_metrics["total_acc_m_s"] / (end - start).total_seconds() if (
                                                                                                             end - start).total_seconds() > 0 else np.nan,
                "total_brake_m_s": acc_metrics["total_brake_m_s"]
            })

        fuel_intervals = pd.DataFrame(results)

        # Amount of NaN values analysis
        rows_with_nan = fuel_intervals.isna().any(axis=1).sum()
        # print("Rows with ≥1 NaN:", rows_with_nan)

        # print("NaN per column:\n", fuel_intervals.isna().sum())

        total_nans = fuel_intervals.isna().sum().sum()
        # print("Total NaNs:", total_nans)

        rows_all_nan = fuel_intervals.isna().all(axis=1).sum()
        # print("Column with only NaNs:", rows_all_nan)

        # Drop rows with NaN in mean_speed
        fuel_intervals = fuel_intervals.dropna(subset=["mean_speed"]).reset_index(drop=True)

        """
        #95 quantile analysis for fuel_diff_ml
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
    
            after = len(fuel_intervals)
            print(f"Quantile: low={q_low}, high={q_high}")
            print(f"Outliers removed: {removed} / {before} -> Rows left: {after}")
    
            # Optional: als CSV speichern
            # fuel_intervals.to_csv('fuel_intervals_filtered.csv', index=False, encoding='utf-8')
        """

        # IQR analysis for fuel_diff_ml
        s = fuel_intervals["fuel_diff_ml"]
        valid = s.dropna()
        if len(valid) < 10:
            print("To few values in fuel_diff:", len(valid))
        else:
            q1 = valid.quantile(0.25)
            q3 = valid.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            is_outlier = s.notna() & ((s < lower_bound) | (s > upper_bound))
            removed = int(is_outlier.sum())
            before = len(fuel_intervals)

            fuel_intervals = fuel_intervals.loc[~is_outlier].reset_index(drop=True)

            after = len(fuel_intervals)
            # print(f"IQR: Q1={q1}, Q3={q3}, IQR={iqr}, lower_bound={lower_bound}, upper_bound={upper_bound}")
            # print(f"Outliers removed: {removed} / {before} -> Rows left: {after}")


        # Save in CSV
        output_path = DATA_DIR / f"{p.stem}_intervals.csv"
        fuel_intervals.to_csv(output_path, index=False, encoding='utf-8')
        print("Written:", output_path.resolve())
    except Exception as e:
        print(f"Error processing file `{p}`: {e}")
        continue