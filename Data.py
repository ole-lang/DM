import pandas as pd
import numpy as np
from Acceleration import _acc_brake_totals_mdi

df = pd.read_csv("fuel_data/863609060549064.csv")

df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time")

#Extract GPS speeds
gps_speed_df = df.dropna(subset=["TRACKS.MUNIC.GPS_SPEED (km/h)"])[["time", "TRACKS.MUNIC.GPS_SPEED (km/h)"]].copy()
gps_speed_df = gps_speed_df.rename(columns={"TRACKS.MUNIC.GPS_SPEED (km/h)": "speed"})
gps_speed_df["source"] = "gps"

#Extract MDI speeds
mdi_speed_df = df.dropna(subset=["TRACKS.MUNIC.MDI_OBD_SPEED (km/h)"])[["time", "TRACKS.MUNIC.MDI_OBD_SPEED (km/h)"]].copy()
mdi_speed_df = mdi_speed_df.rename(columns={"TRACKS.MUNIC.MDI_OBD_SPEED (km/h)": "speed"})
mdi_speed_df["source"] = "mdi"

#Combine GPS and MDI speeds
speed_all = pd.concat([gps_speed_df, mdi_speed_df], ignore_index=True).sort_values("time").reset_index(drop=True)

#Calculate fuel differences
fuel_df = df.dropna(subset=["TRACKS.MUNIC.MDI_OBD_FUEL (ml)"]).sort_values("time").copy()
fuel_df["fuel_diff_ml"] = fuel_df["TRACKS.MUNIC.MDI_OBD_FUEL (ml)"].diff()


fuel_df = fuel_df.sort_values("time").reset_index(drop=True)
speed_all = speed_all.sort_values("time").reset_index(drop=True)

results = []
for i in range(1, len(fuel_df)):
    start = fuel_df.loc[i - 1, "time"]
    end = fuel_df.loc[i, "time"]
    fuel_diff = fuel_df.loc[i, "fuel_diff_ml"]

    mask = (mdi_speed_df["time"] > start) & (mdi_speed_df["time"] <= end)
    speeds = mdi_speed_df.loc[mask, "speed"].dropna()

    n_points = int(speeds.size)
    mean_speed = speeds.mean() if n_points > 0 else np.nan
    std_speed = speeds.std(ddof=1) if n_points > 1 else (0.0 if n_points == 1 else np.nan)

    if n_points >= 2:
        speed_times = mdi_speed_df.loc[mask, "time"].reset_index(drop=True)
        speed_values = speeds.reset_index(drop=True)
        acc_metrics = _acc_brake_totals_mdi(speed_times, speed_values)
    else:
        acc_metrics = {"total_acc_m_s": 0.0, "total_brake_m_s": 0.0}

    results.append({
        "start_time": start,
        "end_time": end,
        "fuel_diff_ml": fuel_diff,
        "n_speed_points": n_points,
        "mean_speed": mean_speed,
        "std_speed": std_speed,
        "total_acc_m_s": acc_metrics["total_acc_m_s"],
        "total_brake_m_s": acc_metrics["total_brake_m_s"]
    })

fuel_intervals = pd.DataFrame(results)
print(fuel_intervals["total_acc_m_s"].to_string(index=False))


#Amount of NaN values analysis
rows_with_nan = fuel_intervals.isna().any(axis=1).sum()
print("Rows with ≥1 NaN:", rows_with_nan)

print("NaN per column:\n", fuel_intervals.isna().sum())

total_nans = fuel_intervals.isna().sum().sum()
print("Total NaNs:", total_nans)

rows_all_nan = fuel_intervals.isna().all(axis=1).sum()
print("Column with only NaNs:", rows_all_nan)

#Drop rows with NaN in mean_speed
fuel_intervals = fuel_intervals.dropna(subset=["mean_speed"]).reset_index(drop=True)

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

#Save in CSV
#fuel_intervals.to_csv('fuel_intervals.csv', index=False, encoding='utf-8')
#print("Geschrieben:", os.path.abspath('fuel_intervals.csv'))