import pandas as pd
import numpy as np

df = pd.read_csv("863609060548926.csv")

df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time")
gps_df = df.dropna(subset=["TRACKS.MUNIC.GPS_SPEED (km/h)"]).copy()
fuel_df = df.dropna(subset=["TRACKS.MUNIC.MDI_OBD_FUEL (ml)"]).copy()
fuel_df["ΔFuel"] = fuel_df["TRACKS.MUNIC.MDI_OBD_FUEL (ml)"].diff()
fuel_df["Δt"] = fuel_df["time"].diff().dt.total_seconds()
fuel_df["fuel_rate"] = fuel_df["ΔFuel"] / fuel_df["Δt"]
fuel_df = fuel_df[(fuel_df["fuel_rate"] >= 0)]
q99 = fuel_df["fuel_rate"].quantile(0.99)
fuel_df = fuel_df[fuel_df["fuel_rate"] <= q99].copy()
fuel_df["fuel_rate_lph"] = fuel_df["fuel_rate"] * 3.6  # ml/s → l/h
if "gps_speed" not in fuel_df.columns:
    fuel_df = fuel_df.copy()
    fuel_df["gps_speed"] = np.interp(
        x=fuel_df["time"].astype(np.int64),
        xp=gps_df["time"].astype(np.int64),
        fp=gps_df["TRACKS.MUNIC.GPS_SPEED (km/h)"]
    )


df_model = fuel_df[["gps_speed", "fuel_rate"]].copy()
df_model["gps_speed"] = pd.to_numeric(df_model["gps_speed"], errors="coerce")
df_model["fuel_rate"] = pd.to_numeric(df_model["fuel_rate"], errors="coerce")

fuel_df["speed_ms"] = fuel_df["gps_speed"] / 3.6

# Roh-Beschleunigung: delta speed / delta t -> m/s²
# Achtung: Δt kann 0 oder NaN sein -> setze dann NaN
fuel_df["acc_ms2"] = fuel_df["speed_ms"].diff() / fuel_df["Δt"]
fuel_df.loc[fuel_df["Δt"] <= 0, "acc_ms2"] = np.nan

# Zeitbasiertes Glätten (empfohlen) — z.B. 5s Fenster, benötigt datetime-index
if "time" in fuel_df.columns:
    tmp = fuel_df.set_index("time")
    tmp["acc_ms2_smooth"] = tmp["acc_ms2"].rolling("5s", min_periods=1).mean()
    fuel_df["acc_ms2_smooth"] = tmp["acc_ms2_smooth"].values
else:
    # fallback: gleite über 3 Messpunkte
    fuel_df["acc_ms2_smooth"] = fuel_df["acc_ms2"].rolling(window=3, min_periods=1, center=True).mean()

df_model = fuel_df[[
    "gps_speed", "fuel_rate", "acc_ms2_smooth"

]].copy()

df_model = df_model.dropna().reset_index(drop=True)

if df_model.empty:
    raise ValueError("No valid data available for modeling after preprocessing.")

print(df_model.head())