# python
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import os

df = pd.read_csv("fuel_data/863609060559592.csv")
df["time"] = pd.to_datetime(df["time"])
gps_col = next((c for c in df.columns if "gps_speed" in c.lower()), None)
mdi_col = next((c for c in df.columns if "mdi_obd_speed" in c.lower()), None)
frames = []
if gps_col:
    g = df.dropna(subset=[gps_col])[["time", gps_col]].rename(columns={gps_col: "speed"}).copy()
    g["source"] = "gps"
    frames.append(g)
if mdi_col:
    m = df.dropna(subset=[mdi_col])[["time", mdi_col]].rename(columns={mdi_col: "speed"}).copy()
    m["source"] = "mdi"
    frames.append(m)
if not frames:
    raise RuntimeError("Keine GPS/MDI Speed-Spalten gefunden in CSV.")
speed_all = pd.concat(frames, ignore_index=True).sort_values("time").reset_index(drop=True)

# Sicherstellen, dass time datetime ist
speed_all = speed_all.copy()
speed_all["time"] = pd.to_datetime(speed_all["time"])

plt.style.use("default")
fig, ax = plt.subplots(figsize=(12, 5))

for src, group in speed_all.groupby("source"):
    ax.plot(group["time"], group["speed"], label=src.upper(), lw=1.2, alpha=0.9)

ax.set_ylabel("Speed (km/h)")
ax.set_xlabel("Zeit")
ax.set_title("MDI und GPS Speed über Zeit")
ax.legend()
ax.grid(True, alpha=0.3)

# Datumformatierung
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
fig.autofmt_xdate()

plt.tight_layout()
plt.show()

