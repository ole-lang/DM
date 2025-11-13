import pandas as pd
import numpy as np

def _acc_brake_totals_mdi(speed_times, speeds_kmh):

    if len(speeds_kmh) < 2:
        return {"total_acc_m_s": 0.0, "total_brake_m_s": 0.0}

    times = pd.to_datetime(speed_times).to_numpy().astype("datetime64[ns]")
    speeds_ms = np.asarray(speeds_kmh, dtype=float) / 3.6  # km/h -> m/s

    dt = np.diff(times).astype("timedelta64[ns]").astype(np.float64) / 1e9
    dv = np.diff(speeds_ms)

    valid = dt > 0
    if not valid.any():
        return {"total_acc_m_s": 0.0, "total_brake_m_s": 0.0}

    dv = dv[valid]

    total_pos_dv = float(np.sum(dv[dv > 0]) if np.any(dv > 0) else 0.0)
    total_neg_dv = float(-np.sum(dv[dv < 0]) if np.any(dv < 0) else 0.0)

    return {"total_acc_m_s": total_pos_dv, "total_brake_m_s": total_neg_dv}