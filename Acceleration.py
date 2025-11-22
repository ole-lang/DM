import pandas as pd
import numpy as np


def _acc_brake_totals(speed_times, speeds_kmh):
    """Only totals plus duration and intensity of accelerations:
    - total_acc_m_s / total_brake_m_s: sum of positive/negative Δv in m/s
    - total_acc_duration_s / total_brake_duration_s: cumulative time in s with a>0 / a<0
    - peak_acc_m_s2 / peak_brake_m_s2: peak acceleration values in m/s**2 (braking reported as positive)
    - mean_acc_m_s2 / mean_brake_m_s2: average intensity (only over positive/negative intervals)
    - acc_event_count / brake_event_count: number of contiguous acceleration/braking events"""

    if len(speeds_kmh) < 2:
        return {
            "total_acc_m_s": 0.0, "total_brake_m_s": 0.0,
            "total_acc_duration_s": 0.0, "total_brake_duration_s": 0.0,
            "peak_acc_m_s2": 0.0, "peak_brake_m_s2": 0.0,
            "mean_acc_m_s2": 0.0, "mean_brake_m_s2": 0.0,
            "acc_event_count": 0, "brake_event_count": 0,
        }

    times = pd.to_datetime(speed_times).to_numpy().astype("datetime64[ns]")
    speeds_ms = np.asarray(speeds_kmh, dtype=float) / 3.6  # km/h -> m/s

    dt_all = np.diff(times).astype("timedelta64[ns]").astype(np.float64) / 1e9
    dv_all = np.diff(speeds_ms)

    valid = dt_all > 0
    if not valid.any():
        return {
            "total_acc_m_s": 0.0, "total_brake_m_s": 0.0,
            "total_acc_duration_s": 0.0, "total_brake_duration_s": 0.0,
            "peak_acc_m_s2": 0.0, "peak_brake_m_s2": 0.0,
            "mean_acc_m_s2": 0.0, "mean_brake_m_s2": 0.0,
            "acc_event_count": 0, "brake_event_count": 0,
        }

    dt = dt_all[valid]
    dv = dv_all[valid]

    a = dv / dt  # m/s**2

    # ensure we only use finite values for masks (NaN/Inf vermeiden)
    finite_mask = np.isfinite(a) & np.isfinite(dv) & np.isfinite(dt)
    pos_mask = (a > 0) & finite_mask
    neg_mask = (a < 0) & finite_mask

    total_pos_dv = float(np.sum(dv[pos_mask]) if np.any(pos_mask) else 0.0)
    total_neg_dv = float(-np.sum(dv[neg_mask]) if np.any(neg_mask) else 0.0)

    total_acc_duration = float(np.sum(dt[pos_mask]) if np.any(pos_mask) else 0.0)
    total_brake_duration = float(np.sum(dt[neg_mask]) if np.any(neg_mask) else 0.0)

    peak_pos_acc = float(a[pos_mask].max()) if np.any(pos_mask) else 0.0
    peak_neg_acc = float(-a[neg_mask].min()) if np.any(neg_mask) else 0.0

    mean_pos_acc = float(a[pos_mask].mean()) if np.any(pos_mask) else 0.0
    mean_neg_acc = float(-a[neg_mask].mean()) if np.any(neg_mask) else 0.0

    # Count acceleration/braking events based on threshold crossings
    # thresh = 0.1  # m/s^2
    pos_events = (a > 0)
    neg_events = (a < 0)

    def count_events(mask):
        if not mask.any():
            return 0
        edges = np.diff(mask.astype(int))
        return int(np.sum(edges == 1) + (1 if mask[0] else 0))

    acc_event_count = count_events(pos_events)
    brake_event_count = count_events(neg_events)

    return {
        "total_acc_m_s": total_pos_dv,
        "total_brake_m_s": total_neg_dv,
        "total_acc_duration_s": total_acc_duration,
        "total_brake_duration_s": total_brake_duration,
        "peak_acc_m_s2": peak_pos_acc,
        "peak_brake_m_s2": peak_neg_acc,
        "mean_acc_m_s2": mean_pos_acc,
        "mean_brake_m_s2": mean_neg_acc,
        "acc_event_count": acc_event_count,
        "brake_event_count": brake_event_count,
    }
