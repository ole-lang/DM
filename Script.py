# python
from pathlib import Path
import pandas as pd
import numpy as np
from Data import data as load_data
from Linear_Regression_w_acc import regression as run_regression

DATA_DIR = Path("fuel_data")
CSV_GLOB = "*.csv"
OUT_CSV = Path("regression_r2_per_file.csv")

def main():
    results = []
    for p in sorted(DATA_DIR.glob(CSV_GLOB)):
        print(p)
        fname = p.name
        try:
            fuel_intervals = load_data(fname)  # Data.data erwartet Dateiname innerhalb fuel_data/
        except Exception as e:
            results.append({"file": fname, "r2": np.nan, "error": f"load_data_error: {e}"})
            continue

        try:
            r2 = run_regression(fuel_intervals)
            if r2 is None:
                r2_val = np.nan
            else:
                r2_val = float(r2)
            results.append({"file": fname, "r2": r2_val, "error": ""})
        except Exception as e:
            results.append({"file": fname, "r2": np.nan, "error": f"regression_error: {e}"})

    df_res = pd.DataFrame(results)
    df_res.to_csv(OUT_CSV, index=False)
    print(f"Wrote results to `{OUT_CSV}`")
    print(df_res)

if __name__ == "__main__":
    main()