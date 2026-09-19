

import pandas as pd
import numpy as np
import os
from paths import PROJECT_ROOT, RESULTS_DIR

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def evaluate_predictions(csv_path, corruption_type="clean", severity="none"):
    df = pd.read_csv(csv_path)
    df["error_km"] = haversine(df["true_lat"], df["true_lon"], df["pred_lat"], df["pred_lon"])

    metrics = {
        "corruption_type": corruption_type,
        "severity": severity,
        "n_samples": len(df),
        "mean_error_km": df["error_km"].mean(),
        "median_error_km": df["error_km"].median(),
        "acc_street_1km": (df["error_km"] < 1).mean() * 100,
        "acc_city_25km": (df["error_km"] < 25).mean() * 100,
        "acc_region_200km": (df["error_km"] < 200).mean() * 100,
        "acc_country_750km": (df["error_km"] < 750).mean() * 100,
        "acc_continent_2500km": (df["error_km"] < 2500).mean() * 100,
    }
    return metrics, df


def append_to_summary(metrics, summary_path):
    row_df = pd.DataFrame([metrics])
    if os.path.exists(summary_path):
        existing = pd.read_csv(summary_path)
        existing = existing[
            ~((existing["corruption_type"] == metrics["corruption_type"]) &
              (existing["severity"] == metrics["severity"]))
        ]
        combined = pd.concat([existing, row_df], ignore_index=True)
    else:
        combined = row_df
    combined.to_csv(summary_path, index=False)
    return combined


if __name__ == "__main__":
    BASE_DIR = PROJECT_ROOT
    CLEAN_CSV = RESULTS_DIR / "clean" / "baseline_predictions.csv"
    SUMMARY_PATH = RESULTS_DIR / "summary_metrics.csv"

    metrics, df_with_errors = evaluate_predictions(CLEAN_CSV, corruption_type="clean", severity="none")

    print("نتایج ارزیابی (Clean Baseline)")
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")

    df_with_errors.to_csv(CLEAN_CSV.replace(".csv", "_with_errors.csv"), index=False)
    summary = append_to_summary(metrics, SUMMARY_PATH)
    print("\nجدول خلاصه ذخیره شد در:", SUMMARY_PATH)
    print(summary)
