import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LogisticRegression
from paths import RESULTS_DIR

CALIB_PATH = RESULTS_DIR / "confidence" / "calib_test_with_confidence.csv"
ALL_CONF_PATH = RESULTS_DIR / "confidence" / "all_with_confidence.csv"
MODEL_OUT_PATH = RESULTS_DIR / "abstention_model.pkl"
COMPARISON_OUT_PATH = RESULTS_DIR / "abstention_comparison.csv"

COUNTRY_THRESHOLD_KM = 750
TARGET_COVERAGES = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def load_and_fix(path):
    df = pd.read_csv(path)
    df["corruption_type"] = df["corruption_type"].astype(object) if "corruption_type" in df.columns else "clean"
    if "severity" in df.columns:
        df["severity"] = df["severity"].astype(object)
    mask = df["corruption_type"].isna() if "corruption_type" in df.columns else pd.Series([True]*len(df))
    df.loc[mask, "corruption_type"] = "clean"
    if "severity" in df.columns:
        df.loc[mask, "severity"] = "none"
    if "error_km" not in df.columns or df["error_km"].isna().any():
        df["error_km"] = haversine(df["true_lat"], df["true_lon"], df["pred_lat"], df["pred_lon"])
    return df


# گام ۱: بارگذاری calibration و آموزش مدل
print("در حال بارگذاری calibration set...")
calib = load_and_fix(CALIB_PATH)
calib["correct"] = (calib["error_km"] < COUNTRY_THRESHOLD_KM).astype(int)

# رویکرد ۱: threshold ساده روی entropy
entropy_thresholds = {cov: calib["entropy"].quantile(cov) for cov in TARGET_COVERAGES}

# رویکرد ۲: Logistic Regression
X_calib = calib[["top1_score", "margin", "entropy"]].values
y_calib = calib["correct"].values
clf = LogisticRegression()
clf.fit(X_calib, y_calib)

print("مدل abstention آموزش داده شد.")
print("ضرایب LR:", dict(zip(["top1_score", "margin", "entropy"], clf.coef_[0])))

# گام ۲: ذخیره‌ی رسمی مدل
abstention_artifact = {
    "logistic_regression": clf,
    "entropy_thresholds_by_coverage": entropy_thresholds,
    "country_threshold_km": COUNTRY_THRESHOLD_KM,
    "feature_order": ["top1_score", "margin", "entropy"],
    "calibration_set_size": len(calib),
}
with open(MODEL_OUT_PATH, "wb") as f:
    pickle.dump(abstention_artifact, f)
print(f"✅ مدل ذخیره شد در: {MODEL_OUT_PATH}")


# گام ۳: ارزیابی نهایی روی داده‌ی کاملاً جدا
print("\nدر حال بارگذاری داده‌ی کامل تست (clean + corrupted)...")
full = load_and_fix(ALL_CONF_PATH)
full["p_correct"] = clf.predict_proba(full[["top1_score", "margin", "entropy"]].values)[:, 1]

clean_final = full[full["corruption_type"] == "clean"].copy()
corrupted_final = full[full["corruption_type"] != "clean"].copy()

rows = []

# baseline بدون abstention (خط مرجع)
rows.append({
    "subset": "clean", "method": "no_abstention", "target_coverage": 1.0,
    "actual_coverage_pct": 100.0,
    "selective_country_acc_pct": (clean_final["error_km"] < COUNTRY_THRESHOLD_KM).mean() * 100,
})
rows.append({
    "subset": "corrupted", "method": "no_abstention", "target_coverage": 1.0,
    "actual_coverage_pct": 100.0,
    "selective_country_acc_pct": (corrupted_final["error_km"] < COUNTRY_THRESHOLD_KM).mean() * 100,
})

# روش threshold ساده (entropy)
for subset_name, subset_df in [("clean", clean_final), ("corrupted", corrupted_final)]:
    for cov, thr in entropy_thresholds.items():
        kept = subset_df[subset_df["entropy"] <= thr]
        rows.append({
            "subset": subset_name, "method": "entropy_threshold", "target_coverage": cov,
            "actual_coverage_pct": len(kept) / len(subset_df) * 100,
            "selective_country_acc_pct": (kept["error_km"] < COUNTRY_THRESHOLD_KM).mean() * 100 if len(kept) else np.nan,
        })

# روش Logistic Regression
for subset_name, subset_df in [("clean", clean_final), ("corrupted", corrupted_final)]:
    for p_thr in [0.0, 0.3, 0.4, 0.5, 0.6, 0.7]:
        kept = subset_df[subset_df["p_correct"] >= p_thr]
        rows.append({
            "subset": subset_name, "method": "logistic_regression", "target_coverage": p_thr,
            "actual_coverage_pct": len(kept) / len(subset_df) * 100,
            "selective_country_acc_pct": (kept["error_km"] < COUNTRY_THRESHOLD_KM).mean() * 100 if len(kept) else np.nan,
        })

comparison_df = pd.DataFrame(rows)
comparison_df.to_csv(COMPARISON_OUT_PATH, index=False)
print(f"\n✅ جدول مقایسه‌ی before/after ذخیره شد در: {COMPARISON_OUT_PATH}")
print(comparison_df.to_string(index=False))
