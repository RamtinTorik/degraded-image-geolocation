import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from paths import PROJECT_ROOT, RESULTS_DIR

BASE_DIR = PROJECT_ROOT
CALIB_PATH = RESULTS_DIR / "confidence" / "calib_test_with_confidence.csv"
ALL_CONF_PATH = RESULTS_DIR / "confidence" / "all_with_confidence.csv"
OUT_DIR = RESULTS_DIR / "ablation"
os.makedirs(OUT_DIR, exist_ok=True)

COUNTRY_ACC_KM = 750


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def load_fix(path):
    df = pd.read_csv(path)
    if "corruption_type" in df.columns:
        df["corruption_type"] = df["corruption_type"].astype(object)
        df["severity"] = df["severity"].astype(object)
        mask = df["corruption_type"].isna()
        df.loc[mask, "corruption_type"] = "clean"
        df.loc[mask, "severity"] = "none"
        if df["error_km"].isna().any():
            df.loc[mask, "error_km"] = haversine(df.loc[mask,"true_lat"], df.loc[mask,"true_lon"], df.loc[mask,"pred_lat"], df.loc[mask,"pred_lon"])
    return df


calib = load_fix(CALIB_PATH)
calib["correct"] = (calib["error_km"] < COUNTRY_ACC_KM).astype(int)
full = load_fix(ALL_CONF_PATH)
full["correct"] = full["error_km"] < COUNTRY_ACC_KM

clean_final = full[full["corruption_type"] == "clean"].copy()
corrupted_final = full[full["corruption_type"] != "clean"].copy()

TARGET_COVERAGES = [0.9, 0.8, 0.7, 0.6, 0.5]

# ========================================================================
# آزمایش ۱: مقایسه‌ی feature-هایی که به تنهایی یا با هم استفاده میشن
# ========================================================================
print("="*70)
print("آزمایش ۱: اهمیت هر Feature (تک‌تک در مقابل ترکیبی)")
print("="*70)

feature_sets = {
    "entropy_only": ["entropy"],
    "margin_only": ["margin"],
    "top1_only": ["top1_score"],
    "all_three": ["top1_score", "margin", "entropy"],
}

rows_feat = []
for name, feats in feature_sets.items():
    clf = LogisticRegression()
    clf.fit(calib[feats].values, calib["correct"].values)

    for subset_name, subset_df in [("clean", clean_final), ("corrupted", corrupted_final)]:
        p = clf.predict_proba(subset_df[feats].values)[:, 1]
        for cov in TARGET_COVERAGES:
            thr = np.quantile(clf.predict_proba(calib[feats].values)[:, 1], 1 - cov)
            kept = subset_df[p >= thr]
            acc = (kept["error_km"] < COUNTRY_ACC_KM).mean() * 100 if len(kept) else np.nan
            actual_cov = len(kept) / len(subset_df) * 100
            rows_feat.append({
                "feature_set": name, "subset": subset_name,
                "target_coverage": cov, "actual_coverage_pct": actual_cov,
                "selective_acc_pct": acc,
            })

feat_df = pd.DataFrame(rows_feat)
feat_df.to_csv(os.path.join(OUT_DIR, "ablation_feature_sets.csv"), index=False)
print(feat_df.pivot_table(index=["subset", "target_coverage"], columns="feature_set", values="selective_acc_pct").round(1))

# ========================================================================
# آزمایش ۲: حساسیت به threshold (coverage گسترده‌تر، شامل حالت‌های افراطی)
# ========================================================================
print("\n" + "="*70)
print("آزمایش ۲: حساسیت به Threshold (coverage از ۱۰۰٪ تا ۱۰٪)")
print("="*70)

clf_full = LogisticRegression()
clf_full.fit(calib[["top1_score", "margin", "entropy"]].values, calib["correct"].values)

fine_coverages = [1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
rows_sens = []
for subset_name, subset_df in [("clean", clean_final), ("corrupted", corrupted_final)]:
    p_calib = clf_full.predict_proba(calib[["top1_score","margin","entropy"]].values)[:, 1]
    p_subset = clf_full.predict_proba(subset_df[["top1_score","margin","entropy"]].values)[:, 1]
    for cov in fine_coverages:
        thr = np.quantile(p_calib, 1 - cov)
        kept = subset_df[p_subset >= thr]
        acc = (kept["error_km"] < COUNTRY_ACC_KM).mean() * 100 if len(kept) else np.nan
        actual_cov = len(kept) / len(subset_df) * 100
        rows_sens.append({"subset": subset_name, "target_coverage": cov, "actual_coverage_pct": actual_cov, "selective_acc_pct": acc})

sens_df = pd.DataFrame(rows_sens)
sens_df.to_csv(os.path.join(OUT_DIR, "ablation_threshold_sensitivity.csv"), index=False)
print(sens_df.pivot_table(index="target_coverage", columns="subset", values="selective_acc_pct").round(1))

# ========================================================================
# آزمایش ۳: calibration با فقط clean در مقابل calibration با clean+corrupted
# ========================================================================
print("\n" + "="*70)
print("آزمایش ۳: تاثیر دیدن Corruption در حین Calibration")
print("="*70)

# ترکیب calib (که فقط clean بود) با نمونه‌ای از corrupted final برای شبیه‌سازی سناریوی دوم
# (برای عدم نشتی، از نیمی از corrupted استفاده می‌کنیم و نیمه‌ی دیگه برای ارزیابی می‌مونه)
rng = np.random.default_rng(2024)
corrupted_ids = corrupted_final["id"].unique()
rng.shuffle(corrupted_ids)
half = len(corrupted_ids)//2
calib_corrupt_ids = set(corrupted_ids[:half])
eval_corrupt_ids = set(corrupted_ids[half:])

calib_mixed = pd.concat([calib, corrupted_final[corrupted_final["id"].isin(calib_corrupt_ids)]], ignore_index=True)
eval_corrupted_only = corrupted_final[corrupted_final["id"].isin(eval_corrupt_ids)]

clf_clean_only = LogisticRegression().fit(calib[["top1_score","margin","entropy"]].values, calib["correct"].values)
clf_mixed = LogisticRegression().fit(calib_mixed[["top1_score","margin","entropy"]].values, calib_mixed["correct"].values)

rows_cal = []
for name, clf_ in [("calibrated_on_clean_only", clf_clean_only), ("calibrated_on_clean_plus_corrupted", clf_mixed)]:
    p_calib_ref = clf_.predict_proba(calib[["top1_score","margin","entropy"]].values)[:,1]
    p_eval = clf_.predict_proba(eval_corrupted_only[["top1_score","margin","entropy"]].values)[:,1]
    for cov in TARGET_COVERAGES:
        thr = np.quantile(p_calib_ref, 1-cov)
        kept = eval_corrupted_only[p_eval >= thr]
        acc = (kept["error_km"] < COUNTRY_ACC_KM).mean()*100 if len(kept) else np.nan
        rows_cal.append({"calibration_strategy": name, "target_coverage": cov, "selective_acc_on_held_out_corrupted_pct": acc})

cal_df = pd.DataFrame(rows_cal)
cal_df.to_csv(os.path.join(OUT_DIR, "ablation_calibration_strategy.csv"), index=False)
print(cal_df.pivot_table(index="target_coverage", columns="calibration_strategy", values="selective_acc_on_held_out_corrupted_pct").round(1))

print(f"\n✅ همه‌ی نتایج ablation در پوشه‌ی زیر ذخیره شد:\n{OUT_DIR}")