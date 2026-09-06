# run_corruption_benchmark.py (اسکریپت اصلی هفته ۲)

import sys, os
from paths import OSV5M_REPO, CODES_DIR, SUBSET_TEST_DIR, BASELINE_PATH, RESULTS_DIR

sys.path.append(str(OSV5M_REPO))
os.chdir(OSV5M_REPO)
sys.path.append(str(CODES_DIR))  # مسیر corruptions.py و evaluate.py

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from models.huggingface import Geolocalizer
from corruptions import CORRUPTIONS, SEVERITIES
from evaluate import evaluate_predictions, append_to_summary

SUBSET_DIR = SUBSET_TEST_DIR
RESULTS_DIR = RESULTS_DIR / "corrupted"
SUMMARY_PATH = RESULTS_DIR.parent / "summary_metrics.csv"
BATCH_SIZE = 16

os.makedirs(RESULTS_DIR, exist_ok=True)

print("در حال بارگذاری مدل baseline...")
geoloc = Geolocalizer.from_pretrained(BASELINE_PATH)
geoloc.eval()

df = pd.read_csv(os.path.join(SUBSET_DIR, "subset_metadata.csv"))
df["id_str"] = df["id"].astype(str)

def run_inference_with_corruption(corruption_fn, severity):
    results = []
    with torch.no_grad():
        for start in tqdm(range(0, len(df), BATCH_SIZE), desc=f"{corruption_fn.__name__ if corruption_fn else 'clean'}_sev{severity}"):
            batch_rows = df.iloc[start:start+BATCH_SIZE]
            imgs, valid_rows = [], []
            for _, row in batch_rows.iterrows():
                img_path = os.path.join(SUBSET_DIR, "images", row["id_str"] + ".jpg")
                if not os.path.exists(img_path):
                    continue
                img = Image.open(img_path).convert("RGB")
                if corruption_fn is not None:
                    img = corruption_fn(img, severity)
                imgs.append(geoloc.transform(img))
                valid_rows.append(row)

            if not imgs:
                continue
            x = torch.stack(imgs)
            gps = geoloc(x)
            for i, row in enumerate(valid_rows):
                results.append({
                    "id": row["id"],
                    "true_lat": row["latitude"],
                    "true_lon": row["longitude"],
                    "pred_lat": torch.rad2deg(gps[i][0]).item(),
                    "pred_lon": torch.rad2deg(gps[i][1]).item(),
                    "country": row["country"],
                })
    return pd.DataFrame(results)


# ===== حلقه‌ی اصلی روی تمام corruption × severity =====
all_combos = [(name, sev) for name in CORRUPTIONS for sev in SEVERITIES]
print(f"تعداد کل حالت‌های corruption: {len(all_combos)}")

for corruption_name, severity in all_combos:
    out_csv = os.path.join(RESULTS_DIR, f"{corruption_name}_sev{severity}_predictions.csv")

    if os.path.exists(out_csv):
        print(f"⏭️  رد شد (از قبل موجوده): {corruption_name} severity {severity}")
    else:
        corruption_fn = CORRUPTIONS[corruption_name]
        out_df = run_inference_with_corruption(corruption_fn, severity)
        out_df.to_csv(out_csv, index=False)
        print(f"✅ ذخیره شد: {out_csv}")

    metrics, _ = evaluate_predictions(out_csv, corruption_type=corruption_name, severity=str(severity))
    append_to_summary(metrics, SUMMARY_PATH)
    print(f"   -> mean_error_km={metrics['mean_error_km']:.1f} | acc_country_750km={metrics['acc_country_750km']:.1f}%")

print("\n🎉 کل benchmark هفته ۲ تموم شد. جدول نهایی:")
print(pd.read_csv(SUMMARY_PATH))