# run_baseline.py نسخه‌ی بهینه‌شده با batch processing

import sys, os
from paths import OSV5M_REPO, SUBSET_TEST_DIR, BASELINE_PATH, RESULTS_DIR

sys.path.append(str(OSV5M_REPO))
os.chdir(OSV5M_REPO)

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from models.huggingface import Geolocalizer

SUBSET_DIR = SUBSET_TEST_DIR
BATCH_SIZE = 16   # اگه RAM اذیت کرد، به 8 کاهش داده شود

geoloc = Geolocalizer.from_pretrained(BASELINE_PATH)
geoloc.eval()

df = pd.read_csv(os.path.join(SUBSET_DIR, "subset_metadata.csv"))
df["id_str"] = df["id"].astype(str)

results = []
with torch.no_grad():
    for start in tqdm(range(0, len(df), BATCH_SIZE)):
        batch_rows = df.iloc[start:start+BATCH_SIZE]
        imgs = []
        valid_rows = []
        for _, row in batch_rows.iterrows():
            img_path = os.path.join(SUBSET_DIR, "images", row["id_str"] + ".jpg")
            if not os.path.exists(img_path):
                continue
            img = Image.open(img_path).convert("RGB")
            imgs.append(geoloc.transform(img))
            valid_rows.append(row)

        if not imgs:
            continue

        x = torch.stack(imgs)  # (B, C, H, W)
        gps = geoloc(x)        # (B, 2)

        for i, row in enumerate(valid_rows):
            pred_lat = torch.rad2deg(gps[i][0]).item()
            pred_lon = torch.rad2deg(gps[i][1]).item()
            results.append({
                "id": row["id"],
                "true_lat": row["latitude"],
                "true_lon": row["longitude"],
                "pred_lat": pred_lat,
                "pred_lon": pred_lon,
                "country": row["country"],
            })

out_df = pd.DataFrame(results)
out_path = RESULTS_DIR / "clean" / "baseline_predictions.csv"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
out_df.to_csv(out_path, index=False)
print("اجرای baseline تموم شد، نتایج ذخیره شد در:", out_path)
