import sys, os
from paths import OSV5M_REPO, CODES_DIR, SUBSET_TEST_CALIB_DIR, BASELINE_PATH, RESULTS_DIR

sys.path.append(str(OSV5M_REPO))
os.chdir(OSV5M_REPO)
sys.path.append(str(CODES_DIR))

import pandas as pd
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from models.huggingface import Geolocalizer
from confidence_utils import compute_confidence_features

SUBSET_DIR = SUBSET_TEST_CALIB_DIR
RESULTS_DIR = RESULTS_DIR / "confidence"
BATCH_SIZE = 16

os.makedirs(RESULTS_DIR, exist_ok=True)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

print("در حال بارگذاری مدل baseline...")
geoloc = Geolocalizer.from_pretrained(BASELINE_PATH)
geoloc.eval()

df = pd.read_csv(os.path.join(SUBSET_DIR, "subset_metadata.csv"))
df["id_str"] = df["id"].astype(str)

results = []
with torch.no_grad():
    for start in tqdm(range(0, len(df), BATCH_SIZE)):
        batch_rows = df.iloc[start:start+BATCH_SIZE]
        imgs, valid_rows = [], []
        for _, row in batch_rows.iterrows():
            img_path = os.path.join(SUBSET_DIR, "images", row["id_str"] + ".jpg")
            if not os.path.exists(img_path):
                continue
            img = Image.open(img_path).convert("RGB")
            imgs.append(geoloc.transform(img))
            valid_rows.append(row)
        if not imgs:
            continue
        x = torch.stack(imgs)
        output = geoloc.head(geoloc.mid(geoloc.backbone({"img": x})), None)
        gps, logits = output["gps"], output["label"]
        conf = compute_confidence_features(logits)
        for i, row in enumerate(valid_rows):
            pred_lat = torch.rad2deg(gps[i][0]).item()
            pred_lon = torch.rad2deg(gps[i][1]).item()
            results.append({
                "id": row["id"], "true_lat": row["latitude"], "true_lon": row["longitude"],
                "pred_lat": pred_lat, "pred_lon": pred_lon, "country": row["country"],
                "error_km": haversine(row["latitude"], row["longitude"], pred_lat, pred_lon),
                "top1_score": conf["top1_score"][i].item(),
                "margin": conf["margin"][i].item(),
                "entropy": conf["entropy"][i].item(),
            })

out_df = pd.DataFrame(results)
out_path = os.path.join(RESULTS_DIR, "calib_test_with_confidence.csv")
out_df.to_csv(out_path, index=False)
print(f"✅ ذخیره شد: {out_path}")
print(out_df[["top1_score","margin","entropy","error_km"]].describe())