import sys, os
from paths import OSV5M_REPO, CODES_DIR, SUBSET_TEST_DIR, BASELINE_PATH, RESULTS_DIR

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
from corruptions import CORRUPTIONS, SEVERITIES

SUBSET_DIR = SUBSET_TEST_DIR
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

def run_inference_with_confidence(corruption_fn, severity, tag):
    results = []
    with torch.no_grad():
        for start in tqdm(range(0, len(df), BATCH_SIZE), desc=tag):
            batch_rows = df.iloc[start:start + BATCH_SIZE]
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
            output = geoloc.head(geoloc.mid(geoloc.backbone({"img": x})), None)
            gps, logits = output["gps"], output["label"]
            conf = compute_confidence_features(logits)
            for i, row in enumerate(valid_rows):
                pred_lat = torch.rad2deg(gps[i][0]).item()
                pred_lon = torch.rad2deg(gps[i][1]).item()
                results.append({
                    "id": row["id"], "true_lat": row["latitude"], "true_lon": row["longitude"],
                    "pred_lat": pred_lat, "pred_lon": pred_lon, "country": row["country"],
                    "corruption_type": tag.split("_sev")[0] if "_sev" in tag else "clean",
                    "severity": tag.split("_sev")[1] if "_sev" in tag else "none",
                    "error_km": haversine(row["latitude"], row["longitude"], pred_lat, pred_lon),
                    "top1_score": conf["top1_score"][i].item(),
                    "margin": conf["margin"][i].item(),
                    "entropy": conf["entropy"][i].item(),
                })
    return pd.DataFrame(results)

all_dfs = []
combos = [(None, None, "clean")] + [(CORRUPTIONS[n], s, f"{n}_sev{s}") for n in CORRUPTIONS for s in SEVERITIES]
for fn, sev, tag in combos:
    out_csv = os.path.join(RESULTS_DIR, f"{tag}_with_confidence.csv")
    if os.path.exists(out_csv):
        print(f"رد شد: {tag}")
        out_df = pd.read_csv(out_csv)
    else:
        out_df = run_inference_with_confidence(fn, sev, tag)
        out_df.to_csv(out_csv, index=False)
        print(f"ذخیره شد: {out_csv}")
    all_dfs.append(out_df)

full_df = pd.concat(all_dfs, ignore_index=True)
full_path = os.path.join(RESULTS_DIR, "all_with_confidence.csv")
full_df.to_csv(full_path, index=False)
print(f"\nفایل جامع ذخیره شد: {full_path}")
print(f"تعداد کل رکورد: {len(full_df)}")
