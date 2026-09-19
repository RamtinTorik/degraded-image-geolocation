import sys, os
from paths import OSV5M_REPO, CODES_DIR, SUBSET_TEST_DIR, BASELINE_PATH, RESULTS_DIR

sys.path.append(str(OSV5M_REPO))
os.chdir(OSV5M_REPO)
sys.path.append(str(CODES_DIR))

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from models.huggingface import Geolocalizer
from confidence_utils import compute_confidence_features
from corruptions import CORRUPTIONS, SEVERITIES

SUBSET_DIR = SUBSET_TEST_DIR
RESULTS_DIR = RESULTS_DIR / "confidence"
BATCH_SIZE = 16

os.makedirs(RESULTS_DIR, exist_ok=True)

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

            # دسترسی مستقیم به کل خروجی مدل
            output = geoloc.head(geoloc.mid(geoloc.backbone({"img": x})), None)
            gps = output["gps"]
            logits = output["label"]

            conf = compute_confidence_features(logits)

            for i, row in enumerate(valid_rows):
                results.append({
                    "id": row["id"],
                    "true_lat": row["latitude"],
                    "true_lon": row["longitude"],
                    "pred_lat": torch.rad2deg(gps[i][0]).item(),
                    "pred_lon": torch.rad2deg(gps[i][1]).item(),
                    "country": row["country"],
                    "top1_score": conf["top1_score"][i].item(),
                    "margin": conf["margin"][i].item(),
                    "entropy": conf["entropy"][i].item(),
                })
    return pd.DataFrame(results)


if __name__ == "__main__":
    # برای اطمینان از درستی کد
    out_df = run_inference_with_confidence(None, None, "clean_with_confidence")
    out_path = os.path.join(RESULTS_DIR, "clean_with_confidence.csv")
    out_df.to_csv(out_path, index=False)
    print(f"✅ ذخیره شد: {out_path}")
    print(out_df[["top1_score", "margin", "entropy"]].describe())
