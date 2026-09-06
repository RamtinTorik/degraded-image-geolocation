import pandas as pd
import os
import shutil
import math
from paths import TRAIN_CSV, TRAIN_IMG_DIR, SUBSET_TRAIN_CALIB_DIR

OUTPUT_DIR = SUBSET_TRAIN_CALIB_DIR
MAX_TOTAL = 1200   # برای calibration نیازی به subset بزرگ نیست
RANDOM_SEED = 121  # seed متفاوت از test، تا اریب نشیم

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)

df = pd.read_csv(TRAIN_CSV, low_memory=False)
print("تعداد کل رکورد در train.csv:", len(df))

available_files = {}
for root, dirs, files in os.walk(TRAIN_IMG_DIR):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            file_id = os.path.splitext(f)[0]
            available_files[file_id] = os.path.join(root, f)

print("تعداد عکس واقعی پیدا شده:", len(available_files))

df["id_str"] = df["id"].astype(str)
df_avail = df[df["id_str"].isin(available_files.keys())].copy()
print("تعداد رکورد متادیتا که عکسشون موجوده:", len(df_avail))

n_countries = df_avail["country"].nunique()
n_per_country = math.ceil(MAX_TOTAL / n_countries) + 1

subset_list = []
for country, grp in df_avail.groupby("country"):
    n = min(len(grp), n_per_country)
    subset_list.append(grp.sample(n, random_state=RANDOM_SEED))
df_avail = pd.concat(subset_list, ignore_index=True)

if len(df_avail) > MAX_TOTAL:
    df_avail = df_avail.sample(MAX_TOTAL, random_state=RANDOM_SEED)

print("تعداد نهایی subset calibration:", len(df_avail))
print("تعداد کشورهای متفاوت:", df_avail["country"].nunique())

for _, row in df_avail.iterrows():
    src = available_files[row["id_str"]]
    dst = os.path.join(OUTPUT_DIR, "images", os.path.basename(src))
    shutil.copy2(src, dst)

cols_to_keep = ["id", "latitude", "longitude", "country", "region", "sub-region", "city"]
df_avail[cols_to_keep].to_csv(os.path.join(OUTPUT_DIR, "subset_metadata.csv"), index=False)

print("✅ subset calibration ساخته شد در:", OUTPUT_DIR)