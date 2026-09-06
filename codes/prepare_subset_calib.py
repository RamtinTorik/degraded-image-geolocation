import pandas as pd
import os
import shutil
import math
from paths import TEST_CSV, TEST_IMG_DIR, SUBSET_TEST_DIR, SUBSET_TEST_CALIB_DIR

EXISTING_SUBSET_METADATA = SUBSET_TEST_DIR / "subset_metadata.csv"
OUTPUT_DIR = SUBSET_TEST_CALIB_DIR

MAX_TOTAL = 500
RANDOM_SEED = 777   # seed متفاوت از قبلی‌ها

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)

# ===== id هایی که قبلاً توی subset اصلی (۱۱۰۰ تایی) استفاده شدن رو کنار می‌ذاریم =====
existing_ids = set(pd.read_csv(EXISTING_SUBSET_METADATA)["id"].astype(str))
print("تعداد id های قبلاً استفاده‌شده (باید حذف بشن):", len(existing_ids))

df = pd.read_csv(TEST_CSV, low_memory=False)
df["id_str"] = df["id"].astype(str)

available_files = {}
for root, dirs, files in os.walk(TEST_IMG_DIR):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            file_id = os.path.splitext(f)[0]
            available_files[file_id] = os.path.join(root, f)

df_avail = df[df["id_str"].isin(available_files.keys())].copy()

# ===== حذف عکس‌هایی که در subset اصلی هستن (جلوگیری از نشتی/overlap) =====
df_avail = df_avail[~df_avail["id_str"].isin(existing_ids)].copy()
print("تعداد رکورد باقی‌مونده بعد از حذف overlap:", len(df_avail))

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

# ===== تایید نهایی نبود overlap =====
overlap_check = set(df_avail["id_str"]) & existing_ids
print("بررسی overlap (باید 0 باشه):", len(overlap_check))

for _, row in df_avail.iterrows():
    src = available_files[row["id_str"]]
    dst = os.path.join(OUTPUT_DIR, "images", os.path.basename(src))
    shutil.copy2(src, dst)

cols_to_keep = ["id", "latitude", "longitude", "country", "region", "sub-region", "city"]
df_avail[cols_to_keep].to_csv(os.path.join(OUTPUT_DIR, "subset_metadata.csv"), index=False)

print("✅ subset calibration (بدون overlap با test اصلی) ساخته شد در:", OUTPUT_DIR)