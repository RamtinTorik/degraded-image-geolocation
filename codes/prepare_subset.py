# prepare_subset.py (نسخه‌ی جدید 1095 تایی، اندازه‌ی subset قابل تنظیم)

import pandas as pd
import os
import shutil
import math
from paths import TEST_CSV, TEST_IMG_DIR, SUBSET_TEST_DIR

# ===== تنظیمات =====
OUTPUT_DIR = SUBSET_TEST_DIR

MAX_TOTAL = 1100            # <-- همینجا اندازه‌ی subset رو تغییر بده
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)

df = pd.read_csv(TEST_CSV, low_memory=False)
print("تعداد کل رکورد در test.csv:", len(df))

available_files = {}
for root, dirs, files in os.walk(TEST_IMG_DIR):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            file_id = os.path.splitext(f)[0]
            available_files[file_id] = os.path.join(root, f)

print("تعداد عکس واقعی پیدا شده:", len(available_files))

df["id_str"] = df["id"].astype(str)
df_avail = df[df["id_str"].isin(available_files.keys())].copy()
print("تعداد رکورد متادیتا که عکسشون موجوده:", len(df_avail))

N_PER_COUNTRY = math.ceil(MAX_TOTAL / df_avail["country"].nunique()) + 1

subset_list = []
for country, grp in df_avail.groupby("country"):
    n = min(len(grp), N_PER_COUNTRY)
    subset_list.append(grp.sample(n, random_state=RANDOM_SEED))
df_avail = pd.concat(subset_list, ignore_index=True)

if len(df_avail) > MAX_TOTAL:
    df_avail = df_avail.sample(MAX_TOTAL, random_state=RANDOM_SEED)

print("تعداد نهایی subset:", len(df_avail))
print("تعداد کشورهای متفاوت:", df_avail["country"].nunique())

for _, row in df_avail.iterrows():
    src = available_files[row["id_str"]]
    dst = os.path.join(OUTPUT_DIR, "images", os.path.basename(src))
    shutil.copy2(src, dst)

cols_to_keep = ["id", "latitude", "longitude", "country", "region", "sub-region", "city"]
df_avail[cols_to_keep].to_csv(os.path.join(OUTPUT_DIR, "subset_metadata.csv"), index=False)

print("✅ subset ساخته شد در:", OUTPUT_DIR)