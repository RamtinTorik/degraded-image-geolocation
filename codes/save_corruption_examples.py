# save_corruption_examples.py (جدید — برای گزارش)

import sys, os
from paths import CODES_DIR, SUBSET_TEST_DIR, RESULTS_DIR

sys.path.append(str(CODES_DIR))

from PIL import Image
from corruptions import CORRUPTIONS, SEVERITIES

SUBSET_DIR = SUBSET_TEST_DIR
OUTPUT_DIR = RESULTS_DIR / "example_corrupted_images"
N_EXAMPLES = 5   # تعداد عکس‌های نمونه برای گزارش (می‌تونی بیشتر/کمترش کنی)

os.makedirs(OUTPUT_DIR, exist_ok=True)

image_dir = os.path.join(SUBSET_DIR, "images")
all_images = sorted(os.listdir(image_dir))[:N_EXAMPLES]

for img_name in all_images:
    img_id = os.path.splitext(img_name)[0]
    original = Image.open(os.path.join(image_dir, img_name)).convert("RGB")

    # پوشه‌ی مخصوص این عکس
    img_folder = os.path.join(OUTPUT_DIR, img_id)
    os.makedirs(img_folder, exist_ok=True)

    # ذخیره‌ی نسخه‌ی clean
    original.save(os.path.join(img_folder, "clean.jpg"))

    # ذخیره‌ی همه‌ی نسخه‌های corrupted
    for corruption_name, fn in CORRUPTIONS.items():
        for severity in SEVERITIES:
            corrupted = fn(original, severity)
            save_path = os.path.join(img_folder, f"{corruption_name}_sev{severity}.jpg")
            corrupted.save(save_path)

    print(f"✅ نمونه‌های {img_id} ذخیره شدن در {img_folder}")

print(f"\n🎉 تمام شد. {N_EXAMPLES} عکس نمونه × {len(CORRUPTIONS)} نوع corruption × {len(SEVERITIES)} severity ذخیره شد.")
print(f"مسیر: {OUTPUT_DIR}")