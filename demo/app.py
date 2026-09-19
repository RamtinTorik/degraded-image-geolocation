import sys, os
import io, base64, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# مسیرها
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OSV5M_REPO = PROJECT_ROOT / "codes" / "github-osv5m" / "osv5m"
CODES_DIR = PROJECT_ROOT / "codes"
DEMO_DIR = PROJECT_ROOT / "demo"
STATIC_DIR = DEMO_DIR / "static"

sys.path.append(str(OSV5M_REPO))
sys.path.append(str(CODES_DIR))

os.chdir(OSV5M_REPO)  # برای پیدا کردن quadtree_10_1000.csv

from models.huggingface import Geolocalizer
from confidence_utils import compute_confidence_features
from corruptions import CORRUPTIONS, SEVERITIES

BASELINE_PATH = PROJECT_ROOT / "codes" / "baseline-huggingface" / "baseline"
RESULTS_DIR = PROJECT_ROOT / "results"
ABSTENTION_MODEL_PATH = RESULTS_DIR / "abstention_model.pkl"
SUMMARY_CSV = RESULTS_DIR / "summary_metrics.csv"
COMPARISON_CSV = RESULTS_DIR / "abstention_comparison.csv"
ALL_CONFIDENCE_CSV = RESULTS_DIR / "confidence" / "all_with_confidence.csv"
TEST_METADATA_CSV = PROJECT_ROOT / "datasets" / "subset_test" / "subset_metadata.csv"

COUNTRY_ACC_THRESHOLD_KM = 750
CORRUPTION_ORDER = ["blur", "jpeg", "noise", "resize", "crop", "lowlight", "fog"]


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def img_to_thumb_b64(img, max_size=220, quality=70):
    thumb = img.copy()
    thumb.thumbnail((max_size, max_size))
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# بارگذاری مدل و داده‌ها (فقط یک‌بار، هنگام استارت)
print("در حال بارگذاری مدل baseline...")
geoloc = Geolocalizer.from_pretrained(BASELINE_PATH)
geoloc.eval()
print("مدل آماده شد.")

with open(ABSTENTION_MODEL_PATH, "rb") as f:
    abstention = pickle.load(f)
clf = abstention["logistic_regression"]

print("در حال بارگذاری متادیتای تست (برای تشخیص خودکار ground truth)...")
test_meta = pd.read_csv(TEST_METADATA_CSV)
test_meta["id_str"] = test_meta["id"].astype(str)
test_meta_lookup = test_meta.set_index("id_str").to_dict(orient="index")
print(f"{len(test_meta_lookup)} رکورد متادیتای تست بارگذاری شد.")


def fix_confidence_df(path):
    df = pd.read_csv(path)
    df["corruption_type"] = df["corruption_type"].astype(object)
    df["severity"] = df["severity"].astype(object)
    mask = df["corruption_type"].isna()
    df.loc[mask, "corruption_type"] = "clean"
    df.loc[mask, "severity"] = "none"
    if df["error_km"].isna().any():
        df.loc[mask, "error_km"] = haversine(
            df.loc[mask, "true_lat"], df.loc[mask, "true_lon"],
            df.loc[mask, "pred_lat"], df.loc[mask, "pred_lon"]
        )
    return df


print("در حال بارگذاری و پیش‌پردازش نتایج تحلیلی (۱۶۵۰۰ رکورد)...")
all_conf_df = fix_confidence_df(ALL_CONFIDENCE_CSV)
all_conf_df["p_correct"] = clf.predict_proba(all_conf_df[["top1_score", "margin", "entropy"]].values)[:, 1]
all_conf_df["rejected"] = all_conf_df["p_correct"] < 0.5
all_conf_df["correct"] = all_conf_df["error_km"] < COUNTRY_ACC_THRESHOLD_KM

summary_df = pd.read_csv(SUMMARY_CSV)
comparison_df = pd.read_csv(COMPARISON_CSV)

# پیش‌محاسبه‌ی جدول جامع "تاثیر هر corruption" (entropy, error, reject rate, acc before/after)
corruption_impact_df = all_conf_df.groupby(["corruption_type", "severity"]).apply(
    lambda g: pd.Series({
        "n": len(g),
        "mean_entropy": g["entropy"].mean(),
        "mean_margin": g["margin"].mean(),
        "mean_top1": g["top1_score"].mean(),
        "mean_error_km": g["error_km"].mean(),
        "country_acc_before_pct": g["correct"].mean() * 100,
        "reject_rate_pct": g["rejected"].mean() * 100,
        "country_acc_after_pct": (g.loc[~g["rejected"], "correct"].mean() * 100) if (~g["rejected"]).any() else np.nan,
    }), include_groups=False
).reset_index()
order = ["clean"] + CORRUPTION_ORDER
corruption_impact_df["corruption_type"] = pd.Categorical(corruption_impact_df["corruption_type"], categories=order, ordered=True)
corruption_impact_df = corruption_impact_df.sort_values(["corruption_type", "severity"])

# هیستوگرام entropy: clean در مقابل corrupted
bins = np.linspace(0, all_conf_df["entropy"].max(), 21)
clean_entropy = all_conf_df.loc[all_conf_df["corruption_type"] == "clean", "entropy"]
corrupted_entropy = all_conf_df.loc[all_conf_df["corruption_type"] != "clean", "entropy"]
clean_hist, _ = np.histogram(clean_entropy, bins=bins)
corrupted_hist, _ = np.histogram(corrupted_entropy, bins=bins, density=False)
corrupted_hist_norm = corrupted_hist / (all_conf_df["corruption_type"] != "clean").sum() * len(clean_entropy)  # نرمال‌سازی برای مقایسه‌ی شکلی

# تحلیل decile: رابطه‌ی entropy با خطا (نسخه‌ی ریزتر از quartile)
all_conf_df["entropy_decile"] = pd.qcut(all_conf_df["entropy"], 10, labels=False, duplicates="drop")
decile_df = all_conf_df.groupby("entropy_decile").agg(
    mean_entropy=("entropy", "mean"),
    mean_error_km=("error_km", "mean"),
    country_acc_pct=("correct", lambda x: x.mean() * 100),
    n=("entropy", "count"),
).reset_index()

# اهمیت فیچرها (از ضرایب Logistic Regression)
feat_names = ["top1_score", "margin", "entropy"]
coefs = clf.coef_[0]
importance_df = pd.DataFrame({
    "feature": feat_names,
    "coefficient": coefs,
    "abs_importance_pct": np.abs(coefs) / np.abs(coefs).sum() * 100,
})

print("همه‌ی پیش‌محاسبات آماده شد. سرور در حال اجراست.")

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/summary")
def api_summary():
    return jsonify(summary_df.to_dict(orient="records"))


@app.route("/api/comparison")
def api_comparison():
    return jsonify(comparison_df.to_dict(orient="records"))


@app.route("/api/corruption_impact")
def api_corruption_impact():
    return jsonify(corruption_impact_df.to_dict(orient="records"))


@app.route("/api/entropy_histogram")
def api_entropy_histogram():
    return jsonify({
        "bin_edges": bins.tolist(),
        "clean_counts": clean_hist.tolist(),
        "corrupted_counts": corrupted_hist_norm.tolist(),
    })


@app.route("/api/error_vs_entropy_decile")
def api_decile():
    return jsonify(decile_df.to_dict(orient="records"))


@app.route("/api/feature_importance")
def api_feature_importance():
    return jsonify(importance_df.to_dict(orient="records"))


def run_single_inference(img):
    with torch.no_grad():
        x = geoloc.transform(img).unsqueeze(0)
        output = geoloc.head(geoloc.mid(geoloc.backbone({"img": x})), None)
        gps = output["gps"]
        logits = output["label"]
        conf = compute_confidence_features(logits)

    pred_lat = torch.rad2deg(gps[0][0]).item()
    pred_lon = torch.rad2deg(gps[0][1]).item()
    top1 = conf["top1_score"][0].item()
    margin = conf["margin"][0].item()
    entropy = conf["entropy"][0].item()
    p_correct = float(clf.predict_proba([[top1, margin, entropy]])[0, 1])
    decision = "accept" if p_correct >= 0.5 else "reject"
    return {
        "pred_lat": pred_lat, "pred_lon": pred_lon,
        "top1_score": top1, "margin": margin, "entropy": entropy,
        "p_correct": p_correct, "decision": decision,
    }


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "image" not in request.files:
        return jsonify({"error": "no image uploaded"}), 400

    img_file = request.files["image"]
    original_name = secure_filename(img_file.filename or "")
    file_id = os.path.splitext(original_name)[0]

    mode = request.form.get("mode", "single")  # 'single' یا 'all'
    corruption_type = request.form.get("corruption_type", "none")
    severity = request.form.get("severity", "1")

    original_img = Image.open(img_file.stream).convert("RGB")

    # تشخیص خودکار که آیا این عکس از دیتاست تسته؟
    gt = test_meta_lookup.get(file_id)
    is_known = gt is not None
    true_lat = gt["latitude"] if is_known else None
    true_lon = gt["longitude"] if is_known else None
    true_country = gt["country"] if is_known else None

    def build_variant(name, sev, corrupted_img):
        r = run_single_inference(corrupted_img)
        r["corruption_type"] = name
        r["severity"] = sev if sev else "none"
        r["thumbnail_b64"] = img_to_thumb_b64(corrupted_img)
        if is_known:
            r["error_km"] = haversine(true_lat, true_lon, r["pred_lat"], r["pred_lon"])
        return r

    results = []
    if mode == "all":
        results.append(build_variant("clean", "none", original_img))
        for cname in CORRUPTION_ORDER:
            for sev in SEVERITIES:
                corrupted_img = CORRUPTIONS[cname](original_img, sev)
                results.append(build_variant(cname, sev, corrupted_img))
    else:
        img_to_use = original_img
        if corruption_type != "none" and corruption_type in CORRUPTIONS:
            img_to_use = CORRUPTIONS[corruption_type](original_img, int(severity))
        results.append(build_variant(corruption_type, severity if corruption_type != "none" else "none", img_to_use))

    return jsonify({
        "is_known_test_image": is_known,
        "true_lat": true_lat,
        "true_lon": true_lon,
        "true_country": true_country,
        "mode": mode,
        "results": results,
    })


if __name__ == "__main__":
    app.run(debug=False, port=5000)
