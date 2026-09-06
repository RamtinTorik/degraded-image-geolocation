import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CODES_DIR = PROJECT_ROOT / "codes"
OSV5M_REPO = CODES_DIR / "github-osv5m" / "osv5m"
# Use the public Hugging Face model by default. Set OSV5M_BASELINE to a local
# model directory when the weights have already been downloaded.
BASELINE_PATH = os.environ.get("OSV5M_BASELINE", "osv5m/baseline")
DATASETS_DIR = PROJECT_ROOT / "datasets"
RESULTS_DIR = PROJECT_ROOT / "results"

# The raw OSV5M dataset is too large to keep in this repository. Override this
# location with OSV5M_DATA_DIR when it is stored outside the project folder.
OSV5M_DATA_DIR = Path(os.environ.get("OSV5M_DATA_DIR", PROJECT_ROOT / "data" / "osv5m"))
TEST_CSV = OSV5M_DATA_DIR / "test.csv"
TEST_IMG_DIR = OSV5M_DATA_DIR / "images" / "test" / "extracted"
TRAIN_CSV = OSV5M_DATA_DIR / "train.csv"
TRAIN_IMG_DIR = OSV5M_DATA_DIR / "images" / "train" / "extracted"

SUBSET_TEST_DIR = DATASETS_DIR / "subset_test"
SUBSET_TEST_CALIB_DIR = DATASETS_DIR / "subset_test_calib"
SUBSET_TRAIN_CALIB_DIR = DATASETS_DIR / "subset_train_calib"