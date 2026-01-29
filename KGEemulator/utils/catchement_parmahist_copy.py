import os
import shutil
import re

# Source and destination roots
SRC_ROOT = "/eos/jeodpp/data/projects/FLOODS-RIVER/LisfloodFP/GloFASv5_dataset/v5_allresultsfromLeonardo_23dec2025"
DST_ROOT = "/home/chaliol/Documents/catch_param_history"

# Subtree to ignore completely
EXCLUDED_PATH = os.path.join(
    SRC_ROOT,
    "SouthAmerica",
    "MagdalenaIDEAMter"
)

# Regex: accept ONLY pure numeric catchment IDs
VALID_CATCHMENT_REGEX = re.compile(r"^\d+$")

os.makedirs(DST_ROOT, exist_ok=True)

n_copied = 0
n_skipped = 0

for root, dirs, files in os.walk(SRC_ROOT, topdown=True):

    # ---- PRUNE excluded subtree ----
    if os.path.commonpath([root, EXCLUDED_PATH]) == EXCLUDED_PATH:
        dirs[:] = []   # stop descending
        continue

    if "paramsHistory.csv" not in files:
        continue

    catchment_dir = os.path.basename(root)

    # Skip catchments with suffixes (_A, _test, etc.)
    if not VALID_CATCHMENT_REGEX.match(catchment_dir):
        n_skipped += 1
        continue

    src_csv = os.path.join(root, "paramsHistory.csv")

    dst_catchment_dir = os.path.join(DST_ROOT, catchment_dir)
    os.makedirs(dst_catchment_dir, exist_ok=True)

    dst_csv = os.path.join(dst_catchment_dir, "paramsHistory.csv")

    shutil.copy2(src_csv, dst_csv)
    n_copied += 1

print(f"✅ Copied {n_copied} paramsHistory.csv files")
print(f"⏭️  Skipped {n_skipped} catchments (suffixes or excluded)")

