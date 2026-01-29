import os
import pandas as pd

SRC_ROOT = "/home/chaliol/Documents/catch_param_history"
OUT_CSV = "/home/chaliol/Documents/all_catchments_paramsHistory.csv"

dfs = []
n_loaded = 0
n_failed = 0

for catchment_id in sorted(os.listdir(SRC_ROOT)):
    catchment_dir = os.path.join(SRC_ROOT, catchment_id)
    csv_path = os.path.join(catchment_dir, "paramsHistory.csv")

    if not os.path.isdir(catchment_dir):
        continue

    if not os.path.isfile(csv_path):
        continue

    try:
        df = pd.read_csv(csv_path)
        df["catchment_id"] = catchment_id
        dfs.append(df)
        n_loaded += 1
    except Exception as e:
        print(f"⚠️ Failed to load {csv_path}: {e}")
        n_failed += 1

# Concatenate everything
if not dfs:
    raise RuntimeError("No paramsHistory.csv files were loaded.")

giant_df = pd.concat(dfs, ignore_index=True)

# Move catchment_id to first column (optional but nice)
cols = ["catchment_id"] + [c for c in giant_df.columns if c != "catchment_id"]
giant_df = giant_df[cols]

giant_df.to_csv(OUT_CSV, index=False)

print(f"✅ Loaded {n_loaded} catchments")
print(f"⚠️ Failed {n_failed} catchments")
print(f"�� Giant CSV written to: {OUT_CSV}")
print(f"�� Total rows: {len(giant_df)}")