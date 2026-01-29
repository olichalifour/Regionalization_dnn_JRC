#!/bin/bash


# --- Activate LISFLOOD environment ---
source /home/chaliol/miniforge3/bin/activate
conda activate lisflood

set -euo pipefail

export NUMBA_THREADING_LAYER='tbb'
export NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR="/local0/chaliol/numba_cache_dir"

# Base directory containing all catchments
BASE_DIR="/home/chaliol/Lisflood_param_test_glofas5/85"

echo "Starting LISFLOOD batch processing..."
echo "---------------------------------------"

SETTINGS_DIR="$BASE_DIR/settings"

# Construct paths to XML files
PRERUN_FILE="/home/chaliol/Lisflood_param_test_glofas5/85/settings/OSLisfloodGloFASv5calibration_v1_Europe_ElbeOderEmsWeserPreRun0.xml"
RUN_FILE="/home/chaliol/Lisflood_param_test_glofas5/85/settings/OSLisfloodGloFASv5calibration_v1_Europe_ElbeOderEmsWeserRun0.xml"

echo "Running PreRun..."
python /home/chaliol/lisflood-code/src/lisf1.py "$PRERUN_FILE"


echo "PreRun finished for catchment $CATCHMENT_ID"

# Loop over catchment folders
#for CATCHMENT_DIR in "$BASE_DIR"/*; do
    # Make sure this is a directory
    #[ -d "$CATCHMENT_DIR" ] || continue

    #CATCHMENT_ID=$(basename "$CATCHMENT_DIR")
    #SETTINGS_DIR="$CATCHMENT_DIR/settings"

    # Construct paths to XML files
    #PRERUN_FILE="$SETTINGS_DIR/settings_lisflood_wusewregion_FULLinit_GloFASnext-PreRunX.xml"
    #RUN_FILE="$SETTINGS_DIR/settings_lisflood_wusewregion_FULLinit_GloFASnext-RunX.xml"

    #echo ""
    #echo ">>> Processing catchment: $CATCHMENT_ID"
    #echo "    PreRun: $PRERUN_FILE"
    #echo "    Run:    $RUN_FILE"

    # --- Run PreRun ---
    #echo "Running PreRun..."
    #lisflood "$PRERUN_FILE"

    #echo "PreRun finished for catchment $CATCHMENT_ID"

    # --- Run Run ---
    #echo "Running main Run..."
    #lisflood "$RUN_FILE"

    #echo "Run finished for catchment $CATCHMENT_ID"
    #echo "---------------------------------------"
#done

# Cleanup
rm -rf /local0/chaliol/numba_cache_dir
echo "All catchments completed."