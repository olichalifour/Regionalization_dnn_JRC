#!/bin/bash
#PBS -q long
#PBS -l walltime=120:00:00
#PBS -l nodes=1:ppn=32
#PBS -N lisflood_parallel
#PBS -j oe

# ---- PREVENT GDAL/PROJ unbound-variable crashes ----
export GDAL_DATA="${GDAL_DATA:-}"
export GDAL_DRIVER_PATH="${GDAL_DRIVER_PATH:-}"
export PROJ_LIB="${PROJ_LIB:-}"
export GDAL_CONFIG="${GDAL_CONFIG:-}"

echo "=============================================="
echo " LISFLOOD PARALLEL JOB STARTED"
echo " Host: $(hostname)"
echo " JobID: $PBS_JOBID"
echo "=============================================="

# --------------------------------------------------
# ENVIRONMENT
# --------------------------------------------------
source /home/chaliol/miniforge3/etc/profile.d/conda.sh
conda activate lisflood

set -euo pipefail

export NUMBA_THREADING_LAYER=tbb
export NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR="/local0/chaliol/numba_cache_${PBS_JOBID}"
#mkdir -p "$NUMBA_CACHE_DIR"

# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE_DIR="/home/chaliol/Lisflood_param_test_glofas5"
N_PARALLEL=25

# --------------------------------------------------
# DISCOVER CATCHMENTS
# --------------------------------------------------
mapfile -t CATCHMENTS < <(ls -d "$BASE_DIR"/*/ | xargs -n1 basename)

echo "Found ${#CATCHMENTS[@]} catchments"
echo "Running $N_PARALLEL in parallel"
echo "----------------------------------------------"

# --------------------------------------------------
# FUNCTION
# --------------------------------------------------
run_catchment () {
    CID="$1"
    CDIR="$BASE_DIR/$CID"
    SDIR="$CDIR/settings"

    echo ""
    echo ">>> [$CID] START"

    PRERUN_FILES=("$SDIR"/*PreRun*.xml)
    RUN_FILES=("$SDIR"/*Run*.xml)

    shopt -u nullglob

    if [[ ${#PRERUN_FILES[@]} -eq 0 || ${#RUN_FILES[@]} -eq 0 ]]; then
      echo "!!! [$CID] Missing XML files"
      echo "    Found PreRun: ${#PRERUN_FILES[@]}"
      echo "    Found Run:    ${#RUN_FILES[@]}"
      return
    fi

    PRERUN="${PRERUN_FILES[0]}"
    RUN="${RUN_FILES[0]}"

    echo "[$CID] PreRun XML: $PRERUN"
    echo "[$CID] Run XML:    $RUN"

    echo "[$CID] PreRun"
    python /home/chaliol/lisflood-code/src/lisf1.py "$PRERUN"

    echo "[$CID] Run"
    python /home/chaliol/lisflood-code/src/lisf1.py "$RUN"

    echo "<<< [$CID] DONE"
}

export -f run_catchment
export BASE_DIR

# --------------------------------------------------
# PARALLEL EXECUTION
# --------------------------------------------------
printf "%s\n" "${CATCHMENTS[@]}" | parallel -j "$N_PARALLEL" run_catchment {}

# --------------------------------------------------
# CLEANUP
# --------------------------------------------------
rm -rf "$NUMBA_CACHE_DIR"

echo "=============================================="
echo " ALL CATCHMENTS COMPLETED"
echo "=============================================="