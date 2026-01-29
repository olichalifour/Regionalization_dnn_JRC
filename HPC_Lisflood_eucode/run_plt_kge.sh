#!/bin/sh
source /home/chaliol/miniforge3/bin/activate
source activate lisflood
set -euo pipefail
export NUMBA_THREADING_LAYER='tbb'
export NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR="/local0/chaliol/numba_cache_dir"
cd /home/chaliol
time python /home/chaliol/analyse_kge.py
rm -Rf /local0/chaliol/numba_cache_dir