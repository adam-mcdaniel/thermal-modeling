#!/bin/sh
# Regenerate every fitted constant from the raw data, in one command:    ./reproduce_constants.sh
#   gpu_frontier.h       the numbers predict_gpu.c uses, from your rocHPL trace (GPU power + junction sensor only)
#   cooling_frontier.h   the numbers for predict_cooling.c, from the cabinet telemetry (3Cab_POW.txt)
set -e
cd "$(dirname "$0")"
PYTHON="${PYTHON:-$HOME/miniconda3/envs/ampere_dev/bin/python}"
"$PYTHON" export_fit_data.py 2>/dev/null | grep -E 'By-hand|GPU [0-9]:'
cc -O2 fit_cooling_model.c -lm -o fit_cooling_model
echo; ./fit_cooling_model gpu frontier fit_data/gpu0.csv fit_data/gpu1.csv fit_data/gpu2.csv fit_data/gpu3.csv
echo; ./fit_cooling_model group frontier fit_data/frontier_group.csv 1400
