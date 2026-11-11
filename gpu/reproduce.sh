#!/bin/sh
# Regenerate the samples and gpu_fit.h from the Score-P traces, and build everything, in one command:    ./reproduce.sh
set -e
cd "$(dirname "$0")"
PYTHON="${PYTHON:-$HOME/miniconda3/envs/ampere_dev/bin/python}"
"$PYTHON" make_samples.py 2>/dev/null | grep ' GPU '
cc -O2 fit_gpu.c -lm -o fit_gpu
./fit_gpu gpu_fit.h samples/gpu0.txt samples/gpu1.txt samples/gpu2.txt samples/gpu3.txt
cc -O2 predict_gpu.c -lm -o predict_gpu
"$PYTHON" fit_water.py
cc -O2 predict_gpu.c -lm -o predict_gpu
cc -O2 replay_gpu.c -lm -o replay_gpu
