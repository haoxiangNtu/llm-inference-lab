#!/bin/bash
# Full benchmark of one served model: perf sweep, then quality suite.
# Usage: run_model.sh <base e.g. http://localhost:8010> <served_model_name> <tokenizer_path> <tag> [extra quality args...]
set -u
BASE=$1; MODEL=$2; TOK=$3; TAG=$4; shift 4
W=/mnt/pfs/4n3evq/lhx/llm_deploy; R=$W/bench/results/$TAG; mkdir -p $R/perf
source /root/miniforge/etc/profile.d/conda.sh && conda activate llm
export LD_LIBRARY_PATH=/root/cuda-compat-13/extract/usr/local/cuda-13.1/compat:$CONDA_PREFIX/lib
echo "[$(date '+%F %T')] === PERF $TAG ==="
bash $W/bench/perf.sh $BASE $MODEL $TOK $R/perf
echo "[$(date '+%F %T')] === QUALITY $TAG ==="
python3 $W/bench/quality.py --base-url $BASE/v1 --model $MODEL --out $R/quality.json "$@"
echo "[$(date '+%F %T')] === ALL_DONE $TAG ==="
