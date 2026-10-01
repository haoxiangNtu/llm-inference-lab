#!/bin/bash
# Serving-performance sweep with `vllm bench serve` (random dataset, fixed lengths, ignore_eos).
# Usage: perf.sh <base_url e.g. http://localhost:8010> <served_model_name> <tokenizer_path> <result_dir> [quick]
set -u
BASE=$1; MODEL=$2; TOK=$3; OUT=$4; QUICK=${5:-}
source /root/miniforge/etc/profile.d/conda.sh && conda activate llm
export LD_LIBRARY_PATH=/root/cuda-compat-13/extract/usr/local/cuda-13.1/compat:$CONDA_PREFIX/lib
mkdir -p $OUT
run() { # name input_len output_len num_prompts concurrency
  local name=$1 il=$2 ol=$3 np=$4 c=$5
  echo "[$(date +%T)] $name: in=$il out=$ol prompts=$np conc=$c"
  timeout 3600 vllm bench serve --backend openai-chat --base-url $BASE --endpoint /v1/chat/completions \
    --model $MODEL --tokenizer $TOK --trust-remote-code \
    --dataset-name random --random-input-len $il --random-output-len $ol --random-range-ratio 0.0 \
    --num-prompts $np --max-concurrency $c --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --save-result --result-dir $OUT --result-filename $name.json > $OUT/$name.log 2>&1
  grep -E "Successful requests|Request throughput|Output token throughput|Total Token throughput|Mean TTFT|Median TTFT|P99 TTFT|Mean TPOT|Median TPOT|P99 TPOT|Mean ITL|Median E2EL" $OUT/$name.log | sed 's/^/    /'
}
# 1. decode speed, single stream (short prompt)
run single_128in_256out 128 256 8 1
# 2. prefill scaling: TTFT vs input length, single stream
run prefill_2k 2048 32 6 1
run prefill_8k 8192 32 4 1
run prefill_32k 32768 32 3 1
# 3. throughput vs concurrency (1K in / 256 out)
for c in 4 8 16 32; do run conc_${c}_1kin_256out 1024 256 $((c*3)) $c; done
[ -n "$QUICK" ] && { echo PERF_DONE; exit 0; }
# 4. heavier: 4K in / 512 out at 16 concurrency (mixed prefill+decode)
run conc_16_4kin_512out 4096 512 48 16
echo PERF_DONE
