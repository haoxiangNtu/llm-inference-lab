#!/bin/bash
# Same yardstick as the deployed models (bench/quality.py, vllm bench serve), with reduced sample sizes
# because the nanochat engine serves one request at a time.
# Usage: bench_nanochat.sh <result_tag> <port> <served_name>     env: SKIP_QUALITY=1, SKIP_PERF=1
W=/mnt/pfs/4n3evq/lhx/llm_deploy; TAG=$1; PORT=$2; NAME=$3; R=$W/bench/results/$TAG; mkdir -p $R/perf
source /root/miniforge/etc/profile.d/conda.sh && conda activate llm
export LD_LIBRARY_PATH=/root/cuda-compat-13/extract/usr/local/cuda-13.1/compat:$CONDA_PREFIX/lib
if [ "${SKIP_QUALITY:-0}" != "1" ]; then
  echo "[$(date '+%F %T')] quality $TAG"
  python3 $W/bench/quality.py --base-url http://localhost:$PORT/v1 --model $NAME --out $R/quality.json \
    --only gsm8k,humaneval,ceval,json,tools --gsm8k 100 --humaneval 82 --ceval 150 --max-tokens 512 --concurrency 4
fi
if [ "${SKIP_PERF:-0}" != "1" ]; then
  echo "[$(date '+%F %T')] perf $TAG"
  for spec in "single_128in_256out 128 256 6 1" "conc_4_1kin_256out 1024 256 8 4"; do
    set -- $spec
    vllm bench serve --backend openai-chat --base-url http://localhost:$PORT --endpoint /v1/chat/completions --model $NAME \
      --tokenizer /mnt/pfs/4n3evq/lhx/models/Qwen3.8-27B --dataset-name random --random-input-len $2 --random-output-len $3 \
      --random-range-ratio 0.0 --num-prompts $4 --max-concurrency $5 --ignore-eos --percentile-metrics ttft,tpot,itl,e2el \
      --metric-percentiles 50,90,99 --save-result --result-dir $R/perf --result-filename $1.json > $R/perf/$1.log 2>&1
    grep -E "Output token throughput|Median TTFT|Median TPOT" $R/perf/$1.log | sed 's/^/    /'
  done
fi
echo "[$(date '+%F %T')] NANOCHAT_BENCH_DONE $TAG"
