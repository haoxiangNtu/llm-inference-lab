#!/bin/bash
# Re-run the shared benchmark suite on all six nanochat chat models (SFT and RL of d12/d24/d32) after fixing the
# calculator-in-worker-thread bug in scripts/openai_server.py, then regenerate SFT-vs-RL math samples on 12 problems.
# Uses GPUs 4-7 only. Servers are left running afterwards: SFT on 8020-8022, RL on 8030-8032.
W=/mnt/pfs/4n3evq/lhx/llm_deploy; T=$W/train; R=$W/bench/results
source /root/miniforge/etc/profile.d/conda.sh && conda activate nanochat
export LD_LIBRARY_PATH=/root/cuda-compat-13/extract/usr/local/cuda-13.1/compat:$CONDA_PREFIX/lib
export OMP_NUM_THREADS=1 NANOCHAT_BASE_DIR=/mnt/pfs/4n3evq/lhx/nanochat_cache NANOCHAT_FA3_PATH=$T/fa3_kernel
cd $T/nanochat
pkill -f "scripts.openai_server"; sleep 5
# tag src gpu port
SPECS=("d32 sft 4 8022" "d32 rl 5 8032" "d24 sft 6 8020" "d24 rl 6 8031" "d12 sft 7 8021" "d12 rl 7 8030")
for spec in "${SPECS[@]}"; do set -- $spec
  CUDA_VISIBLE_DEVICES=$3 nohup python -m scripts.openai_server --source $2 --model-tag $1 --port $4 --served-name nanochat-$1-$2 \
    > $T/logs/server_$1_$2.log 2>&1 < /dev/null &
done
for spec in "${SPECS[@]}"; do set -- $spec; for k in $(seq 1 120); do curl -sf localhost:$4/v1/models >/dev/null && break; sleep 3; done; done
echo "[$(date '+%F %T')] servers up"
echo "--- calculator smoke test (d32 rl) ---"
curl -s localhost:8032/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"nanochat-d32-rl","messages":[{"role":"user","content":"A farmer has 17 cows and buys 3 times as many sheep. How many animals does he have now?"}],"max_tokens":200,"temperature":0}' \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
pids=()
for spec in "${SPECS[@]}"; do set -- $spec
  d=$R/nanochat-$1-$2; [ -f $d/quality.json ] && [ ! -f $d/quality_calc_bug.json ] && cp $d/quality.json $d/quality_calc_bug.json
  SKIP_PERF=1 bash $T/bench_nanochat.sh nanochat-$1-$2 $4 nanochat-$1-$2 > $W/bench/logs_rebench_nanochat-$1-$2.txt 2>&1 &
  pids+=($!)
done
wait "${pids[@]}"
echo "[$(date '+%F %T')] benches done"
for tag in d12 d24 d32; do
  CUDA_VISIBLE_DEVICES=7 python -m scripts.math_samples -g $tag -n 12 --out $T/logs/$tag/samples_math.json > $T/logs/$tag/samples_math.log 2>&1
done
echo "[$(date '+%F %T')] REBENCH_DONE"
