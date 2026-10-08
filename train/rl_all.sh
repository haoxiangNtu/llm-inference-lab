#!/bin/bash
# Reinforcement learning for the d24 and d32 chat models (nanochat's simplified GRPO / REINFORCE on GSM8K),
# run one after the other on GPUs 4-7, then:
#   1. full chat eval (ARC-E, ARC-C, MMLU, GSM8K, HumanEval) of each RL model, to see what RL cost elsewhere
#   2. greedy SFT-vs-RL answers on a few held-out GSM8K problems (scripts/math_samples.py)
#   3. the same full chat eval for the existing d12 RL model (its first run only scored GSM8K)
#   4. the shared benchmark suite (bench/quality.py) on all RL models, through scripts/openai_server.py
# Usage on the server:  nohup bash rl_all.sh > logs/rl_all.log 2>&1 < /dev/null &
set -uo pipefail
W=/mnt/pfs/4n3evq/lhx/llm_deploy; T=$W/train
source /root/miniforge/etc/profile.d/conda.sh && conda activate nanochat
export LD_LIBRARY_PATH=/root/cuda-compat-13/extract/usr/local/cuda-13.1/compat:$CONDA_PREFIX/lib
export OMP_NUM_THREADS=1 NCCL_DEBUG=WARN NANOCHAT_BASE_DIR=/mnt/pfs/4n3evq/lhx/nanochat_cache NANOCHAT_FA3_PATH=$T/fa3_kernel
GPUS=${GPUS:-4,5,6,7}; NGPU=${NGPU:-4}; FIRST_GPU=${GPUS%%,*}
# device-batch-size 16 = all 16 rollouts of a problem in one generation call (the default 8 needs two calls).
# The estimator is unchanged: same 16 problems x 16 samples per step, same advantages, same learning rate.
RL_DBS=${RL_DBS:-16}
cd $T/nanochat
TR="torchrun --standalone --nproc_per_node=$NGPU"
stamp() { echo "[$(date '+%F %T')] === $1 ===" | tee -a "$2"; }
run() { # tag stage_name command...
  local tag=$1 name=$2; shift 2; local L=$T/logs/$tag
  stamp "$name START" $L/pipeline.log; local t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=$GPUS "$@" > $L/$name.log 2>&1; local rc=$?
  stamp "$name END rc=$rc secs=$(( $(date +%s)-t0 ))" $L/pipeline.log; return $rc
}

for tag in ${TAGS:-d24 d32}; do
  run $tag chat_rl $TR -m scripts.chat_rl -- --model-tag=$tag --run=dummy --device-batch-size=$RL_DBS --save-every=240 \
    || { echo "[$(date '+%F %T')] RL FAILED for $tag"; continue; }
  run $tag chat_eval_rl $TR -m scripts.chat_eval -- -i rl -g $tag
  CUDA_VISIBLE_DEVICES=$FIRST_GPU python -m scripts.math_samples -g $tag --out $T/logs/$tag/samples_math.json > $T/logs/$tag/samples_math.log 2>&1
done

if [ "${D12_EVAL:-1}" = "1" ] && [ -d $NANOCHAT_BASE_DIR/chatrl_checkpoints/d12 ]; then
  [ -f $T/logs/d12/chat_eval_rl_gsm8k_only.log ] || cp $T/logs/d12/chat_eval_rl.log $T/logs/d12/chat_eval_rl_gsm8k_only.log
  run d12 chat_eval_rl_all $TR -m scripts.chat_eval -- -i rl -g d12 && cp $T/logs/d12/chat_eval_rl_all.log $T/logs/d12/chat_eval_rl.log
  CUDA_VISIBLE_DEVICES=$FIRST_GPU python -m scripts.math_samples -g d12 --out $T/logs/d12/samples_math.json > $T/logs/d12/samples_math.log 2>&1
fi

IFS=, read -ra G <<< "$GPUS"; i=0; pids=()
for tag in d12 d24 d32; do
  [ -d $NANOCHAT_BASE_DIR/chatrl_checkpoints/$tag ] || continue
  port=$((8030+i)); gpu=${G[$i]}
  CUDA_VISIBLE_DEVICES=$gpu nohup python -m scripts.openai_server --source rl --model-tag $tag --port $port --served-name nanochat-$tag-rl \
    > $T/logs/server_${tag}_rl.log 2>&1 < /dev/null &
  for k in $(seq 1 100); do curl -sf localhost:$port/v1/models >/dev/null && break; sleep 3; done
  SKIP_PERF=1 bash $T/bench_nanochat.sh nanochat-$tag-rl $port nanochat-$tag-rl > $W/bench/logs_nanochat-$tag-rl.txt 2>&1 &
  pids+=($!); i=$((i+1))
done
[ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"
pkill -f "openai_server --source rl"
echo "[$(date '+%F %T')] RL_ALL_DONE"
