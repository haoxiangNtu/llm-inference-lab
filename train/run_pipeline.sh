#!/bin/bash
# Full nanochat lifecycle on 8x A800: tokenizer -> pretrain -> eval -> SFT -> eval -> RL -> eval.
# Usage: DEPTH=12 TAG=d12 SHARDS=24 BASE_ARGS="..." bash run_pipeline.sh
set -uo pipefail
W=/mnt/pfs/4n3evq/lhx/llm_deploy; T=$W/train
source /root/miniforge/etc/profile.d/conda.sh && conda activate nanochat
export LD_LIBRARY_PATH=/root/cuda-compat-13/extract/usr/local/cuda-13.1/compat:$CONDA_PREFIX/lib
export OMP_NUM_THREADS=1 NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-/mnt/pfs/4n3evq/lhx/nanochat_cache} HF_ENDPOINT=https://hf-mirror.com
DEPTH=${DEPTH:-12}; TAG=${TAG:-d$DEPTH}; SHARDS=${SHARDS:-24}; NGPU=${NGPU:-8}; L=$T/logs/$TAG; mkdir -p $L
cd $T/nanochat
stage() { echo "[$(date '+%F %T')] === $1 ==="; }
run() { local name=$1; shift; stage "$name START"; local t0=$(date +%s); "$@" > $L/$name.log 2>&1; local rc=$?; stage "$name END rc=$rc secs=$(( $(date +%s)-t0 ))"; return $rc; }
TR="torchrun --standalone --nproc_per_node=$NGPU"
if [ ! -f $NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl ]; then
  run data_first python -m nanochat.dataset -n 8 -w 8 || exit 1
  run tok_train python -m scripts.tok_train || exit 1
  run tok_eval python -m scripts.tok_eval
fi
run data python -m nanochat.dataset -n $SHARDS -w 8 || exit 1
run base_train $TR -m scripts.base_train -- --depth=$DEPTH --model-tag=$TAG --run=dummy ${BASE_ARGS:-} || exit 1
run base_eval $TR -m scripts.base_eval -- --model-tag=$TAG --device-batch-size=${EVAL_BS:-16}
run chat_sft $TR -m scripts.chat_sft -- --model-tag=$TAG --run=dummy ${SFT_ARGS:-} || exit 1
run chat_eval_sft $TR -m scripts.chat_eval -- -i sft -g $TAG
if [ "${DO_RL:-1}" = "1" ]; then
  run chat_rl $TR -m scripts.chat_rl -- --model-tag=$TAG --run=dummy ${RL_ARGS:-}
  run chat_eval_rl $TR -m scripts.chat_eval -- -i rl -g $TAG -a GSM8K
fi
for q in "Why is the sky blue?" "What is 12 times 7?" "Who are you?" "Write a haiku about GPUs."; do
  echo "### $q" >> $L/samples_sft.txt; python -m scripts.chat_cli -i sft -g $TAG -p "$q" 2>/dev/null | tail -n +1 >> $L/samples_sft.txt; echo >> $L/samples_sft.txt
done
stage "PIPELINE_DONE $TAG"
