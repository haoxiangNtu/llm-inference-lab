#!/bin/bash
# GLM-5.3-Flash (official FP8) on 8x A800 via the patched vllm-backport chroot.
# Mirrors Mrzhiyao/glm53-a800-vllm scripts/run-server.sh (TP4 x PP2, 1M ctx, cudagraph).
W=/mnt/pfs/4n3evq/lhx/llm_deploy
MODEL=${MODEL:-/mnt/pfs/4n3evq/lhx/models/GLM-5.3-Flash}
TP=${TP:-4}; PP=${PP:-2}; GPUS=${GPUS:-0,1,2,3,4,5,6,7}; PORT=${PORT:-8010}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-1048576}
GMU=${GMU:-0.80}
MAX_SEQS=${MAX_SEQS:-16}; MAX_BATCHED=${MAX_BATCHED:-16384}
PP_LAYER_PARTITION=${PP_LAYER_PARTITION:-24,21}
LOG=${LOG:-$W/logs/serve_glm53.log}
test -s $MODEL/model.safetensors.index.json || { echo "model index missing"; exit 1; }
ROOTFS=/root/vllm_backport_rootfs COMPAT=image nohup $W/glm53_chroot.sh env \
  CUDA_VISIBLE_DEVICES=$GPUS VLLM_USE_DEEP_GEMM=0 VLLM_PP_LAYER_PARTITION=$PP_LAYER_PARTITION \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_ALGO=Ring NCCL_PROTO=Simple \
  vllm serve $MODEL --served-model-name GLM-5.3-Flash glm-5.3-flash \
  --tensor-parallel-size $TP --pipeline-parallel-size $PP --trust-remote-code \
  --max-model-len $MAX_MODEL_LEN --gpu-memory-utilization $GMU \
  --max-num-seqs $MAX_SEQS --max-num-batched-tokens $MAX_BATCHED \
  --disable-custom-all-reduce --limit-mm-per-prompt '{"image":1,"video":0}' \
  --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 \
  --host 0.0.0.0 --port $PORT "$@" > $LOG 2>&1 &
echo "PID=$! LOG=$LOG"
