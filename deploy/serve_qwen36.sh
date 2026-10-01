#!/bin/bash
# Qwen3.6-35B-A3B warm-up on the CUDA 12.9 vllm-openai image (stock vLLM 0.28.1) via chroot.
W=/mnt/pfs/4n3evq/lhx/llm_deploy
MODEL=/mnt/pfs/4n3evq/lhx/models/Qwen3.6-35B-A3B
TP=${TP:-2}; GPUS=${GPUS:-0,1}; PORT=${PORT:-8000}; MAXLEN=${MAXLEN:-32768}
LOG=$W/logs/serve_qwen36.log
ROOTFS=/root/vllm_glm53_rootfs COMPAT=none nohup $W/glm53_chroot.sh env CUDA_VISIBLE_DEVICES=$GPUS \
  vllm serve $MODEL --served-model-name qwen3.6-35b-a3b \
  --tensor-parallel-size $TP --max-model-len $MAXLEN --gpu-memory-utilization 0.85 \
  --max-num-seqs 32 --host 0.0.0.0 --port $PORT --trust-remote-code \
  --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 \
  > $LOG 2>&1 &
echo "PID=$! LOG=$LOG"
