#!/bin/bash
# Phase 2: stop GLM, serve Qwen3.6-35B-A3B (GPU 0,1) and Qwen3.8-27B (GPU 2,3) via the cu129 image chroot,
# run perf sweeps one at a time, then the quality suites in parallel.
W=/mnt/pfs/4n3evq/lhx/llm_deploy; M=/mnt/pfs/4n3evq/lhx/models
log() { echo "[$(date '+%F %T')] $*"; }
log "stopping GLM"
pkill -f "vllm serve /mnt/pfs/4n3evq/lhx/models/GLM-5.3-Flash"; sleep 10; pkill -9 -f "VLLM::" 2>/dev/null; sleep 5
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr "\n" " "; echo

serve() { # name model port gpus tp maxlen reasoning_parser
  local name=$1 model=$2 port=$3 gpus=$4 tp=$5 maxlen=$6 rp=$7
  ROOTFS=/root/vllm_glm53_rootfs COMPAT=image nohup $W/glm53_chroot.sh env CUDA_VISIBLE_DEVICES=$gpus VLLM_USE_FLASHINFER_SAMPLER=0 \
    vllm serve $model --served-model-name $name --tensor-parallel-size $tp --max-model-len $maxlen \
    --gpu-memory-utilization 0.88 --max-num-seqs 64 --host 0.0.0.0 --port $port --trust-remote-code \
    --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser $rp --limit-mm-per-prompt '{"image":1,"video":0}' \
    > $W/logs/serve_$name.log 2>&1 &
  log "started $name pid $! on GPUs $gpus port $port"
}
serve qwen3.6-35b-a3b $M/Qwen3.6-35B-A3B 8000 0,1 2 131072 qwen3
serve qwen3.8-27b     $M/Qwen3.8-27B     8001 2,3 2 131072 qwen3
for p in 8000 8001; do
  for i in $(seq 1 240); do curl -sf localhost:$p/v1/models >/dev/null && break; sleep 5; done
  curl -sf localhost:$p/v1/models >/dev/null && log "port $p ready" || { log "port $p FAILED to start"; tail -20 $W/logs/serve_qwen3.*.log; }
done
grep -hE "Model loading took|Available KV cache memory|GPU KV cache size|Maximum concurrency" $W/logs/serve_qwen3.6-35b-a3b.log $W/logs/serve_qwen3.8-27b.log | cut -c1-160

source /root/miniforge/etc/profile.d/conda.sh && conda activate llm
export LD_LIBRARY_PATH=/root/cuda-compat-13/extract/usr/local/cuda-13.1/compat:$CONDA_PREFIX/lib
for spec in "qwen3.6-35b-a3b 8000 Qwen3.6-35B-A3B" "qwen3.8-27b 8001 Qwen3.8-27B"; do
  set -- $spec; R=$W/bench/results/$3; mkdir -p $R/perf
  log "=== PERF $3 ==="; bash $W/bench/perf.sh http://localhost:$2 $1 $M/$3 $R/perf
done
log "=== QUALITY both (parallel) ==="
python3 $W/bench/quality.py --base-url http://localhost:8000/v1 --model qwen3.6-35b-a3b --out $W/bench/results/Qwen3.6-35B-A3B/quality.json --vision > $W/bench/quality_Qwen3.6.log 2>&1 &
python3 $W/bench/quality.py --base-url http://localhost:8001/v1 --model qwen3.8-27b --out $W/bench/results/Qwen3.8-27B/quality.json > $W/bench/quality_Qwen3.8.log 2>&1 &
wait
log "=== PHASE2_DONE ==="
