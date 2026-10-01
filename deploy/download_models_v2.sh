#!/bin/bash
source /root/miniforge/etc/profile.d/conda.sh && conda activate llm
M=/mnt/pfs/4n3evq/lhx/models; W=/mnt/pfs/4n3evq/lhx/llm_deploy
export MODELSCOPE_CACHE=$M/.msc_cache
dl() { # org/name
  local name=${1#*/}
  echo "[$(date '+%F %T')] START $1"
  for try in $(seq 1 12); do
    modelscope download --model "$1" --local_dir "$M/$name" --max-workers 16
    if python3 $W/verify_model.py "$1" "$M/$name"; then echo "[$(date '+%F %T')] DONE $1"; return 0; fi
    echo "[$(date '+%F %T')] pass $try incomplete for $1, retrying"; sleep 20
  done
  echo "[$(date '+%F %T')] FAILED $1"
}
case "$1" in
  A) dl Qwen/Qwen3.6-35B-A3B; dl Qwen/Qwen3.8-27B; dl Wan-AI/Wan2.2-T2V-A14B ;;
  B) dl ZhipuAI/GLM-5.3-Flash ;;
  V) dl "$2" ;;
esac
echo "[$(date '+%F %T')] QUEUE $1 FINISHED"
