#!/bin/bash
# Wait for the GLM-5.3-Flash download to be complete and verified, then stop the Qwen warm-up and launch GLM.
W=/mnt/pfs/4n3evq/lhx/llm_deploy; G=/mnt/pfs/4n3evq/lhx/models/GLM-5.3-Flash
source /root/miniforge/etc/profile.d/conda.sh && conda activate llm
log() { echo "[$(date '+%F %T')] $*"; }
# 1. wait for aria2 to finish
while pgrep -f "aria2c -i /tmp/glm_missing" >/dev/null; do sleep 30; done
log "aria2 finished; verifying"
# 2. verify (retry API timeouts); re-run aria2 on anything missing, up to 5 rounds
for round in 1 2 3 4 5; do
  if python3 $W/verify_model.py ZhipuAI/GLM-5.3-Flash $G; then log "GLM COMPLETE"; break; fi
  log "round $round: incomplete, re-fetching missing shards"
  python3 - <<'PY' > /tmp/glm_missing.txt
import json,os,urllib.request,time
model="ZhipuAI/GLM-5.3-Flash"; local="/mnt/pfs/4n3evq/lhx/models/GLM-5.3-Flash"
for t in range(5):
    try:
        files=json.load(urllib.request.urlopen(f"https://www.modelscope.cn/api/v1/models/{model}/repo/files?Recursive=true",timeout=60))["Data"]["Files"]; break
    except Exception: time.sleep(10)
for f in files:
    if f.get("Type")=="tree": continue
    p=os.path.join(local,f["Path"])
    if os.path.exists(p) and os.path.getsize(p)==f["Size"] and not os.path.exists(p+".aria2"): continue
    print(f"https://www.modelscope.cn/models/{model}/resolve/master/{f['Path']}")
PY
  [ -s /tmp/glm_missing.txt ] && aria2c -i /tmp/glm_missing.txt -d $G -c -x 6 -s 6 -k 8M -j 4 --file-allocation=none --auto-file-renaming=false --allow-overwrite=true --max-tries=30 --retry-wait=5 --timeout=45 --console-log-level=warn --summary-interval=120 >> $W/logs/aria2_glm3.log 2>&1
done
python3 $W/verify_model.py ZhipuAI/GLM-5.3-Flash $G || { log "GLM STILL INCOMPLETE, giving up"; exit 1; }
# 3. stop the Qwen warm-up (needs all 8 GPUs)
log "stopping Qwen warm-up server"
pkill -f "vllm serve /mnt/pfs/4n3evq/lhx/models/Qwen3.6"; sleep 8; pkill -9 -f "VLLM::" 2>/dev/null; sleep 5
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr "\n" " "; echo
# 4. launch GLM
log "launching GLM-5.3-Flash (TP4 x PP2)"
bash $W/serve_glm53.sh
log "LAUNCHED"
