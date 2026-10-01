#!/bin/bash
# Pipelined upload: as each layer finishes downloading locally, rsync it to the server.
S=/tmp/claude-1000/-home-ps-Downloads-BaiDuGPU-LLM-test/630b046e-de65-4573-870a-786387a27c57/scratchpad
SRC=$S/docker_layers
DST=root@<SERVER_IP>:/mnt/pfs/4n3evq/lhx/llm_deploy/docker_layers/
SSH="ssh -i $HOME/.ssh/id_baidu_aihc -p <SSH_PORT>"
LOG=$S/upload_layers.log
n=$(python3 -c "import json;print(len(json.load(open('$SRC/manifest.json'))['layers']))")
echo "[$(date +%T)] $n layers to upload" >> $LOG
for f in manifest.json config.json $(for i in $(seq 0 $((n-1))); do printf "layer_%02d\n" $i; done); do
  # wait for the file to be complete (no .part)
  while true; do
    p=$(ls $SRC/$f* 2>/dev/null | grep -v '\.part$' | head -1)
    [ -n "$p" ] && break
    if grep -q "giving up" $S/local_pull.log 2>/dev/null; then echo "[$(date +%T)] local pull failed" >> $LOG; exit 1; fi
    sleep 15
  done
  sz=$(stat -c %s "$p")
  t0=$(date +%s)
  for try in 1 2 3 4 5; do
    rsync -a --partial --inplace -e "$SSH" "$p" "$DST" && break
    echo "[$(date +%T)] rsync retry $try for $(basename $p)" >> $LOG; sleep 20
  done
  t1=$(date +%s)
  echo "[$(date +%T)] uploaded $(basename $p) $((sz/1000000)) MB in $((t1-t0))s" >> $LOG
done
echo "[$(date +%T)] UPLOAD_ALL_DONE" >> $LOG
