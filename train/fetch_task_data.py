#!/usr/bin/env python3
"""Pre-populate nanochat's task_data cache (same layout as tasks/common.py::load_hub_dataset) from huggingface.co.
Run on a machine that can reach HF, then rsync <out>/ to $NANOCHAT_BASE_DIR/task_data/ on the training box."""
import json, os, sys, urllib.request
out = sys.argv[1]
SETS = [("HuggingFaceTB/smol-smoltalk", "default", "train"), ("HuggingFaceTB/smol-smoltalk", "default", "test"),
        ("cais/mmlu", "all", "auxiliary_train"), ("cais/mmlu", "all", "test"), ("cais/mmlu", "all", "validation"), ("cais/mmlu", "all", "dev"),
        ("openai/gsm8k", "main", "train"), ("openai/gsm8k", "main", "test"),
        ("allenai/ai2_arc", "ARC-Easy", "train"), ("allenai/ai2_arc", "ARC-Easy", "validation"), ("allenai/ai2_arc", "ARC-Easy", "test"),
        ("allenai/ai2_arc", "ARC-Challenge", "train"), ("allenai/ai2_arc", "ARC-Challenge", "validation"), ("allenai/ai2_arc", "ARC-Challenge", "test"),
        ("openai/openai_humaneval", "openai_humaneval", "test")]
for repo, subset, split in SETS:
    d = os.path.join(out, repo.replace("/", "--"), subset, split); man = os.path.join(d, "manifest.json")
    if os.path.exists(man): print("have", repo, subset, split); continue
    os.makedirs(d, exist_ok=True)
    try:
        urls = json.loads(urllib.request.urlopen(f"https://huggingface.co/api/datasets/{repo}/parquet/{subset}/{split}", timeout=60).read())
    except Exception as e:
        print("LIST FAIL", repo, subset, split, e); continue
    names = []
    for i, u in enumerate(urls):
        fn = f"{i:05d}.parquet"
        for attempt in range(5):
            try:
                data = urllib.request.urlopen(u, timeout=600).read(); break
            except Exception as e:
                print("  retry", attempt, e)
        open(os.path.join(d, fn), "wb").write(data); names.append(fn)
        print(f"  {repo} {subset}/{split} {fn} {len(data)/1e6:.1f} MB", flush=True)
    json.dump(names, open(man, "w"))
print("FETCH_DONE")
