#!/usr/bin/env python3
"""verify_model.py <org/name> <local_dir> -> exit 0 if every remote file exists locally with the right size."""
import json, os, sys, urllib.request
model, local = sys.argv[1], sys.argv[2]
url = f"https://www.modelscope.cn/api/v1/models/{model}/repo/files?Recursive=true"
files = json.load(urllib.request.urlopen(url, timeout=60))["Data"]["Files"]
bad = []
for f in files:
    if f.get("Type") == "tree": continue
    p = os.path.join(local, f["Path"])
    if not os.path.exists(p) or os.path.getsize(p) != f["Size"]:
        bad.append((f["Path"], f["Size"], os.path.getsize(p) if os.path.exists(p) else None))
inc = [x for x in os.listdir(local) if x.endswith(".incomplete")] if os.path.isdir(local) else []
if bad or inc:
    print(f"INCOMPLETE {model}: {len(bad)} bad/missing files, {len(inc)} .incomplete"); [print("  ", b) for b in bad[:8]]; sys.exit(1)
print(f"COMPLETE {model}: {len(files)} files verified"); sys.exit(0)
