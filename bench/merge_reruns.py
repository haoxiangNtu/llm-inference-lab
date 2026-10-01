#!/usr/bin/env python3
"""Fold rerun_*.json results into quality.json for each model dir under <root>.
rerun_jt.json    -> replaces json/tools (fair 2048-token budget)
rerun_tools.json -> replaces tools (correct tool-call parser)
rerun_needle.json-> replaces the errored 131072 needle entry with the 110K measurement
A 'notes' list records what was replaced so the report can footnote it."""
import json, os, sys
root = sys.argv[1]
for m in sorted(os.listdir(root)):
    qp = os.path.join(root, m, "quality.json")
    if not os.path.exists(qp): continue
    q = json.load(open(qp)); q.setdefault("notes", [])
    def take(fn, tests, note):
        p = os.path.join(root, m, fn)
        if not os.path.exists(p): return
        r = json.load(open(p))
        for t in tests:
            if t in r.get("tests", {}):
                q["tests"][t] = r["tests"][t];
        if note not in q["notes"]: q["notes"].append(note)
    take("rerun_jt.json", ["json", "tools"], "json/tools 以 2048 token 预算重跑（首轮 512 预算被思考占满）")
    take("rerun_tools.json", ["tools"], "tools 使用 qwen3_xml 解析器重跑（首轮 hermes 解析器不匹配该模型的 XML 格式）")
    p = os.path.join(root, m, "rerun_needle.json")
    if os.path.exists(p):
        r = json.load(open(p))["tests"].get("needle")
        n = q["tests"].get("needle")
        if r and n:
            pl = n["summary"].setdefault("per_length", {})
            bad = [k for k, v in pl.items() if int(k) >= 131072 and v.get("prompt_tokens") is None]
            for k in bad: pl.pop(k)
            pl.update(r["summary"].get("per_length", {}))
            recs = [x for x in n.get("records", []) if not x.get("error")] + r.get("records", [])
            n["records"] = recs; ok = sum(1 for x in recs if x.get("correct")); n["summary"].update({"n": len(recs), "correct": ok, "accuracy": ok / len(recs) if recs else None, "errors": 0})
            note = "128K 档超过服务的 max-model-len 131072（实际 149K token）被拒，改测 110K 档"
            if note not in q["notes"]: q["notes"].append(note)
    json.dump(q, open(qp, "w"), ensure_ascii=False, indent=1)
    print(m, "merged; notes:", q["notes"])
