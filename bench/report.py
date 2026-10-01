#!/usr/bin/env python3
"""Aggregate perf (vllm bench serve JSON) + quality (quality.py JSON) results into a markdown comparison.
Usage: report.py <results_root>   (expects <root>/<model>/perf/*.json and <root>/<model>/quality.json)
"""
import glob, json, os, sys
root = sys.argv[1]
models = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
def load(p):
    try: return json.load(open(p))
    except Exception: return None
perf_rows = []; qual = {}
for m in models:
    ORDER = ["single_128in_256out", "prefill_2k", "prefill_8k", "prefill_32k", "conc_4_1kin_256out", "conc_8_1kin_256out", "conc_16_1kin_256out", "conc_32_1kin_256out", "conc_16_4kin_512out"]
    found = {os.path.basename(p)[:-5]: p for p in glob.glob(os.path.join(root, m, "perf", "*.json"))}
    for name in ORDER + sorted(set(found) - set(ORDER)):
        if name not in found: continue
        d = load(found[name])
        if not d: continue
        n = max(1, d.get("num_prompts") or d.get("completed") or 1)
        d["_in"] = round((d.get("total_input_tokens") or 0) / n); d["_out"] = round((d.get("total_output_tokens") or 0) / n)
        d["_prefill_tps"] = (d["_in"] / (d["median_ttft_ms"] / 1000)) if d.get("median_ttft_ms") else None
        perf_rows.append((m, name, d))
    q = load(os.path.join(root, m, "quality.json"))
    if q: qual[m] = q
def g(d, k, nd=1):
    v = d.get(k)
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, (int, float)) else str(v))
out = []
out.append("## 性能（vllm bench serve，random 数据集，ignore_eos）\n")
out.append("| 模型 | 场景 | 并发 | 输入/输出 token | 输出吞吐 tok/s | 总吞吐 tok/s | TTFT 中位 ms | TTFT P99 ms | Prefill 速度 tok/s | TPOT 中位 ms | TPOT P99 ms | 单流速度 tok/s | 端到端中位 s |")
out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for m, name, d in perf_rows:
    sp = (1000 / d["median_tpot_ms"]) if d.get("median_tpot_ms") else None
    out.append(f"| {m} | {name} | {d.get('max_concurrency','-')} | {d['_in']}/{d['_out']} | {g(d,'output_throughput',0)} | {g(d,'total_token_throughput',0)} | {g(d,'median_ttft_ms',0)} | {g(d,'p99_ttft_ms',0)} | {g(d,'_prefill_tps',0)} | {g(d,'median_tpot_ms',1)} | {g(d,'p99_tpot_ms',1)} | {sp and f'{sp:.0f}' or '-'} | {(d.get('median_e2el_ms') or 0)/1000:.1f} |")
out.append("\n## 质量\n")
tests = ["gsm8k", "humaneval", "ceval", "json", "tools", "vision"]
out.append("| 模型 | " + " | ".join(tests) + " |"); out.append("|---|" + "---|" * len(tests))
for m, q in qual.items():
    cells = []
    for t in tests:
        s = q.get("tests", {}).get(t, {}).get("summary")
        cells.append("-" if not s or s.get("accuracy") is None else f"{100*s['accuracy']:.1f}% ({s['correct']}/{s['n']})")
    out.append(f"| {m} | " + " | ".join(cells) + " |")
out.append("\n## 长上下文大海捞针（准确率 / 平均延迟）\n")
lens = sorted({int(L) for q in qual.values() for L in q.get("tests", {}).get("needle", {}).get("summary", {}).get("per_length", {})})
if lens:
    out.append("| 模型 | " + " | ".join(f"{L//1024}K" for L in lens) + " |"); out.append("|---|" + "---|" * len(lens))
    for m, q in qual.items():
        pl = q.get("tests", {}).get("needle", {}).get("summary", {}).get("per_length", {})
        cells = []
        for L in lens:
            s = pl.get(str(L)) or pl.get(L)
            cells.append("-" if not s else f"{100*s['accuracy']:.0f}% / {s['avg_latency_s']:.1f}s")
        out.append(f"| {m} | " + " | ".join(cells) + " |")
out.append("\n## 平均输出长度（token）和平均单题耗时（秒），反映思考模式的开销\n")
out.append("| 模型 | " + " | ".join(tests[:3]) + " |"); out.append("|---|" + "---|" * 3)
for m, q in qual.items():
    cells = []
    for t in tests[:3]:
        s = q.get("tests", {}).get(t, {}).get("summary")
        cells.append("-" if not s or not s.get("avg_completion_tokens") else f"{s['avg_completion_tokens']:.0f} tok / {s['avg_latency_s']:.1f}s")
    out.append(f"| {m} | " + " | ".join(cells) + " |")
print("\n".join(out))
