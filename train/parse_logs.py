#!/usr/bin/env python3
"""Parse a nanochat pipeline log directory (pipeline.log, tok_*.log, base_*.log, chat_*.log, samples_sft.txt)
into one compact JSON for the site.   Usage: parse_logs.py <logs_dir> <out.json> [tag]"""
import json, os, re, sys
D, OUT = sys.argv[1], sys.argv[2]
TAG = sys.argv[3] if len(sys.argv) > 3 else os.path.basename(os.path.normpath(D))
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
def read(name):
    p = os.path.join(D, name)
    if not os.path.exists(p): return ""
    return ANSI.sub("", open(p, encoding="utf-8", errors="ignore").read().replace("\r", "\n"))
def num(s): return float(s.replace(",", ""))
def down(xs, n=400):
    if len(xs) <= n: return xs
    k = len(xs) / n
    return [xs[int(i * k)] for i in range(n)] + [xs[-1]]
R = {"tag": TAG}

# ---- pipeline timings
stages = []
for m in re.finditer(r"\[(\S+ \S+)\] === (\S+) END rc=(\d+) secs=(\d+) ===", read("pipeline.log")):
    stages.append({"name": m.group(2), "end": m.group(1), "rc": int(m.group(3)), "secs": int(m.group(4))})
R["stages"] = stages

# ---- tokenizer
t = read("tok_train.log"); m = re.search(r"Training time: ([\d.]+)s", t)
R["tokenizer"] = {"train_secs": float(m.group(1)) if m else None,
                  "vocab_size": (re.search(r"vocab_size: ([\d,]+)", t) or [None, "0"])[1],
                  "max_chars": (re.search(r"max_chars: ([\d,]+)", t) or [None, "0"])[1], "compare": {}}
te = read("tok_eval.log")
for ref in ("GPT-2", "GPT-4"):
    blk = te.split(f"Comparison with {ref}:")
    if len(blk) < 2: continue
    rows = []
    for line in blk[1].split("Comparison with")[0].splitlines():
        m = re.match(r"\s*(\S+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)\s+([+-][\d.]+)%", line)
        if m: rows.append({"text": m.group(1), "bytes": int(m.group(2)), "ref_tokens": int(m.group(3)), "ref_ratio": float(m.group(4)),
                           "our_tokens": int(m.group(5)), "our_ratio": float(m.group(6)), "diff_pct": float(m.group(7))})
    R["tokenizer"]["compare"][ref] = rows

# ---- base_train
b = read("base_train.log"); base = {}
m = re.search(r"Model config:\s*(\{.*?\})", b, re.S)
if m:
    try: base["config"] = json.loads(m.group(1))
    except Exception: pass
base["params"] = {k: int(v.replace(",", "")) for k, v in re.findall(r"^(\w+)\s*:\s*([\d,]+)\s*$", b.split("Parameter counts:")[1].split("Estimated")[0], re.M)} if "Parameter counts:" in b else {}
for key, pat in [("flops_per_token", r"Estimated FLOPs per token: ([\d.e+]+)"), ("total_batch_size", r"optimal batch size: ([\d,]+)"),
                 ("iterations", r"Calculated number of iterations.*: ([\d,]+)"), ("total_tokens", r"Total number of training tokens: ([\d,]+)"),
                 ("total_flops", r"Total training FLOPs estimate: ([\d.e+]+)"), ("grad_accum", r"gradient accumulation steps: (\d+)"),
                 ("peak_mem_mib", r"Peak memory usage: ([\d.]+)MiB"), ("train_minutes", r"Total training time: ([\d.]+)m"), ("min_val_bpb", r"Minimum validation bpb: ([\d.]+)")]:
    m = re.search(pat, b); base[key] = num(m.group(1)) if m else None
m = re.search(r"Tokens / micro-batch / rank: (\d+) x (\d+)", b); base["device_batch"] = int(m.group(1)) if m else None
base["fa3"] = "Using Flash Attention 3" in b
steps = [{"step": int(a), "loss": float(l), "dt_ms": float(dt), "tok_s": num(ts), "mfu": float(mf), "min": float(tt)} for a, l, dt, ts, mf, tt in
         re.findall(r"^step (\d+)/\d+ \([\d.]+%\) \| loss: ([\d.]+) \| lrm: [\d.]+ \| dt: ([\d.]+)ms \| tok/sec: ([\d,]+) \| bf16_mfu: ([\d.]+) .*?total time: ([\d.]+)m", b, re.M)]
if steps:
    body = steps[20:] or steps
    base["avg_tok_s"] = sum(s["tok_s"] for s in body) / len(body); base["avg_mfu"] = sum(s["mfu"] for s in body) / len(body); base["avg_dt_ms"] = sum(s["dt_ms"] for s in body) / len(body)
    base["last_step"] = steps[-1]["step"]
base["loss_curve"] = [[s["step"], round(s["loss"], 4)] for s in down(steps)]
base["val_bpb"] = [[int(a), float(v)] for a, v in re.findall(r"Step (\d+) \| Validation bpb: ([\d.]+)", b)]
base["core_in_train"] = [[int(a), float(v)] for a, v in re.findall(r"Step (\d+) \| CORE metric: ([\d.]+)", b)]
R["base_train"] = base

# ---- base_eval
e = read("base_eval.log"); be = {}
for key, pat in [("train_bpb", r"train bpb: ([\d.]+)"), ("val_bpb", r"val bpb: ([\d.]+)"), ("core", r"CORE metric: ([\d.]+)")]:
    m = re.findall(pat, e); be[key] = float(m[-1]) if m else None
be["core_tasks"] = [{"task": a, "type": ty, "acc": float(acc), "centered": float(c)} for a, ty, acc, c in
                    re.findall(r"Evaluating: (\S+) \(\d+-shot, type: (\w+)\)\.\.\. accuracy: ([\d.]+) \| centered: ([-\d.]+)", e)]
be["samples"] = [s.strip()[:420] for s in re.findall(r"<\|bos\|>([^\n]+(?:\n(?!-{20}|={20}|<\|bos\|>)[^\n]*){0,3})", e)][:6]
R["base_eval"] = be

# ---- chat_sft
s = read("chat_sft.log"); sft = {}
m = re.search(r"Training mixture: ([\d,]+) rows", s); sft["mixture_rows"] = int(num(m.group(1))) if m else None
ss = [{"step": int(a), "loss": float(l), "tok_s": num(ts), "mfu": float(mf)} for a, l, ts, mf in
      re.findall(r"^step (\d+) \([\d.]+%\) \| loss: ([\d.]+) \| lrm: [\d.]+ \| dt: [\d.]+ms \| tok/sec: ([\d,]+) \| mfu: ([\d.]+)", s, re.M)]
sft["loss_curve"] = [[x["step"], round(x["loss"], 4)] for x in down(ss)]
sft["steps"] = ss[-1]["step"] if ss else None
sft["val_bpb"] = [[int(a), float(v)] for a, v in re.findall(r"Step (\d+) \| Validation bpb: ([\d.]+)", s)]
sft["chatcore"] = [[int(a), float(v), float(c)] for a, v, c in re.findall(r"Step (\d+) \| ChatCORE: ([\d.]+) \| ChatCORE_cat: ([\d.]+)", s)]
m = re.search(r"Total training time: ([\d.]+)m", s); sft["train_minutes"] = float(m.group(1)) if m else None
R["chat_sft"] = sft

# ---- chat evals
def chat_eval(name):
    t = read(name); d = {k: float(v) for k, v in re.findall(r"^([\w-]+) accuracy: ([\d.]+)%", t, re.M)}
    m = re.search(r"ChatCORE metric: ([\d.]+)", t)
    if m: d["ChatCORE"] = float(m.group(1))
    return d
R["chat_eval_sft"] = chat_eval("chat_eval_sft.log"); R["chat_eval_rl"] = chat_eval("chat_eval_rl.log")

# ---- chat_rl
r = read("chat_rl.log"); rl = {}
rew = [[int(a), float(v), float(L)] for a, v, L in re.findall(r"^Step (\d+)/\d+ \| Average reward: ([\d.]+) \| Average sequence length: ([\d.]+)", r, re.M)]
rl["steps"] = len(rew); rl["reward_curve"] = down(rew, 240)
rl["pass_at_k"] = [{"step": int(a), **{f"pass@{k}": float(v) for k, v in re.findall(r"Pass@(\d+): ([\d.]+)", rest)}} for a, rest in re.findall(r"^Step (\d+) \| (Pass@1: .*)$", r, re.M)]
R["chat_rl"] = rl

# ---- samples
sm = read("samples_sft.txt"); samples = []
for q, body in re.findall(r"### (.*?)\n(.*?)(?=\n### |\Z)", sm, re.S):
    a = re.search(r"Assistant: (.*)", body, re.S)
    samples.append({"q": q.strip(), "a": (a.group(1).strip() if a else "").strip()[:900]})
R["samples"] = samples
json.dump(R, open(OUT, "w"), ensure_ascii=False, indent=1)
print(f"{TAG}: stages={len(stages)} base_steps={len(steps)} val_pts={len(base['val_bpb'])} core_tasks={len(be['core_tasks'])} sft_steps={sft['steps']} rl_steps={rl['steps']} samples={len(samples)} -> {OUT}")
