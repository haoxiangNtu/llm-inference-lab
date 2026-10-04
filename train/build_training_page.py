#!/usr/bin/env python3
"""Build docs/training.html from results/training/<tag>.json (produced by train/parse_logs.py).
Usage: build_training_page.py <repo_root> [status note for runs still in progress]"""
import glob, json, os, sys
ROOT = sys.argv[1]; NOTE = sys.argv[2] if len(sys.argv) > 2 else ""
runs = {}
for p in sorted(glob.glob(os.path.join(ROOT, "results", "training", "*.json"))):
    d = json.load(open(p)); runs[d["tag"]] = d
tags = [t for t in ["d12", "d24"] if t in runs] + [t for t in runs if t not in ("d12", "d24")]
def f(v, nd=0, suf=""): return "–" if v is None else f"{v:,.{nd}f}{suf}"
def st(run, name):
    for s in run["stages"]:
        if s["name"] == name: return s["secs"]
    return None
def hms(s):
    if s is None: return "–"
    return f"{s//3600}h {s%3600//60}m" if s >= 3600 else (f"{s//60}m {s%60}s" if s >= 60 else f"{s}s")

# ---------- tables
def row(label, fn): return "<tr><td>" + label + "</td>" + "".join(f"<td class='num'>{fn(runs[t])}</td>" for t in tags) + "</tr>"
head = "<tr><th></th>" + "".join(f"<th class='num'>{t}</th>" for t in tags) + "</tr>"
cfg = "".join([
    row("层数 × 宽度", lambda r: f"{r['base_train'].get('config',{}).get('n_layer','–')} × {r['base_train'].get('config',{}).get('n_embd','–')}"),
    row("总参数", lambda r: f(r["base_train"]["params"].get("total", 0) / 1e6, 0, " M")),
    row("其中 Transformer 矩阵", lambda r: f(r["base_train"]["params"].get("transformer_matrices", 0) / 1e6, 0, " M")),
    row("训练 token 数", lambda r: f((r["base_train"]["total_tokens"] or 0) / 1e9, 2, " B")),
    row("总计算量", lambda r: f"{(r['base_train']['total_flops'] or 0):.2e} FLOP"),
    row("总批大小（token）", lambda r: f(r["base_train"]["total_batch_size"])),
    row("每卡批 × 序列长度", lambda r: f"{r['base_train']['device_batch']} × 2048，梯度累积 {f(r['base_train']['grad_accum'])}"),
    row("优化步数", lambda r: f(r["base_train"].get("last_step", 0) + 1 if r["base_train"].get("last_step") is not None else None)),
    row("注意力内核", lambda r: "FlashAttention 3" if r["base_train"]["fa3"] else "PyTorch SDPA（回退）"),
])
perf = "".join([
    row("吞吐", lambda r: f(r["base_train"].get("avg_tok_s"), 0, " tok/s")),
    row("每步耗时", lambda r: f(r["base_train"].get("avg_dt_ms"), 0, " ms")),
    row("MFU（bf16 峰值算力利用率）", lambda r: f(r["base_train"].get("avg_mfu"), 1, "%")),
    row("每卡峰值显存", lambda r: f((r["base_train"]["peak_mem_mib"] or 0) / 1024, 1, " GB")),
    row("纯训练时间", lambda r: f(r["base_train"]["train_minutes"], 1, " 分钟")),
    row("最终验证 bpb", lambda r: f(r["base_train"]["min_val_bpb"], 4)),
    row("CORE（22 项，GPT-2 为 0.2565）", lambda r: f(r["base_eval"]["core"], 4)),
])
sftrows = "".join([
    row("训练混合行数", lambda r: f(r["chat_sft"]["mixture_rows"])),
    row("步数 / 时间", lambda r: f"{f(r['chat_sft']['steps'])} 步 / {f(r['chat_sft']['train_minutes'],1)} 分钟"),
    row("验证 bpb（起点 → 终点）", lambda r: (f"{r['chat_sft']['val_bpb'][0][1]:.3f} → {r['chat_sft']['val_bpb'][-1][1]:.3f}" if r["chat_sft"]["val_bpb"] else "–")),
])
EV = ["ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K", "HumanEval", "ChatCORE"]
evrows = "".join(row(k + ("（随机 25%）" if k in ("ARC-Easy", "ARC-Challenge", "MMLU") else ""), lambda r, k=k: (f"{r['chat_eval_sft'][k]:.2f}%" if k != "ChatCORE" else f"{r['chat_eval_sft'][k]:.4f}") if k in r["chat_eval_sft"] else "–") for k in EV)
rlrows = "".join([
    row("步数 / 时间", lambda r: f"{f(r['chat_rl']['steps'])} 步 / {hms(st(r,'chat_rl'))}" if r["chat_rl"]["steps"] else "未运行"),
    row("GSM8K：SFT 后", lambda r: f"{r['chat_eval_sft'].get('GSM8K', 0):.2f}%" if r["chat_eval_sft"] else "–"),
    row("GSM8K：RL 后", lambda r: f"{r['chat_eval_rl']['GSM8K']:.2f}%" if r["chat_eval_rl"].get("GSM8K") is not None else "–"),
    row("训练中 pass@1（400 题）", lambda r: (f"{r['chat_rl']['pass_at_k'][0]['pass@1']*100:.1f}% → {max(p['pass@1'] for p in r['chat_rl']['pass_at_k'])*100:.1f}%" if r["chat_rl"]["pass_at_k"] else "–")),
    row("平均回答长度（token）", lambda r: (f"{r['chat_rl']['reward_curve'][0][2]:.0f} → {r['chat_rl']['reward_curve'][-1][2]:.0f}" if r["chat_rl"]["reward_curve"] else "–")),
])
STAGES = [("tok_train", "训练分词器"), ("tok_eval", "分词器评测"), ("base_train", "预训练"), ("base_eval", "基础模型评测"), ("chat_sft", "SFT"), ("chat_eval_sft", "对话评测"), ("chat_rl", "强化学习"), ("chat_eval_rl", "RL 后评测")]
timerows = "".join(row(lbl, lambda r, n=n: hms(st(r, n))) for n, lbl in STAGES)
r0 = runs[tags[0]]
tok = r0["tokenizer"]
def tokrows(ref):
    return "".join(f"<tr><td>{x['text']}</td><td class='num'>{x['bytes']:,}</td><td class='num'>{x['ref_ratio']:.2f}</td><td class='num'>{x['our_ratio']:.2f}</td><td class='num'>{x['diff_pct']:+.1f}%</td></tr>" for x in tok["compare"].get(ref, []))
core_tasks = "".join(f"<tr><td>{x['task']}</td><td>{'选择题' if x['type']=='multiple_choice' else ('模式匹配' if x['type']=='schema' else '续写')}</td>" + "".join(
    f"<td class='num'>{next((y['acc'] for y in runs[t]['base_eval']['core_tasks'] if y['task']==x['task']), float('nan')):.3f}</td>" for t in tags) + "</tr>" for x in r0["base_eval"]["core_tasks"])
def esc(s): return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
base_samples = "".join(f"<pre>{esc(s)}</pre>" for s in runs[tags[-1]]["base_eval"]["samples"][:3])
chat_samples = "".join(f"<h3>{t}</h3>" + "".join(f"<div class='qa'><div class='q'>{esc(x['q'])}</div><div class='a'>{esc(x['a'])}</div></div>" for x in runs[t]["samples"]) for t in tags if runs[t]["samples"])
chart = {t: {"loss": runs[t]["base_train"]["loss_curve"], "val": runs[t]["base_train"]["val_bpb"], "sft_loss": runs[t]["chat_sft"]["loss_curve"], "sft_val": runs[t]["chat_sft"]["val_bpb"],
             "chatcore": runs[t]["chat_sft"]["chatcore"], "reward": runs[t]["chat_rl"]["reward_curve"], "pass": runs[t]["chat_rl"]["pass_at_k"]} for t in tags}
note_html = f"<div class='note warn'>{NOTE}</div>" if NOTE else ""

T = open(os.path.join(ROOT, "train", "training_template.html"), encoding="utf-8").read()
for k, v in {"__NOTE__": note_html, "__HEAD__": head, "__CFG__": cfg, "__PERF__": perf, "__SFT__": sftrows, "__EV__": evrows, "__RL__": rlrows, "__TIME__": timerows,
             "__TOK_SECS__": f(tok["train_secs"], 0), "__TOK2__": tokrows("GPT-2"), "__TOK4__": tokrows("GPT-4"), "__CORE_TASKS__": core_tasks, "__CORE_HEAD__": "".join(f"<th class='num'>{t}</th>" for t in tags),
             "__BASE_SAMPLES__": base_samples, "__CHAT_SAMPLES__": chat_samples, "__DATA__": json.dumps(chart, ensure_ascii=False), "__TAGS__": json.dumps(tags)}.items():
    T = T.replace(k, v)
open(os.path.join(ROOT, "docs", "training.html"), "w", encoding="utf-8").write(T)
print("docs/training.html written for runs:", tags)
