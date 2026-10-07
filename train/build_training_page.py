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
chart = {t: {"tput": runs[t]["base_train"].get("tput_curve", []), "loss": runs[t]["base_train"]["loss_curve"], "val": runs[t]["base_train"]["val_bpb"], "sft_loss": runs[t]["chat_sft"]["loss_curve"], "sft_val": runs[t]["chat_sft"]["val_bpb"],
             "chatcore": runs[t]["chat_sft"]["chatcore"], "reward": runs[t]["chat_rl"]["reward_curve"], "pass": runs[t]["chat_rl"]["pass_at_k"]} for t in tags}
note_html = f"<div class='note warn'>{NOTE}</div>" if NOTE else ""
tput_note = core_note = ""
if "d24" in runs:
    tp = runs["d24"]["base_train"].get("tput_curve", [])
    early = [p[2] for p in tp if p[0] < 35]; late = [p[2] for p in tp if p[0] > 60]
    if early and late and sum(late)/len(late) > sum(early)/len(early) * 1.15:
        tput_note = f" 更有意思的是 d24 曲线上的那个台阶：前 43 分钟 MFU 是 {sum(early)/len(early):.0f}%，之后跳到 {sum(late)/len(late):.0f}% 并保持到结束。训练启动时 GPU 2 和 GPU 5 上有另一个项目的仿真任务在跑，占走了这两张卡的一部分算力。分布式训练每一步都要等所有卡算完才能汇总梯度，所以两张卡被拖慢，八张卡一起慢。台阶出现的时刻很可能就是那些任务结束的时刻，之后速度再没有波动。结论：多卡训练的速度由最慢的那张卡决定。"
    c_in = runs["d24"]["base_train"]["core_in_train"][-1][1] if runs["d24"]["base_train"]["core_in_train"] else None
    c_full = runs["d24"]["base_eval"]["core"]; mins = runs["d24"]["base_train"]["train_minutes"]
    core_note = (f"<div class='note'><b>d24 到了 GPT-2 的量级。</b> 训练结束时的抽样评测（每个任务 500 题）是 {c_in:.4f}，高于 GPT-2；随后的完整评测是 {c_full:.4f}，略低于 GPT-2。两个数字的差别来自题目抽样，说明我们正好落在 GPT-2 这条线附近。"
                 f"纯训练时间 {mins/60:.1f} 小时。作为参照：OpenAI 2019 年训练 GPT-2 用了 168 小时、约 4.3 万美元；nanochat 在 8 张 H100 上开 FP8 是 1.65 小时。A800 没有 FP8，算力约为 H100 的三分之一，{mins/60:.1f} 小时符合预期。</div>")

# yardstick table from bench results
yard = ""
try:
    bm = {}
    for m in ["GLM-5.3-Flash", "Qwen3.8-27B", "Qwen3.6-35B-A3B", "nanochat-d24-sft", "nanochat-d12-sft"]:
        qp = os.path.join(ROOT, "results", m, "quality.json"); pp = os.path.join(ROOT, "results", m, "perf", "single_128in_256out.json")
        if os.path.exists(qp): bm[m] = {"q": json.load(open(qp))["tests"], "p": json.load(open(pp)) if os.path.exists(pp) else {}}
    tests = [("gsm8k", "GSM8K"), ("humaneval", "HumanEval"), ("ceval", "C-Eval"), ("json", "JSON"), ("tools", "工具调用")]
    rows = ""
    for m, d in bm.items():
        cells = "".join(f"<td class='num'>{100*d['q'][t]['summary']['accuracy']:.1f}% <span class='pill'>{d['q'][t]['summary']['correct']}/{d['q'][t]['summary']['n']}</span></td>" if t in d["q"] and d["q"][t]["summary"].get("accuracy") is not None else "<td class='num'>–</td>" for t, _ in tests)
        tp = d["p"].get("median_tpot_ms"); cells += f"<td class='num'>{(1000/tp):.0f} tok/s</td>" if tp else "<td class='num'>–</td>"
        rows += f"<tr><td>{m}</td>{cells}</tr>"
    if rows: yard = "<div class='tablewrap'><table><tr><th>模型</th>" + "".join(f"<th class='num'>{n}</th>" for _, n in tests) + "<th class='num'>单流速度</th></tr>" + rows + "</table></div>"
except Exception as e:
    yard = f"<p>（评测结果缺失：{e}）</p>"
T = open(os.path.join(ROOT, "train", "training_template.html"), encoding="utf-8").read()
for k, v in {"__YARDSTICK__": yard, "__TPUT_NOTE__": tput_note, "__CORE_NOTE__": core_note, "__NOTE__": note_html, "__HEAD__": head, "__CFG__": cfg, "__PERF__": perf, "__SFT__": sftrows, "__EV__": evrows, "__RL__": rlrows, "__TIME__": timerows,
             "__TOK_SECS__": f(tok["train_secs"], 0), "__TOK2__": tokrows("GPT-2"), "__TOK4__": tokrows("GPT-4"), "__CORE_TASKS__": core_tasks, "__CORE_HEAD__": "".join(f"<th class='num'>{t}</th>" for t in tags),
             "__BASE_SAMPLES__": base_samples, "__CHAT_SAMPLES__": chat_samples, "__DATA__": json.dumps(chart, ensure_ascii=False), "__TAGS__": json.dumps(tags)}.items():
    T = T.replace(k, v)
open(os.path.join(ROOT, "docs", "training.html"), "w", encoding="utf-8").write(T)
print("docs/training.html written for runs:", tags)
