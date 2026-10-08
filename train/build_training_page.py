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
EV5 = ["ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K", "HumanEval", "ChatCORE"]
def rl_cell(r, k):
    a, b = r["chat_eval_sft"].get(k), r["chat_eval_rl"].get(k)
    if a is None or b is None: return "–"
    if k == "ChatCORE": return f"{a:.3f} → {b:.3f}"
    d = b - a; cls = "ok" if d >= 1 else ("bad" if d <= -1 else "")
    return f"{a:.1f}% → {b:.1f}% <span class='{cls}'>{d:+.1f}</span>"
rleval = "".join(row(k, lambda r, k=k: rl_cell(r, k)) for k in EV5)
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
    d32_note = ""
    if "d32" in runs:
        r32 = runs["d32"]; gh32 = r32["base_train"]["train_minutes"] / 60 * 4; gh24 = mins / 60 * 8
        d32_note = (f" <b>d32 的收益。</b> 完整评测 CORE {r32['base_eval']['core']:.4f}，比 d24 高 {r32['base_eval']['core']-c_full:.2f}。它用了 {gh32:.0f} GPU 小时（4 卡 × {r32['base_train']['train_minutes']/60:.1f} 小时），d24 是 {gh24:.0f} GPU 小时，算力多 {gh32/gh24:.1f} 倍。"
                    f"按之前的外推，5 倍算力大约加 0.05，实际加了 {r32['base_eval']['core']-c_full:.2f}，说明在这个区间里规模的回报比保守估计更好。")
    core_note = (f"<div class='note'><b>d24 到了 GPT-2 的量级。</b> 训练结束时的抽样评测（每个任务 500 题）是 {c_in:.4f}，高于 GPT-2；随后的完整评测是 {c_full:.4f}，略低于 GPT-2。两个数字的差别来自题目抽样，说明我们正好落在 GPT-2 这条线附近。"
                 f"纯训练时间 {mins/60:.1f} 小时。作为参照：OpenAI 2019 年训练 GPT-2 用了 168 小时、约 4.3 万美元；nanochat 在 8 张 H100 上开 FP8 是 1.65 小时。A800 没有 FP8，算力约为 H100 的三分之一，{mins/60:.1f} 小时符合预期。{d32_note}</div>")

# yardstick table from bench results
yard = ""
try:
    bm = {}
    for m in ["GLM-5.3-Flash", "Qwen3.8-27B", "Qwen3.6-35B-A3B", "nanochat-d32-sft", "nanochat-d32-rl", "nanochat-d24-sft", "nanochat-d24-rl", "nanochat-d12-sft", "nanochat-d12-rl"]:
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
# ---- RL narrative computed from the runs
rl_runs = [t for t in tags if runs[t]["chat_eval_rl"].get("GSM8K") is not None and runs[t]["chat_eval_sft"].get("GSM8K") is not None]
def pk(r, which):
    p = r["chat_rl"]["pass_at_k"]
    if not p: return None, None
    if which == "first":
        ks = sorted(int(k.split("@")[1]) for k in p[0] if k.startswith("pass@")); return p[0]["pass@1"], (ks[-1], p[0][f"pass@{ks[-1]}"])
    return max(x["pass@1"] for x in p), None
rl_note = ""
if rl_runs:
    gains = "，".join(f"{t} 从 {runs[t]['chat_eval_sft']['GSM8K']:.2f}% 到 {runs[t]['chat_eval_rl']['GSM8K']:.2f}%" for t in rl_runs)
    lens = "，".join(f"{t} 从 {runs[t]['chat_rl']['reward_curve'][0][2]:.0f} 到 {runs[t]['chat_rl']['reward_curve'][-1][2]:.0f}" for t in rl_runs if runs[t]["chat_rl"]["reward_curve"])
    big = rl_runs[-1]; p1_0, (kmax, pk_0) = pk(runs[big], "first"); p1_best, _ = pk(runs[big], "best")
    hrs = "，".join(f"{t} {st(runs[t],'chat_rl')/3600:.1f} 小时" for t in rl_runs if st(runs[t],'chat_rl'))
    rl_note = (f"<div class='note kv'><b>三个现象。</b> 一是效果立竿见影，而且模型越大收益越大：GSM8K 上 {gains}。SFT 让模型看了 4 遍标准答案都没学会的东西，RL 让它自己试出来了。"
               f"二是 RL 不是凭空教会新本事，而是把偶尔能做对变成稳定做对：第 0 步时 {big} 对一道题采样 {kmax} 次、至少对一次的比例已经是 {pk_0*100:.0f}%，单次只对 {p1_0*100:.1f}%；训练后单次做对的比例升到 {p1_best*100:.1f}%。"
               f"三是回答变短了，平均 token 数 {lens}：模型发现啰嗦不加分，超过 256 个 token 被截断还会丢分，直接调用计算器给答案最划算。RL 优化的是你给的奖励，不是你心里想的目标。"
               f"这一步很慢，{hrs}，因为每步要让模型实际生成 256 条回答，走的是推理的速度。d24 和 d32 把每题的 16 个回答放进一次生成调用，d12 当时分两次，算法完全相同。</div>")
rl_eval_note = ""
if rl_runs:
    he = "，".join(f"{t} {runs[t]['chat_eval_rl']['HumanEval']-runs[t]['chat_eval_sft']['HumanEval']:+.1f}" for t in rl_runs if runs[t]['chat_eval_rl'].get('HumanEval') is not None)
    rl_eval_note = (f"<div class='note warn'><b>选择题几乎不动，写代码明显变差。</b> ARC 和 MMLU 的变化都在一两个点以内，属于正常波动。HumanEval 三个模型都掉了，分别是 {he} 个点。"
                    "RL 只奖励数学题的最终答案，模型就把所有回答都往短而直接的风格上推，写代码需要的完整函数和细节被一起压掉了。"
                    "这就是常说的对齐税：针对一个目标优化，会拿别的能力来换。工业界的做法是把多种任务混在一起做 RL，或者在 RL 目标里加一项约束，让模型别离 SFT 版本太远，也就是 PPO 和 GRPO 里的 KL 惩罚。nanochat 为了简单把它去掉了。</div>")
import re as _re
_CALL = _re.compile(r"&lt;\|python_start\|&gt;(.*?)&lt;\|python_end\|&gt;(?:&lt;\|output_start\|&gt;(.*?)&lt;\|output_end\|&gt;)?", _re.S)
def calc_html(t):
    # a call whose expression the calculator rejected has no output part: show it with "?" instead of leaving a tag open
    t = _CALL.sub(lambda m: f"<span class='calc'>计算器 {m.group(1)} = {m.group(2) if m.group(2) is not None else '?'}</span>", esc(t))
    return _re.sub(r"&lt;\|[a-z_]+\|&gt;", "", t)
def badge(x): return f"<span class='{'ok' if x['correct'] else 'bad'}'>{'✓ 对' if x['correct'] else '✗ 错'}</span>"
rl_samples = ""
for t in reversed(rl_runs):
    items = [it for it in runs[t].get("samples_math", []) if "sft" in it and "rl" in it]
    if not items: continue
    gsm = [it for it in items if it["kind"] == "gsm8k"]
    pick = [it for it in gsm if it["rl"]["correct"] and not it["sft"]["correct"]][:2] + [it for it in gsm if not it["rl"]["correct"]][:1] + [it for it in items if it["kind"] == "arith"]
    blocks = ""
    for it in pick:
        blocks += (f"<div class='qa'><div class='q'>{esc(it['q'])} <span class='pill'>参考答案 {esc(str(it['ref']))}</span></div><div class='duo'>"
                   + "".join(f"<div><div class='h'>{lab} · {badge(it[src])} · {it[src]['tokens']} token · 计算器 {it[src]['tool_calls']} 次</div>{calc_html(it[src]['text'])}</div>" for src, lab in (("sft", "SFT 后"), ("rl", "RL 后")))
                   + "</div></div>")
    ns = sum(it["sft"]["correct"] for it in gsm); nr = sum(it["rl"]["correct"] for it in gsm)
    cs = sum(it["sft"]["tool_calls"] for it in gsm); cr = sum(it["rl"]["tool_calls"] for it in gsm)
    ts = sum(it["sft"]["tokens"] for it in gsm) / max(1, len(gsm)); tr = sum(it["rl"]["tokens"] for it in gsm) / max(1, len(gsm))
    rl_samples += (f"<h3>{t}：{len(gsm)} 道测试题里，SFT 做对 {ns} 道，RL 做对 {nr} 道</h3>"
                   f"<p>同样 {len(gsm)} 道题，计算器调用从 {cs} 次变成 {cr} 次，平均回答长度从 {ts:.0f} 个 token 变成 {tr:.0f} 个。下面挑了 RL 做对而 SFT 没做对的题、一道 RL 也没做对的题，以及一道简单乘法。</p>" + blocks)
yard_note = ""
try:
    def acc(m, t):
        x = bm.get(m, {}).get("q", {}).get(t, {}).get("summary", {})
        return None if not x or x.get("accuracy") is None else 100 * x["accuracy"]
    g32s, g32r, glm = acc("nanochat-d32-sft", "gsm8k"), acc("nanochat-d32-rl", "gsm8k"), acc("GLM-5.3-Flash", "gsm8k")
    h32s, h32r = acc("nanochat-d32-sft", "humaneval"), acc("nanochat-d32-rl", "humaneval")
    nano_rl = runs.get("d32", {}).get("chat_eval_rl", {}).get("GSM8K")
    ce = [acc(m, "ceval") for m in bm if m.startswith("nanochat") and acc(m, "ceval") is not None]
    bug_parts = []
    for m in ["nanochat-d12-rl", "nanochat-d24-rl", "nanochat-d32-rl"]:
        bp = os.path.join(ROOT, "results", m, "quality_calc_bug.json")
        if os.path.exists(bp) and acc(m, "gsm8k") is not None:
            ob = json.load(open(bp))["tests"]["gsm8k"]["summary"]["accuracy"] * 100
            bug_parts.append(f"{m.replace('nanochat-', '')} 从 {ob:.0f}% 变成 {acc(m, 'gsm8k'):.0f}%")
    txt = ("<b>一个修正。</b> 第一次测时，接口服务在工作线程里跑生成，而 nanochat 的计算器靠 SIGALRM 设超时，只能在主线程里用。"
           "每次调用计算器都悄悄失败，模型拿不到结果就停止作答。"
           + (f"RL 后的模型几乎每一步都调用计算器、算完立刻写 #### 答案，受害最重，GSM8K 上 {'，'.join(bug_parts)}。SFT 版本的回答本来就冗长、常被截断，修复前后几乎不变，HumanEval 也不受影响。" if bug_parts else "")
           + "上表是修复后的数字。")
    if None not in (g32s, g32r, glm):
        txt += f"<br><b>观察。</b> 差距仍是数量级的：GSM8K 上 d32 做完 RL 是 {g32r:.0f}%，GLM-5.3-Flash 是 {glm:.1f}%。RL 的效果在这把尺子上也看得到，d32 从 {g32s:.0f}% 升到 {g32r:.0f}%。"
        if nano_rl is not None and g32r < nano_rl - 8:
            txt += (f"但比 nanochat 自己评测的 {nano_rl:.1f}% 低：这把尺子会在题目后面加一句要求按步骤解题、最后一行写 #### 数字，"
                    "RL 训练时见到的却是不带这句话的原题。小模型对提示词格式的变化非常敏感，RL 学到的东西有一部分是绑在训练格式上的。")
        elif nano_rl is not None:
            txt += f"这和 nanochat 自己在 1319 道题上评测的 {nano_rl:.1f}% 一致，这里只抽了 100 道题，误差大约正负 4 个点。"
    if None not in (h32s, h32r):
        txt += f" HumanEval 在 RL 之后同样下降，d32 从 {h32s:.0f}% 到 {h32r:.0f}%，和上面 nanochat 自己评测里看到的对齐税一致。"
    if ce:
        txt += f" C-Eval 是中文题，分词器和语料几乎全是英文，自训模型最高只有 {max(ce):.0f}%，还不到四选一随机猜的 25%。"
    yard_note = f"<div class='note'>{txt}</div>"
except Exception as e:
    yard_note = ""
T = open(os.path.join(ROOT, "train", "training_template.html"), encoding="utf-8").read()
for k, v in {"__YARD_NOTE__": yard_note, "__RL_NOTE__": rl_note, "__RL_EVAL__": rleval, "__RL_EVAL_NOTE__": rl_eval_note, "__RL_SAMPLES__": rl_samples, "__YARDSTICK__": yard, "__TPUT_NOTE__": tput_note, "__CORE_NOTE__": core_note, "__NOTE__": note_html, "__HEAD__": head, "__CFG__": cfg, "__PERF__": perf, "__SFT__": sftrows, "__EV__": evrows, "__RL__": rlrows, "__TIME__": timerows,
             "__TOK_SECS__": f(tok["train_secs"], 0), "__TOK2__": tokrows("GPT-2"), "__TOK4__": tokrows("GPT-4"), "__CORE_TASKS__": core_tasks, "__CORE_HEAD__": "".join(f"<th class='num'>{t}</th>" for t in tags),
             "__BASE_SAMPLES__": base_samples, "__CHAT_SAMPLES__": chat_samples, "__DATA__": json.dumps(chart, ensure_ascii=False), "__TAGS__": json.dumps(tags)}.items():
    T = T.replace(k, v)
open(os.path.join(ROOT, "docs", "training.html"), "w", encoding="utf-8").write(T)
print("docs/training.html written for runs:", tags)
