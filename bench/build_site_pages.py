#!/usr/bin/env python3
"""Generate docs/benchmark.html from results/ and docs/lab.html from the interactive lab source.
Usage: build_site_pages.py <repo_root> [lab_source.html]"""
import glob, json, os, re, sys
ROOT = sys.argv[1]; LAB_SRC = sys.argv[2] if len(sys.argv) > 2 else None
RES = os.path.join(ROOT, "results")
NAV = '''<header class="top"><a class="brand" href="../index.html">LLM Inference Lab</a>
<nav><a{l} href="lab.html">原理实验台</a><a href="metrics.html">指标解读</a><a{b} href="benchmark.html">评测报告</a><a href="deployment.html">部署实录</a><a href="training.html">训练实录</a><a href="https://github.com/haoxiangNtu/llm-inference-lab">GitHub</a></nav></header>'''
FOOT = '<footer>LLM Inference Lab · 在 8 × A800 上部署并评测开源大模型的学习记录 · <a href="https://github.com/haoxiangNtu/llm-inference-lab">源码</a></footer>'
HEAD = '''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>{title}</title>
<meta name="description" content="{desc}">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700&family=JetBrains+Mono:wght@400;600&display=swap">
<link rel="stylesheet" href="site.css">
{extra}</head>
<body>
<div class="wrap">
'''
CONFIG = {"GLM-5.3-Flash": "8 × A800，TP4 × PP2，FP8 权重 Marlin W8A16，max-model-len 1M，max-num-seqs 16",
          "Qwen3.6-35B-A3B": "2 × A800，TP2，BF16，max-model-len 128K，max-num-seqs 64",
          "Qwen3.8-27B": "2 × A800，TP2，BF16，max-model-len 128K，max-num-seqs 64",
          "nanochat-d32-sft": "自己训练的 d32（28 亿参数，SFT 后），1 × A800，nanochat 引擎无批处理，上下文 2048，质量题量减半",
          "nanochat-d24-sft": "自己训练的 d24（13.8 亿参数，SFT 后），1 × A800，nanochat 引擎无批处理，上下文 2048，质量题量减半",
          "nanochat-d12-sft": "自己训练的 d12（2.9 亿参数，SFT 后），1 × A800，nanochat 引擎无批处理，上下文 2048，质量题量减半",
          "nanochat-d32-rl": "d32 在 GSM8K 上做完强化学习后，其余同上，只测质量",
          "nanochat-d24-rl": "d24 在 GSM8K 上做完强化学习后，其余同上，只测质量",
          "nanochat-d12-rl": "d12 在 GSM8K 上做完强化学习后，其余同上，只测质量"}
ORDER = ["single_128in_256out", "prefill_2k", "prefill_8k", "prefill_32k", "conc_4_1kin_256out", "conc_8_1kin_256out", "conc_16_1kin_256out", "conc_32_1kin_256out", "conc_16_4kin_512out"]
SCEN = {"single_128in_256out": "单流 128 入 / 256 出", "prefill_2k": "单流 2K 入", "prefill_8k": "单流 8K 入", "prefill_32k": "单流 32K 入",
        "conc_4_1kin_256out": "并发 4，1K 入 / 256 出", "conc_8_1kin_256out": "并发 8，1K 入 / 256 出", "conc_16_1kin_256out": "并发 16，1K 入 / 256 出",
        "conc_32_1kin_256out": "并发 32，1K 入 / 256 出", "conc_16_4kin_512out": "并发 16，4K 入 / 512 出"}
models = [m for m in ["GLM-5.3-Flash", "Qwen3.6-35B-A3B", "Qwen3.8-27B", "nanochat-d32-sft", "nanochat-d32-rl", "nanochat-d24-sft", "nanochat-d24-rl", "nanochat-d12-sft", "nanochat-d12-rl"] if os.path.isdir(os.path.join(RES, m))]
perf, qual = {}, {}
for m in models:
    perf[m] = {}
    for p in glob.glob(os.path.join(RES, m, "perf", "*.json")):
        d = json.load(open(p)); n = max(1, d.get("num_prompts") or 1)
        d["_in"] = round((d.get("total_input_tokens") or 0) / n); d["_out"] = round((d.get("total_output_tokens") or 0) / n)
        perf[m][os.path.basename(p)[:-5]] = d
    qp = os.path.join(RES, m, "quality.json")
    if os.path.exists(qp): qual[m] = json.load(open(qp))
f = lambda v, nd=0: "-" if v is None else f"{v:,.{nd}f}"
rows = []
for m in models:
    for name in ORDER:
        d = perf[m].get(name)
        if not d: continue
        tp = d.get("median_tpot_ms"); ttft = d.get("median_ttft_ms")
        rows.append(f"<tr><td>{m}</td><td>{SCEN.get(name, name)}</td><td class='num'>{d.get('max_concurrency','-')}</td><td class='num'>{d['_in']} / {d['_out']}</td><td class='num'>{f(d.get('output_throughput'))}</td><td class='num'>{f(d.get('total_token_throughput'))}</td><td class='num'>{f(ttft)}</td><td class='num'>{f(d.get('p99_ttft_ms'))}</td><td class='num'>{f(d['_in']/(ttft/1000)) if ttft else '-'}</td><td class='num'>{f(tp,1)}</td><td class='num'>{f(d.get('p99_tpot_ms'),1)}</td><td class='num'>{f(1000/tp) if tp else '-'}</td><td class='num'>{f((d.get('median_e2el_ms') or 0)/1000,1)}</td></tr>")
perf_table = "<div class='tablewrap'><table><tr><th>模型</th><th>场景</th><th class='num'>并发</th><th class='num'>输入 / 输出 token</th><th class='num'>输出吞吐 tok/s</th><th class='num'>总吞吐 tok/s</th><th class='num'>TTFT 中位 ms</th><th class='num'>TTFT P99 ms</th><th class='num'>Prefill tok/s</th><th class='num'>TPOT 中位 ms</th><th class='num'>TPOT P99 ms</th><th class='num'>单流 tok/s</th><th class='num'>端到端中位 s</th></tr>" + "".join(rows) + "</table></div>"
tests = [("gsm8k", "GSM8K"), ("humaneval", "HumanEval"), ("ceval", "C-Eval"), ("json", "JSON"), ("tools", "工具调用"), ("vision", "图形题")]
qrows = []
for m in models:
    cells = []
    for t, _ in tests:
        s = qual.get(m, {}).get("tests", {}).get(t, {}).get("summary")
        cells.append("<td class='num'>-</td>" if not s or s.get("accuracy") is None else f"<td class='num'>{100*s['accuracy']:.1f}% <span class='pill'>{s['correct']}/{s['n']}</span></td>")
    qrows.append(f"<tr><td>{m}</td>" + "".join(cells) + "</tr>")
qual_table = "<div class='tablewrap'><table><tr><th>模型</th>" + "".join(f"<th class='num'>{n}</th>" for _, n in tests) + "</tr>" + "".join(qrows) + "</table></div>"
lens = sorted({int(L) for q in qual.values() for L in q["tests"].get("needle", {}).get("summary", {}).get("per_length", {})})
nrows = []
for m in models:
    pl = qual.get(m, {}).get("tests", {}).get("needle", {}).get("summary", {}).get("per_length", {})
    cells = []
    for L in lens:
        s = pl.get(str(L)) or pl.get(L)
        cells.append("<td class='num'>-</td>" if not s else f"<td class='num'>{100*s['accuracy']:.0f}% / {s['avg_latency_s']:.1f}s</td>")
    nrows.append(f"<tr><td>{m}</td>" + "".join(cells) + "</tr>")
needle_table = "<div class='tablewrap'><table><tr><th>模型</th>" + "".join(f"<th class='num'>{L//1024}K</th>" for L in lens) + "</tr>" + "".join(nrows) + "</table></div>"
trows = []
for m in models:
    cells = []
    for t in ["gsm8k", "humaneval", "ceval"]:
        s = qual.get(m, {}).get("tests", {}).get(t, {}).get("summary")
        cells.append("<td class='num'>-</td>" if not s else f"<td class='num'>{s['avg_completion_tokens']:.0f} tok / {s['avg_latency_s']:.1f}s / 截断 {s.get('truncated',0)}</td>")
    trows.append(f"<tr><td>{m}</td>" + "".join(cells) + "</tr>")
think_table = "<div class='tablewrap'><table><tr><th>模型</th><th class='num'>GSM8K</th><th class='num'>HumanEval</th><th class='num'>C-Eval</th></tr>" + "".join(trows) + "</table></div>"
notes = "".join(f"<li><b>{m}</b>：{'；'.join(q.get('notes', []))}</li>" for m, q in qual.items() if q.get("notes"))
# chart data
chart = {"models": models, "pmodels": [m for m in models if glob.glob(os.path.join(RES, m, "perf", "*.json"))], "qmodels": [m for m in ["GLM-5.3-Flash", "Qwen3.6-35B-A3B", "Qwen3.8-27B", "nanochat-d32-sft", "nanochat-d32-rl"] if m in models], "conc": {}, "quality": {}, "needle": {}}
for m in models:
    pts = []
    for name, c in [("single_128in_256out", 1), ("conc_4_1kin_256out", 4), ("conc_8_1kin_256out", 8), ("conc_16_1kin_256out", 16), ("conc_32_1kin_256out", 32)]:
        d = perf[m].get(name)
        if d: pts.append({"c": c, "tps": d.get("output_throughput"), "tpot": d.get("median_tpot_ms"), "ttft": d.get("median_ttft_ms")})
    chart["conc"][m] = pts
    chart["quality"][m] = {t: (qual.get(m, {}).get("tests", {}).get(t, {}).get("summary", {}) or {}).get("accuracy") for t, _ in tests}
    pl = qual.get(m, {}).get("tests", {}).get("needle", {}).get("summary", {}).get("per_length", {})
    chart["needle"][m] = [{"L": int(k), "lat": v["avg_latency_s"], "acc": v["accuracy"]} for k, v in sorted(pl.items(), key=lambda kv: int(kv[0]))]
bug_parts = []
for m in models:
    bp = os.path.join(RES, m, "quality_calc_bug.json")
    if m.endswith("-rl") and os.path.exists(bp) and m in qual:
        try:
            ob = json.load(open(bp))["tests"]["gsm8k"]["summary"]["accuracy"]; nb = qual[m]["tests"]["gsm8k"]["summary"]["accuracy"]
            bug_parts.append(f"{m.replace('nanochat-', '')} 从 {100*ob:.0f}% 变成 {100*nb:.0f}%")
        except Exception:
            pass
bugtxt = (f"RL 版本几乎每一步都调用计算器，受害最重，GSM8K 上 {'，'.join(bug_parts)}；SFT 版本和 HumanEval 修复前后几乎不变。" if bug_parts else "")
cfg_rows = "".join(f"<tr><td>{m}</td><td>{CONFIG.get(m,'')}</td></tr>" for m in models)
body = f'''{NAV.format(l="", b=' class="on"')}
<div class="eyebrow">Benchmark · 2026-09-30</div>
<h1>三个模型在同一套脚本下的成绩</h1>
<p class="lead">性能用 <code>vllm bench serve</code> 的随机数据集和固定长度压测，质量用 <a href="metrics.html">指标解读</a>里定义的七项测试。三个部署的卡数和并行方式不同，比较时请对照配置。</p>
<div class="toc"><a href="#config">部署配置</a><a href="#perf">性能</a><a href="#quality">质量</a><a href="#needle">长上下文</a><a href="#think">思考开销</a><a href="#fair">修正记录</a></div>

<h2 id="config">部署配置</h2>
<div class="tablewrap"><table><tr><th>模型</th><th>配置</th></tr>{cfg_rows}</table></div>

<h2 id="perf">性能</h2>
<div class="stage"><canvas class="viz" id="cConc" width="1040" height="320"></canvas></div>
<div class="legend">{"".join(f'<span><i style="background:var({c})"></i>{m}</span>' for m, c in zip(chart["pmodels"], ["--compute","--token","--memory","--warn","--kv","--weights"]))}</div>
<p>左图是输出吞吐随并发的变化，右图是单请求 TPOT 随并发的变化。并发 1 的点来自 128 入 / 256 出的单流场景，其余来自 1K 入 / 256 出。</p>
<div class="note mem">三个现象。激活参数决定 decode 速度：同样 2 张卡，Qwen3.6 每步只读 3B 激活参数，Qwen3.8 要读全部 27B，单流速度差 3.5 倍。并发曲线的形状：Qwen3.6 从并发 1 到 32 吞吐涨 11 倍而 TPOT 只从 5.8 涨到 17 ms，这是"权重每步只读一次"的收益。GLM 的吞吐在并发 16 和 32 都是 290 tok/s，32 时 TTFT 跳到 15.6 秒，因为服务的 max-num-seqs 设了 16，多出的请求在排队；加上 Ampere 上的 Triton 回退内核和流水线气泡，每卡吞吐只有 Qwen3.6 的 1/23，是最值得优化的一项。</div>
{perf_table}

<div class="note kv"><b>关于 nanochat 的六行。</b> 这是<a href="training.html">训练实录</a>里自己从零训练的三个模型，各有 SFT 后和强化学习后两个版本，通过一个 OpenAI 兼容的小服务接进同一套脚本。它们跑在单张卡的 nanochat 引擎上，没有批处理，上下文只有 2048，所以只测了单流和并发 4、没有长上下文档，质量测试题量减半，RL 版本只测质量。数字本身不好看，但"自己训的模型和前沿开源模型差多远"从此有了同一口径的答案。<br><b>一个修正：</b>第一次测时，那个小服务在工作线程里跑生成，而 nanochat 的计算器靠 SIGALRM 设超时，只能在主线程用。每次调用计算器都悄悄失败，模型拿不到结果就停止作答。{bugtxt}修复后全部重测，下面是修复后的数字。</div>

<h2 id="quality">质量</h2>
<div class="stage"><canvas class="viz" id="cQual" width="1040" height="300"></canvas></div>
{qual_table}
<p>Qwen3.8-27B 是纯文本模型，没有图形题成绩。截断按答错计。</p>

<h2 id="needle">长上下文大海捞针</h2>
<div class="stage"><canvas class="viz" id="cNeedle" width="1040" height="280"></canvas></div>
{needle_table}
<p>三个模型在所有测试长度上全部检出。Qwen 的 128K 档实际 149K token，超过服务配置的 131072 上限被拒绝，改测 107K 档。延迟差异反映 prefill 速度：Qwen3.6 的 MoE 在长输入上最快，Qwen3.8 稠密最慢。</p>

<h2 id="think">思考模式的开销</h2>
<p>平均输出 token 数、单题耗时（并发 16 下）和截断题数。截断指思考超过 4096 token 预算、没来得及给出答案。</p>
{think_table}
<p>Qwen3.6 是最"话多"的思考者，HumanEval 有 31 题在预算内没写完代码，JSON 抽取这种简单任务也要先想一千多个 token；GLM 的答案最简洁、质量最高，代价是慢。给思考模型更大预算或关掉思考，分数会明显不同。</p>

<h2 id="fair">测试过程中修正的不公平</h2>
<ul>{notes}</ul>
<p>原始结果在仓库 <code>results/</code> 目录，每条记录保留了模型输出的末尾、判分依据和 token 用量。</p>
{FOOT}
</div>
<script>
const DATA={json.dumps(chart, ensure_ascii=False)};
(function(){{
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const COLORS=["--compute","--token","--memory","--warn","--kv","--weights"];
function fit(c){{const r=c.getBoundingClientRect();const dpr=Math.min(2,devicePixelRatio||1);const W=+c.getAttribute('width'),H=+c.getAttribute('height');c.width=Math.round(r.width*dpr);c.height=Math.round(r.width*dpr*H/W);const g=c.getContext('2d');g.setTransform(dpr*r.width/W,0,0,dpr*r.width/W,0,0);return g;}}
function axes(g,x0,y0,x1,y1){{g.strokeStyle=css('--line');g.beginPath();g.moveTo(x0,y0);g.lineTo(x0,y1);g.lineTo(x1,y1);g.stroke();}}
function drawConc(){{const c=document.getElementById('cConc');const g=fit(c);g.clearRect(0,0,1040,320);g.font='11px '+css('--font-mono');g.textBaseline='middle';
  const panels=[{{key:'tps',title:'输出吞吐 tok/s',x0:60,x1:500}},{{key:'tpot',title:'TPOT 中位 ms',x0:600,x1:1020}}];const xs=[1,4,8,16,32];
  panels.forEach(p=>{{const y0=40,y1=270;axes(g,p.x0,y0,p.x1,y1);const vals=[].concat(...DATA.pmodels.map(m=>DATA.conc[m].map(q=>q[p.key]||0)));const max=Math.max(...vals)*1.1||1;
    g.fillStyle=css('--muted');g.textAlign='left';g.fillText(p.title,p.x0,22);g.textAlign='center';xs.forEach((v,i)=>g.fillText(v,p.x0+(p.x1-p.x0)*i/(xs.length-1),y1+14));g.fillText('并发请求数',(p.x0+p.x1)/2,y1+32);
    g.textAlign='right';[0,0.5,1].forEach(t=>{{g.fillText(Math.round(max*t),p.x0-6,y1-(y1-y0)*t);}});
    DATA.pmodels.forEach((m,mi)=>{{const col=css(COLORS[mi]);g.strokeStyle=col;g.fillStyle=col;g.lineWidth=2.5;g.beginPath();DATA.conc[m].forEach((q,i)=>{{const xi=xs.indexOf(q.c);const x=p.x0+(p.x1-p.x0)*xi/(xs.length-1),y=y1-(y1-y0)*(q[p.key]||0)/max;i?g.lineTo(x,y):g.moveTo(x,y);}});g.stroke();DATA.conc[m].forEach(q=>{{const xi=xs.indexOf(q.c);const x=p.x0+(p.x1-p.x0)*xi/(xs.length-1),y=y1-(y1-y0)*(q[p.key]||0)/max;g.beginPath();g.arc(x,y,3.5,0,7);g.fill();}});g.lineWidth=1;}});}});}}
function drawQual(){{const c=document.getElementById('cQual');const g=fit(c);g.clearRect(0,0,1040,300);g.font='11px '+css('--font-mono');g.textBaseline='middle';const tests=[['gsm8k','GSM8K'],['humaneval','HumanEval'],['ceval','C-Eval'],['json','JSON'],['tools','工具调用'],['vision','图形题']];
  const x0=50,x1=1020,y0=30,y1=250;axes(g,x0,y0,x1,y1);g.fillStyle=css('--muted');g.textAlign='right';[0,0.5,1].forEach(t=>g.fillText(Math.round(100*t)+'%',x0-6,y1-(y1-y0)*t));
  const gw=(x1-x0)/tests.length,bw=gw/(DATA.qmodels.length+1.2);
  tests.forEach((t,ti)=>{{g.fillStyle=css('--muted');g.textAlign='center';g.fillText(t[1],x0+gw*ti+gw/2,y1+16);DATA.qmodels.forEach((m,mi)=>{{const v=DATA.quality[m][t[0]];if(v==null)return;const x=x0+gw*ti+bw*0.6+bw*mi;const h=(y1-y0)*v;g.fillStyle=css(COLORS[mi]);g.globalAlpha=0.9;g.fillRect(x,y1-h,bw*0.9,h);g.globalAlpha=1;g.fillStyle=css('--fg');g.fillText(Math.round(v*100),x+bw*0.45,y1-h-9);}});}});
  g.textAlign='left';DATA.qmodels.forEach((m,mi)=>{{g.fillStyle=css(COLORS[mi]);g.fillRect(x0+mi*190,y1+34,10,10);g.fillStyle=css('--muted');g.fillText(m,x0+mi*190+16,y1+39);}});}}
function drawNeedle(){{const c=document.getElementById('cNeedle');const g=fit(c);g.clearRect(0,0,1040,280);g.font='11px '+css('--font-mono');g.textBaseline='middle';const x0=60,x1=1020,y0=30,y1=230;axes(g,x0,y0,x1,y1);
  const Ls=[...new Set([].concat(...DATA.models.map(m=>DATA.needle[m].map(q=>q.L))))].sort((a,b)=>a-b);const lx=L=>x0+(x1-x0)*(Math.log2(L)-Math.log2(Ls[0]))/(Math.log2(Ls[Ls.length-1])-Math.log2(Ls[0]));
  const max=Math.max(...[].concat(...DATA.models.map(m=>DATA.needle[m].map(q=>q.lat))))*1.15;g.fillStyle=css('--muted');g.textAlign='center';Ls.forEach(L=>g.fillText(Math.round(L/1024)+'K',lx(L),y1+14));g.fillText('上下文长度（对数轴）',(x0+x1)/2,y1+32);g.textAlign='right';[0,0.5,1].forEach(t=>g.fillText((max*t).toFixed(0)+'s',x0-6,y1-(y1-y0)*t));g.textAlign='left';g.fillText('平均延迟（含 prefill），点上数字是准确率',x0,18);
  DATA.models.forEach((m,mi)=>{{const col=css(COLORS[mi]);g.strokeStyle=col;g.fillStyle=col;g.lineWidth=2.5;g.beginPath();DATA.needle[m].forEach((q,i)=>{{const x=lx(q.L),y=y1-(y1-y0)*q.lat/max;i?g.lineTo(x,y):g.moveTo(x,y);}});g.stroke();g.lineWidth=1;DATA.needle[m].forEach(q=>{{const x=lx(q.L),y=y1-(y1-y0)*q.lat/max;g.beginPath();g.arc(x,y,3.5,0,7);g.fill();g.fillStyle=css('--fg');g.textAlign='center';g.fillText(Math.round(q.acc*100)+'%',x,y-12);g.fillStyle=col;}});}});}}
function all(){{drawConc();drawQual();drawNeedle();}}all();addEventListener('resize',all);matchMedia('(prefers-color-scheme: dark)').addEventListener('change',all);
}})();
</script>
</body>
</html>
'''
open(os.path.join(ROOT, "docs", "benchmark.html"), "w", encoding="utf-8").write(HEAD.format(title="评测报告 · LLM Inference Lab", desc="GLM-5.3-Flash、Qwen3.6-35B-A3B、Qwen3.8-27B 在 8×A800 上的性能矩阵与质量成绩，附图表和修正记录。", extra="") + body)
print("docs/benchmark.html written")

if LAB_SRC:
    src = open(LAB_SRC, encoding="utf-8").read()
    src = re.sub(r"^<title>.*?</title>\s*", "", src, count=1, flags=re.S)
    src = re.sub(r'<link rel="stylesheet" href="https://fonts.googleapis.com[^>]*>\s*', "", src, count=1)
    style = re.search(r"<style>(.*?)</style>", src, re.S).group(1)
    rest = src[src.index("</style>") + len("</style>"):]
    rest = rest.replace('<div class="wrap">', '<div class="wrap">\n' + NAV.format(l=' class="on"', b=""), 1)
    rest = rest.replace("</div>\n\n<script src=", FOOT + "\n</div>\n\n<script src=", 1)
    page = HEAD.format(title="原理实验台 · LLM Inference Lab", desc="六个可交互实验：一次生成的全景、prefill 与 decode、KV cache 分页 3D 视图、多卡并行、训练与推理的内存账、指标对号入座。", extra="<style>" + style + "</style>\n") + rest.replace("</body>", "").replace("</html>", "") + "\n</body>\n</html>\n"
    open(os.path.join(ROOT, "docs", "lab.html"), "w", encoding="utf-8").write(page)
    print("docs/lab.html written")
