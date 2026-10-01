#!/usr/bin/env python3
"""Quality benchmark for an OpenAI-compatible endpoint.

Usage: quality.py --base-url http://localhost:8010/v1 --model glm-5.3-flash --out results/glm.json
        [--gsm8k N] [--humaneval N] [--ceval N] [--needle LENS] [--vision] [--concurrency C] [--think auto|on|off]

Tests (all auto-graded):
  gsm8k      exact match of the final number ("#### 42" reference)
  humaneval  pass@1 by executing the generated function against the hidden tests (subprocess, timeout)
  ceval      multiple-choice accuracy on C-Eval val (subject-stratified sample)
  needle     passkey retrieval accuracy at several context lengths, plus TTFT per length
  json       JSON-schema compliance rate on 20 structured-output prompts
  tools      function-calling format correctness on 10 prompts (OpenAI tools API)
  vision     synthetic shape/color/count images (only with --vision)
"""
import argparse, base64, concurrent.futures as cf, io, json, math, os, random, re, struct, subprocess, sys, tempfile, time, urllib.request, zlib

ap = argparse.ArgumentParser()
ap.add_argument("--base-url", required=True); ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--data", default="/mnt/pfs/4n3evq/lhx/llm_deploy/bench/data")
ap.add_argument("--gsm8k", type=int, default=200); ap.add_argument("--humaneval", type=int, default=164); ap.add_argument("--ceval", type=int, default=300)
ap.add_argument("--needle", default="4096,16384,32768,65536,131072"); ap.add_argument("--needle-trials", type=int, default=3)
ap.add_argument("--vision", action="store_true"); ap.add_argument("--concurrency", type=int, default=16)
ap.add_argument("--max-tokens", type=int, default=4096); ap.add_argument("--think", default="auto")
ap.add_argument("--only", default="")  # comma list of tests to run
args = ap.parse_args()
random.seed(20260930)
ONLY = set(args.only.split(",")) if args.only else None

# ---------------------------------------------------------------- client
def chat(messages, max_tokens=None, temperature=0.0, tools=None, response_format=None, extra=None, timeout=1800):
    body = {"model": args.model, "messages": messages, "max_tokens": max_tokens or args.max_tokens, "temperature": temperature}
    if tools: body["tools"] = tools; body["tool_choice"] = "auto"
    if response_format: body["response_format"] = response_format
    if args.think == "off": body["chat_template_kwargs"] = {"enable_thinking": False}
    if args.think == "on": body["chat_template_kwargs"] = {"enable_thinking": True}
    if extra: body.update(extra)
    req = urllib.request.Request(args.base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.load(r)
    except Exception as e:
        return {"error": str(e)[:200], "latency": time.time() - t0}
    m = d["choices"][0]["message"]
    return {"content": m.get("content") or "", "reasoning": m.get("reasoning_content") or m.get("reasoning") or "",
            "tool_calls": m.get("tool_calls"), "finish": d["choices"][0].get("finish_reason"),
            "usage": d.get("usage", {}), "latency": time.time() - t0}

def pmap(fn, items, workers=None):
    with cf.ThreadPoolExecutor(max_workers=workers or args.concurrency) as ex:
        return list(ex.map(fn, items))

def load_jsonl(fn, n=None, shuffle=True):
    rows = [json.loads(l) for l in open(os.path.join(args.data, fn), encoding="utf-8")]
    if shuffle: random.shuffle(rows)
    return rows[:n] if n else rows

results = {"model": args.model, "base_url": args.base_url, "started": time.strftime("%F %T"), "tests": {}}
def save():
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(results, open(args.out, "w"), ensure_ascii=False, indent=1)
def want(name): return ONLY is None or name in ONLY
def summarize(name, recs, score_key="correct", extra=None):
    n = len(recs); ok = sum(1 for r in recs if r.get(score_key)); errs = sum(1 for r in recs if r.get("error"))
    toks = [r.get("usage", {}).get("completion_tokens", 0) for r in recs if not r.get("error")]
    lat = [r["latency"] for r in recs if not r.get("error")]
    trunc = sum(1 for r in recs if r.get("finish") == "length")
    s = {"n": n, "correct": ok, "accuracy": ok / n if n else None, "errors": errs, "truncated": trunc,
         "avg_completion_tokens": sum(toks) / len(toks) if toks else None, "avg_latency_s": sum(lat) / len(lat) if lat else None}
    if extra: s.update(extra)
    results["tests"][name] = {"summary": s, "records": recs}
    print(f"[{name}] acc={s['accuracy']:.3f} ({ok}/{n}) errors={errs} truncated={trunc} avg_tokens={s['avg_completion_tokens'] and round(s['avg_completion_tokens'])} avg_latency={s['avg_latency_s'] and round(s['avg_latency_s'],1)}s", flush=True)
    save()

# ---------------------------------------------------------------- gsm8k
NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
def last_number(text):
    m = re.findall(r"####\s*(-?[\d,]*\.?\d+)", text)
    if m: return m[-1].replace(",", "")
    m = re.findall(r"\\boxed\{(-?[\d,]*\.?\d+)\}", text)
    if m: return m[-1].replace(",", "")
    nums = NUM_RE.findall(text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None
def num_eq(a, b):
    try: return a is not None and abs(float(a) - float(b)) < 1e-6
    except Exception: return False
if want("gsm8k") and args.gsm8k and os.path.exists(os.path.join(args.data, "gsm8k_test.jsonl")):
    rows = load_jsonl("gsm8k_test.jsonl", args.gsm8k)
    def run(r):
        ref = r["answer"].split("####")[-1].strip().replace(",", "")
        o = chat([{"role": "user", "content": r["question"] + "\n\nSolve step by step, then give the final numeric answer on the last line in the form: #### <number>"}])
        pred = None if o.get("error") else last_number(o["content"])
        o.update({"ref": ref, "pred": pred, "correct": num_eq(pred, ref)}); o.pop("reasoning", None); o["content"] = (o.get("content") or "")[-300:]
        return o
    summarize("gsm8k", pmap(run, rows))

# ---------------------------------------------------------------- humaneval
def extract_code(text):
    blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
    return (blocks[-1] if blocks else text)
def run_tests(prompt, completion, test, entry_point, timeout=20):
    code = completion if ("def " + entry_point) in completion else prompt + completion
    prog = code + "\n\n" + test + f"\n\ncheck({entry_point})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(prog); fn = f.name
    try:
        p = subprocess.run([sys.executable, fn], capture_output=True, timeout=timeout, env={"PATH": os.environ.get("PATH", "")})
        return p.returncode == 0, (p.stderr.decode(errors="ignore")[-300:] if p.returncode else "")
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        os.unlink(fn)
if want("humaneval") and args.humaneval and os.path.exists(os.path.join(args.data, "humaneval.jsonl")):
    rows = load_jsonl("humaneval.jsonl", args.humaneval, shuffle=False)
    def run(r):
        o = chat([{"role": "user", "content": "Complete the following Python function. Return the complete function (including the signature and any imports) in a single ```python code block.\n\n```python\n" + r["prompt"] + "\n```"}])
        if o.get("error"): o["correct"] = False; return o
        code = extract_code(o["content"]); ok, err = run_tests(r["prompt"], code, r["test"], r["entry_point"])
        o.update({"task_id": r["task_id"], "correct": ok, "err": err}); o.pop("reasoning", None); o["content"] = code[:600]
        return o
    summarize("humaneval", pmap(run, rows))

# ---------------------------------------------------------------- ceval
if want("ceval") and args.ceval and os.path.exists(os.path.join(args.data, "ceval_val.jsonl")):
    allrows = load_jsonl("ceval_val.jsonl", None)
    by = {}
    for r in allrows: by.setdefault(r.get("subject", "?"), []).append(r)
    per = max(1, math.ceil(args.ceval / max(1, len(by)))); rows = []
    for s, rs in sorted(by.items()): rows += rs[:per]
    rows = rows[:args.ceval]
    def run(r):
        q = f"{r['question']}\nA. {r['A']}\nB. {r['B']}\nC. {r['C']}\nD. {r['D']}\n\n这是一道单选题，请先简要分析，最后一行只输出：答案：<字母>"
        o = chat([{"role": "user", "content": q}])
        pred = None
        if not o.get("error"):
            m = re.findall(r"答案\s*[:：]\s*\(?([ABCD])", o["content"]) or re.findall(r"\b([ABCD])\b", o["content"][-40:])
            pred = m[-1] if m else None
        o.update({"subject": r.get("subject"), "ref": r["answer"], "pred": pred, "correct": pred == r["answer"]}); o.pop("reasoning", None); o["content"] = (o.get("content") or "")[-200:]
        return o
    recs = pmap(run, rows)
    subj = {}
    for r in recs: subj.setdefault(r["subject"], [0, 0]); subj[r["subject"]][1] += 1; subj[r["subject"]][0] += bool(r["correct"])
    summarize("ceval", recs, extra={"subjects": len(subj)})

# ---------------------------------------------------------------- needle in a haystack
FILLER = ("The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. ")
if want("needle") and args.needle:
    lens = [int(x) for x in args.needle.split(",") if x]
    recs = []
    for L in lens:
        for t in range(args.needle_trials):
            key = "".join(random.choice("0123456789") for _ in range(7))
            n_fill = int(L * 0.95 / 20)  # ~20 tokens per filler sentence group
            depth = [0.1, 0.5, 0.9][t % 3]
            k = int(n_fill * depth)
            text = FILLER * k + f"\nThe secret passkey is {key}. Remember it.\n" + FILLER * (n_fill - k)
            o = chat([{"role": "user", "content": text + "\n\nWhat is the secret passkey mentioned in the text above? Reply with the digits only."}], max_tokens=256, timeout=3600)
            ok = (not o.get("error")) and key in (o.get("content") or "") + (o.get("reasoning") or "")
            recs.append({"context_tokens_target": L, "depth": depth, "prompt_tokens": o.get("usage", {}).get("prompt_tokens"),
                         "correct": ok, "latency": o.get("latency"), "error": o.get("error"), "usage": o.get("usage", {})})
            print(f"  needle L={L} depth={depth} prompt_tokens={recs[-1]['prompt_tokens']} correct={ok} latency={o.get('latency'):.1f}s", flush=True)
    byL = {}
    for r in recs: byL.setdefault(r["context_tokens_target"], []).append(r)
    per_len = {L: {"accuracy": sum(bool(r["correct"]) for r in rs) / len(rs), "avg_latency_s": sum(r["latency"] or 0 for r in rs) / len(rs),
                   "prompt_tokens": rs[0]["prompt_tokens"]} for L, rs in byL.items()}
    summarize("needle", recs, extra={"per_length": per_len})

# ---------------------------------------------------------------- json schema compliance
if want("json"):
    schema = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}, "city": {"type": "string"},
              "hobbies": {"type": "array", "items": {"type": "string"}}}, "required": ["name", "age", "city", "hobbies"], "additionalProperties": False}
    people = [("Alice", 30, "Beijing", ["reading", "cycling"]), ("Bob", 45, "Shanghai", ["cooking"]), ("张伟", 28, "杭州", ["跑步", "摄影", "围棋"])]
    prompts = []
    for i in range(20):
        n, a, c, h = people[i % 3]
        prompts.append((f"Extract the person's info as JSON with keys name, age, city, hobbies. Text: {n} is {a + i} years old, lives in {c} and enjoys {', '.join(h)}. Output only the JSON object.", {"name": n, "age": a + i, "city": c, "hobbies": h}))
    def run(p):
        o = chat([{"role": "user", "content": p[0]}], max_tokens=args.max_tokens)
        ok = False; parsed = None
        if not o.get("error"):
            m = re.search(r"\{.*\}", o["content"], re.S)
            try:
                parsed = json.loads(m.group(0)) if m else None
                ok = isinstance(parsed, dict) and set(parsed) == set(schema["required"]) and parsed["name"] == p[1]["name"] and parsed["age"] == p[1]["age"] and isinstance(parsed["hobbies"], list)
            except Exception: ok = False
        o.update({"correct": ok, "parsed": parsed}); o.pop("reasoning", None); o["content"] = (o.get("content") or "")[-300:]
        return o
    summarize("json", pmap(run, prompts))

# ---------------------------------------------------------------- tool calling
if want("tools"):
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get current weather for a city",
              "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}},
             {"type": "function", "function": {"name": "convert_currency", "description": "Convert an amount between currencies",
              "parameters": {"type": "object", "properties": {"amount": {"type": "number"}, "from": {"type": "string"}, "to": {"type": "string"}}, "required": ["amount", "from", "to"]}}}]
    cases = [("What's the weather in Tokyo right now?", "get_weather", {"city": "Tokyo"}), ("北京今天天气怎么样？", "get_weather", {"city": "北京"}),
             ("Convert 100 US dollars to euros.", "convert_currency", {"amount": 100, "from": "USD", "to": "EUR"}), ("把 250 人民币换成日元", "convert_currency", {"amount": 250, "from": "CNY", "to": "JPY"}),
             ("Is it hot in Cairo today, in fahrenheit?", "get_weather", {"city": "Cairo", "unit": "fahrenheit"}), ("How much is 75 GBP in USD?", "convert_currency", {"amount": 75, "from": "GBP", "to": "USD"}),
             ("上海现在的天气", "get_weather", {"city": "上海"}), ("换算 1000 欧元到英镑", "convert_currency", {"amount": 1000, "from": "EUR", "to": "GBP"}),
             ("weather in Paris please, celsius", "get_weather", {"city": "Paris", "unit": "celsius"}), ("Write a haiku about the sea.", None, None)]
    def run(c):
        o = chat([{"role": "user", "content": c[0]}], max_tokens=args.max_tokens, tools=tools)
        ok = False; got = None
        if not o.get("error"):
            tc = o.get("tool_calls") or []
            if c[1] is None: ok = not tc and bool(o["content"].strip())
            elif tc:
                try:
                    f = tc[0]["function"]; a = json.loads(f.get("arguments") or "{}"); got = {"name": f["name"], "args": a}
                    ok = f["name"] == c[1] and all(str(a.get(k, "")).lower().rstrip("市") == str(v).lower() for k, v in c[2].items() if k != "amount") and (("amount" not in c[2]) or abs(float(a.get("amount", 0)) - c[2]["amount"]) < 1e-6)
                except Exception: ok = False
        o.update({"expected": c[1], "got": got, "correct": ok}); o.pop("reasoning", None); o["content"] = (o.get("content") or "")[-200:]
        return o
    summarize("tools", pmap(run, cases))

# ---------------------------------------------------------------- vision (synthetic)
def png(w, h, draw):
    raw = b"".join(b"\x00" + bytes(v for x in range(w) for v in draw(x, y)) for y in range(h))
    def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
COLORS = {"red": (220, 30, 30), "green": (30, 170, 60), "blue": (30, 60, 220), "yellow": (240, 220, 40), "black": (0, 0, 0)}
if want("vision") and args.vision:
    cases = []
    for i in range(12):
        n = 1 + i % 4; color = list(COLORS)[i % 5]; shape = ["circle", "square"][i % 2]
        centers = [(60 + 90 * j, 128) for j in range(n)]
        def draw(x, y, centers=centers, color=color, shape=shape):
            for cx, cy in centers:
                if (shape == "circle" and (x - cx) ** 2 + (y - cy) ** 2 < 30 ** 2) or (shape == "square" and abs(x - cx) < 30 and abs(y - cy) < 30):
                    return COLORS[color]
            return (255, 255, 255)
        img = base64.b64encode(png(420, 256, draw)).decode()
        q = ["How many shapes are in this image? Answer with a single number.", "What color are the shapes? Answer with one word.", "Are the shapes circles or squares? Answer with one word."][i % 3]
        ans = [str(n), color, shape + "s"][i % 3]
        cases.append((img, q, ans))
    def run(c):
        o = chat([{"role": "user", "content": [{"type": "text", "text": c[1]}, {"type": "image_url", "image_url": {"url": "data:image/png;base64," + c[0]}}]}], max_tokens=256)
        ok = (not o.get("error")) and c[2].lower().rstrip("s") in o["content"].lower()
        o.update({"expected": c[2], "correct": ok}); o.pop("reasoning", None); o["content"] = (o.get("content") or "")[-200:]
        return o
    summarize("vision", pmap(run, cases))

results["finished"] = time.strftime("%F %T"); save()
print("DONE", args.out)
