"""
Greedy answers of the SFT and RL checkpoints of one nanochat model on a few held-out GSM8K test problems
(the same shuffled test split the evals use) plus one bare arithmetic question, with correctness, answer
length and number of calculator calls, so the training log can show what RL changed.

Deployed as scripts/math_samples.py inside the nanochat checkout:
    python -m scripts.math_samples -g d24 --out samples_math.json
"""
import argparse, gc, json, re

import torch
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
from tasks.gsm8k import GSM8K, extract_answer

ap = argparse.ArgumentParser()
ap.add_argument("-g", "--model-tag", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--sources", default="sft,rl")
ap.add_argument("-n", "--num", type=int, default=4, help="number of GSM8K test problems")
args = ap.parse_args()

_, _, _, _, device = compute_init(autodetect_device_type())
task = GSM8K(subset="main", split="test")
items = []
for i in range(args.num):
    conv = task[i]
    ref = extract_answer(conv["messages"][-1]["content"][-1]["text"])
    items.append({"q": conv["messages"][0]["content"], "ref": ref, "kind": "gsm8k", "conv": conv})
items.append({"q": "What is 12 times 7?", "ref": "84", "kind": "arith", "conv": None})

NUM = re.compile(r"-?\d[\d,]*\.?\d*")
def last_number(text):
    nums = NUM.findall(text.replace("<|output_end|>", " "))
    return nums[-1].replace(",", "").rstrip(".") if nums else None

for src in args.sources.split(","):
    model, tok, meta = load_model(src, device, phase="eval", model_tag=args.model_tag)
    eng = Engine(model, tok)
    bos = tok.get_bos_token_id()
    a_end = tok.encode_special("<|assistant_end|>")
    for it in items:
        if it["conv"] is not None:
            ids = tok.render_for_completion(it["conv"])
        else:
            ids = [bos, tok.encode_special("<|user_start|>")] + tok.encode(it["q"]) + \
                  [tok.encode_special("<|user_end|>"), tok.encode_special("<|assistant_start|>")]
        out = []
        for col, _ in eng.generate(ids, num_samples=1, max_tokens=256, temperature=0.0):
            t = col[0]
            if t in (a_end, bos):
                break
            out.append(t)
        text = tok.decode(out)
        strict = extract_answer(text)
        loose = last_number(text)
        ok = (strict == it["ref"]) if it["kind"] == "gsm8k" else (loose == it["ref"])
        it[src] = {"text": text, "pred": strict if it["kind"] == "gsm8k" else loose, "correct": bool(ok),
                   "tokens": len(out), "tool_calls": text.count("<|python_start|>")}
    del eng, model
    gc.collect()
    torch.cuda.empty_cache()

for it in items:
    it.pop("conv")
json.dump({"tag": args.model_tag, "items": items}, open(args.out, "w"), ensure_ascii=False, indent=1)
print("wrote", args.out, "|", [(it["ref"], {s: it[s]["pred"] for s in args.sources.split(",")}) for it in items])
