#!/usr/bin/env python3
"""Minimal OpenAI-compatible chat server around nanochat's Engine, so the same benchmark scripts
(bench/quality.py, vllm bench serve) can be pointed at a model trained here.
Single GPU, requests are served one at a time (a lock serialises generation).

python -m train.nanochat_server --source sft --model-tag d24 --port 8020
"""
import argparse, json, os, sys, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="sft", help="base|sft|rl")
ap.add_argument("--model-tag", default=None); ap.add_argument("--step", type=int, default=None)
ap.add_argument("--port", type=int, default=8020); ap.add_argument("--served-name", default=None)
ap.add_argument("--default-max-tokens", type=int, default=512); ap.add_argument("--top-k", type=int, default=50)
args = ap.parse_args()

from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
device_type = autodetect_device_type()
_, _, _, _, device = compute_init(device_type)
model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
engine = Engine(model, tokenizer)
NAME = args.served_name or f"nanochat-{args.model_tag or 'model'}-{args.source}"
SEQ_LEN = model.config.sequence_len
bos = tokenizer.get_bos_token_id()
sp = {t: tokenizer.encode_special(t) for t in ["<|user_start|>", "<|user_end|>", "<|assistant_start|>", "<|assistant_end|>",
                                                 "<|python_start|>", "<|python_end|>", "<|output_start|>", "<|output_end|>"]}
special_ids = set(sp.values()) | {bos}
lock = threading.Lock()

def render(messages):
    """Render OpenAI-style messages into nanochat tokens, ending with <|assistant_start|>."""
    ids = [bos]
    system = "\n".join(m["content"] for m in messages if m["role"] == "system" and isinstance(m.get("content"), str))
    first_user = True
    for m in messages:
        role = m["role"]
        content = m.get("content")
        if isinstance(content, list):  # multimodal payloads: keep the text parts only
            content = "\n".join(p.get("text", "") for p in content if p.get("type") == "text")
        content = content or ""
        if role == "system": continue
        if role == "user":
            if system and first_user: content = system + "\n\n" + content
            first_user = False
            ids += [sp["<|user_start|>"]] + tokenizer.encode(content) + [sp["<|user_end|>"]]
        elif role == "assistant":
            ids += [sp["<|assistant_start|>"]] + tokenizer.encode(content) + [sp["<|assistant_end|>"]]
    ids.append(sp["<|assistant_start|>"])
    return ids

PIECE_MAP = [("<|python_start|>", " [calc: "), ("<|python_end|>", "]"), ("<|output_start|>", " = "), ("<|output_end|>", " ")]
def stream_tokens(messages, max_tokens, temperature, ignore_eos=False):
    """Generator: yields decoded text pieces; returns (prompt_tokens, completion_tokens, finish_reason) via StopIteration value."""
    ids = render(messages)
    if len(ids) >= SEQ_LEN - 8:
        raise ValueError(f"prompt has {len(ids)} tokens, model context is {SEQ_LEN}")
    max_tokens = max(1, min(max_tokens, SEQ_LEN - len(ids) - 1))
    n, finish = 0, "length"
    with lock:
        for token_column, _ in engine.generate(ids, num_samples=1, max_tokens=max_tokens, temperature=temperature, top_k=args.top_k, seed=int(time.time() * 1000) % 2**31):
            tok = token_column[0]
            if (tok == sp["<|assistant_end|>"] or tok == bos) and not ignore_eos:
                finish = "stop"; break
            n += 1
            if tok in special_ids:
                piece = dict(PIECE_MAP).get(tokenizer.decode([tok]), "") if tok in (sp["<|python_start|>"], sp["<|python_end|>"], sp["<|output_start|>"], sp["<|output_end|>"]) else ""
            else:
                piece = tokenizer.decode([tok])
            if piece: yield piece
    return len(ids), n, finish

def complete(messages, max_tokens, temperature, ignore_eos=False):
    pieces = []; gen = stream_tokens(messages, max_tokens, temperature, ignore_eos)
    while True:
        try: pieces.append(next(gen))
        except StopIteration as e:
            p, c, finish = e.value; break
    return "".join(pieces), p, c, finish

class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # needed for chunked SSE streaming
    def log_message(self, *a): pass
    def _send(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        if self.path.startswith("/v1/models"):
            return self._send(200, {"object": "list", "data": [{"id": NAME, "object": "model", "created": int(time.time()), "owned_by": "nanochat", "max_model_len": SEQ_LEN}]})
        if self.path in ("/health", "/"): return self._send(200, {"status": "ok", "model": NAME})
        self._send(404, {"error": "not found"})
    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0); req = json.loads(self.rfile.read(n) or b"{}")
        if self.path.startswith("/v1/chat/completions"):
            mt = int(req.get("max_tokens") or req.get("max_completion_tokens") or args.default_max_tokens); temp = float(req.get("temperature", 1.0)); ieos = bool(req.get("ignore_eos", False))
            if req.get("stream"):
                try:
                    gen = stream_tokens(req.get("messages", []), mt, temp, ieos)
                except ValueError as e:
                    return self._send(400, {"error": {"message": str(e), "type": "invalid_request_error"}})
                cid = "chatcmpl-" + uuid.uuid4().hex[:12]; t0 = int(time.time())
                self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.send_header("Cache-Control", "no-cache"); self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
                def chunk(obj):
                    data = ("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode()
                    self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n"); self.wfile.flush()
                try:
                    chunk({"id": cid, "object": "chat.completion.chunk", "created": t0, "model": NAME, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]})
                    while True:
                        try: piece = next(gen)
                        except StopIteration as e:
                            p, c, finish = e.value; break
                        chunk({"id": cid, "object": "chat.completion.chunk", "created": t0, "model": NAME, "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]})
                    chunk({"id": cid, "object": "chat.completion.chunk", "created": t0, "model": NAME, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}], "usage": {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}})
                    done = b"data: [DONE]\n\n"
                    self.wfile.write(f"{len(done):x}\r\n".encode() + done + b"\r\n0\r\n\r\n"); self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            try:
                t0 = time.time()
                text, p, c, finish = complete(req.get("messages", []), mt, temp, ieos)
                return self._send(200, {"id": "chatcmpl-" + uuid.uuid4().hex[:12], "object": "chat.completion", "created": int(t0), "model": NAME,
                                        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}],
                                        "usage": {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}})
            except ValueError as e:
                return self._send(400, {"error": {"message": str(e), "type": "invalid_request_error"}})
            except Exception as e:
                return self._send(500, {"error": {"message": repr(e)[:300], "type": "server_error"}})
        if self.path.startswith("/v1/completions"):
            prompt = req.get("prompt", ""); prompt = prompt[0] if isinstance(prompt, list) else prompt
            ids = [bos] + tokenizer.encode(prompt); out = []
            with lock:
                for col, _ in engine.generate(ids, num_samples=1, max_tokens=int(req.get("max_tokens") or 256), temperature=float(req.get("temperature", 1.0)), top_k=args.top_k):
                    out.append(col[0])
            return self._send(200, {"id": "cmpl-" + uuid.uuid4().hex[:12], "object": "text_completion", "model": NAME, "choices": [{"index": 0, "text": tokenizer.decode(out), "finish_reason": "length"}],
                                    "usage": {"prompt_tokens": len(ids), "completion_tokens": len(out), "total_tokens": len(ids) + len(out)}})
        self._send(404, {"error": "not found"})

print(f"serving {NAME} (context {SEQ_LEN}) on http://0.0.0.0:{args.port}/v1", flush=True)
ThreadingHTTPServer(("0.0.0.0", args.port), H).serve_forever()
