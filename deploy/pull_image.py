#!/usr/bin/env python3
"""Pull a Docker image from a registry mirror without docker, and extract it to a rootfs dir.

Usage: pull_image.py <repo> <tag> <layers_dir> <rootfs_dir> [mirror1,mirror2,...]
"""
import json, os, re, subprocess, sys, tarfile, time, urllib.error, urllib.request

repo, tag, layers_dir, rootfs_dir = sys.argv[1:5]
mirrors = (sys.argv[5] if len(sys.argv) > 5 else "docker.1ms.run,dockerproxy.net,hub.rat.dev").split(",")
os.makedirs(layers_dir, exist_ok=True)
os.makedirs(rootfs_dir, exist_ok=True)
ACCEPT = ("application/vnd.docker.distribution.manifest.list.v2+json, "
          "application/vnd.oci.image.index.v1+json, "
          "application/vnd.docker.distribution.manifest.v2+json, "
          "application/vnd.oci.image.manifest.v1+json")


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def get_token(mirror):
    try:
        urllib.request.urlopen(f"https://{mirror}/v2/", timeout=20)
        return None
    except urllib.error.HTTPError as e:
        www = e.headers.get("WWW-Authenticate", "")
        m = re.search(r'realm="([^"]+)"', www)
        s = re.search(r'service="([^"]+)"', www)
        if not m:
            return None
        url = f"{m.group(1)}?service={s.group(1) if s else ''}&scope=repository:{repo}:pull"
        return json.load(urllib.request.urlopen(url, timeout=30)).get("token")


def req(mirror, path, token, accept=None):
    h = {"Accept": accept or ACCEPT}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return urllib.request.Request(f"https://{mirror}{path}", headers=h)


def fetch_json(mirror, path, token, accept=None):
    with urllib.request.urlopen(req(mirror, path, token, accept), timeout=60) as r:
        return json.load(r), r.headers.get("Content-Type", "")


def download_blob(mirror, token, digest, out, expected):
    tmp = out + ".part"
    done = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    if os.path.exists(out) and os.path.getsize(out) == expected:
        return 0.0
    r = req(mirror, f"/v2/{repo}/blobs/{digest}", token)
    if done:
        r.add_header("Range", f"bytes={done}-")
    t0 = time.time()
    with urllib.request.urlopen(r, timeout=120) as resp, open(tmp, "ab" if done else "wb") as f:
        while True:
            chunk = resp.read(4 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
    if os.path.getsize(tmp) != expected:
        raise RuntimeError(f"size mismatch {os.path.getsize(tmp)} != {expected}")
    os.rename(tmp, out)
    return (time.time() - t0)


# ---- 1. resolve manifest -------------------------------------------------
manifest = None
for mirror in mirrors:
    try:
        token = get_token(mirror)
        m, ct = fetch_json(mirror, f"/v2/{repo}/manifests/{tag}", token)
        if "manifests" in m:  # index / manifest list
            amd = [x for x in m["manifests"] if (x.get("platform") or {}).get("architecture") == "amd64"
                   and (x.get("platform") or {}).get("os") == "linux"]
            amd = [x for x in amd if not (x.get("platform") or {}).get("variant") or True]
            digest = amd[0]["digest"]
            m, ct = fetch_json(mirror, f"/v2/{repo}/manifests/{digest}", token)
        manifest = m
        log(f"manifest from {mirror}: {len(m['layers'])} layers, "
            f"{sum(l['size'] for l in m['layers'])/1e9:.2f} GB compressed")
        break
    except Exception as ex:
        log(f"{mirror}: manifest failed: {type(ex).__name__}: {ex}")
if manifest is None:
    sys.exit("no mirror could serve the manifest")
with open(os.path.join(layers_dir, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=1)

# ---- 2. download config + layers ---------------------------------------
layers = manifest["layers"]
cfg = manifest["config"]
targets = [("config.json", cfg["digest"], cfg["size"], cfg.get("mediaType", ""))]
for i, l in enumerate(layers):
    ext = ".tar.zst" if "zstd" in l.get("mediaType", "") else ".tar.gz"
    targets.append((f"layer_{i:02d}{ext}", l["digest"], l["size"], l.get("mediaType", "")))
for name, digest, size, mt in targets:
    out = os.path.join(layers_dir, name)
    for attempt in range(6):
        mirror = mirrors[attempt % len(mirrors)]
        try:
            token = get_token(mirror)
            dt = download_blob(mirror, token, digest, out, size)
            log(f"{name}: {size/1e6:.0f} MB ok via {mirror}" + (f" ({size/1e6/dt:.1f} MB/s)" if dt else " (cached)"))
            break
        except Exception as ex:
            log(f"{name}: attempt {attempt} via {mirror} failed: {type(ex).__name__}: {str(ex)[:100]}")
            time.sleep(5)
    else:
        sys.exit(f"giving up on {name}")

if os.environ.get('SKIP_EXTRACT'):
    log('DOWNLOAD_DONE'); sys.exit(0)
# ---- 3. extract layers in order, honoring whiteouts --------------------
for i, l in enumerate(layers):
    ext = ".tar.zst" if "zstd" in l.get("mediaType", "") else ".tar.gz"
    path = os.path.join(layers_dir, f"layer_{i:02d}{ext}")
    log(f"extracting layer {i:02d} ({l['size']/1e6:.0f} MB)")
    # whiteouts first
    mode = "r:*" if ext == ".tar.gz" else "r|*"
    opener = None
    if ext == ".tar.zst":
        p = subprocess.Popen(["zstd", "-dc", path], stdout=subprocess.PIPE)
        tf = tarfile.open(fileobj=p.stdout, mode="r|")
    else:
        tf = tarfile.open(path, "r:gz")
    wh = []
    for mem in tf:
        base = os.path.basename(mem.name)
        if base.startswith(".wh."):
            wh.append(mem.name)
    tf.close()
    for w in wh:
        d = os.path.dirname(w)
        base = os.path.basename(w)
        if base == ".wh..wh..opq":
            tgt = os.path.join(rootfs_dir, d)
            if os.path.isdir(tgt):
                for e in os.listdir(tgt):
                    subprocess.run(["rm", "-rf", os.path.join(tgt, e)])
        else:
            subprocess.run(["rm", "-rf", os.path.join(rootfs_dir, d, base[4:])])
    if ext == ".tar.zst":
        cmd = f"zstd -dc '{path}' | tar -x --overwrite --exclude='.wh.*' -C '{rootfs_dir}'"
    else:
        cmd = f"tar -x --overwrite --exclude='.wh.*' -C '{rootfs_dir}' -f '{path}'"
    rc = subprocess.run(["bash", "-c", cmd]).returncode
    if rc != 0:
        log(f"WARNING: tar exit {rc} on layer {i:02d}")
with open(os.path.join(layers_dir, "config.json")) as f:
    c = json.load(f)
with open(os.path.join(rootfs_dir, ".image_config.json"), "w") as f:
    json.dump(c, f, indent=1)
log("ENV:", c.get("config", {}).get("Env"))
log("ENTRYPOINT:", c.get("config", {}).get("Entrypoint"), "CMD:", c.get("config", {}).get("Cmd"))
log("EXTRACT_DONE")
