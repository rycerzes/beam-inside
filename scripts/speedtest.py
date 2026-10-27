# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx", "huggingface_hub[hf_xet]"]
# ///
"""Measure how fast a machine can pull Beam's FP8 weights from the Hub, to price a layer-streaming pass.

    hf jobs uv run --flavor cpu-basic --secrets HF_TOKEN --timeout 15m scripts/speedtest.py

Two paths: HTTP range requests on single tensors (what ckpt.Checkpoint does) at several worker counts, and
hf_hub_download of whole shards (xet). Each run is capped at about 60 s; bytes are discarded.
"""

import concurrent.futures as cf
import os
import shutil
import tempfile
import time

import httpx
from huggingface_hub import HfApi, get_token, hf_hub_download

REPO = "reflection/Beam-1.0-501B-A23B-FP8"
BASE = f"https://huggingface.co/{REPO}/resolve/main/"
token = get_token()
client = httpx.Client(follow_redirects=True, timeout=300, headers={"Authorization": f"Bearer {token}"} if token else {},
                      limits=httpx.Limits(max_connections=64, max_keepalive_connections=64))

files = [s for s in HfApi().model_info(REPO, files_metadata=True, token=token).siblings if s.rfilename.endswith(".safetensors")]
total = sum(s.size for s in files)
print(f"{len(files)} shards, {total / 1e9:.0f} GB, largest {max(s.size for s in files) / 1e9:.1f} GB", flush=True)
# Range path: 16 MB chunks (the size of an expert's FP8 weights), walking through the shards in order.
CHUNK = 16 << 20
chunks = iter([(s.rfilename, off) for s in files[2:] for off in range(0, s.size - CHUNK, CHUNK)])


def get(job):
    shard, off = job
    r = client.get(BASE + shard, headers={"Range": f"bytes={off}-{off + CHUNK - 1}"})
    r.raise_for_status()
    return len(r.content)


for workers in (8, 32, 64):
    t0, n = time.time(), 0
    with cf.ThreadPoolExecutor(workers) as pool:
        while time.time() - t0 < 60:
            n += sum(pool.map(get, [next(chunks) for _ in range(workers * 2)]))
    dt = time.time() - t0
    print(f"range  workers {workers:2d}: {n / 1e9:.1f} GB in {dt:.0f}s = {n / 1e6 / dt:.0f} MB/s", flush=True)

# Whole-shard path via hf_xet, two shards concurrently.
tmp = tempfile.mkdtemp()
names = [s.rfilename for s in files[:2]]
t0 = time.time()
with cf.ThreadPoolExecutor(2) as pool:
    paths = list(pool.map(lambda f: hf_hub_download(REPO, f, cache_dir=tmp, token=token), names))
dt = time.time() - t0
n = sum(os.path.getsize(p) for p in paths)
print(f"xet    2 shards: {n / 1e9:.1f} GB in {dt:.0f}s = {n / 1e6 / dt:.0f} MB/s", flush=True)
shutil.rmtree(tmp)
print(f"full FP8 pass at best rate: see above; {total / 1e9:.0f} GB total", flush=True)
