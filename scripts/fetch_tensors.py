# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx", "numpy", "huggingface_hub"]
# ///
"""Fetch selected tensors from a Hub safetensors checkpoint with HTTP range requests, without downloading shards.

    uv run scripts/fetch_tensors.py --repo reflection/Beam-1.0-501B-A23B --out data/bf16 'mlp\\.gate\\.' 'norm'
    uv run scripts/fetch_tensors.py --repo ... --list 'layers\\.3\\.'   # print name, dtype, shape only

Each tensor is saved as data/<out>/<name>.npy with its raw dtype (BF16 is stored as uint16; see load_bf16 in atlas.py).
"""

import argparse
import json
import pathlib
import re
import struct

import httpx
import numpy as np
from huggingface_hub import get_token

parser = argparse.ArgumentParser()
parser.add_argument("--repo", default="reflection/Beam-1.0-501B-A23B")
parser.add_argument("--out", default="data/bf16")
parser.add_argument("--list", action="store_true")
parser.add_argument("patterns", nargs="+")
args = parser.parse_args()

base = f"https://huggingface.co/{args.repo}/resolve/main/"
token = get_token()
client = httpx.Client(follow_redirects=True, timeout=120,
                      headers={"Authorization": f"Bearer {token}"} if token else {})
index = client.get(base + "model.safetensors.index.json").raise_for_status().json()
wanted = sorted(n for n in index["weight_map"] if any(re.search(p, n) for p in args.patterns))
print(f"{len(wanted)} tensors match")

RAW = {"BF16": np.uint16, "F32": np.float32, "F16": np.float16, "F8_E4M3": np.uint8, "U8": np.uint8, "I32": np.int32}


def header(shard, cache={}):
    if shard not in cache:
        n = struct.unpack("<Q", client.get(base + shard, headers={"Range": "bytes=0-7"}).content)[0]
        meta = json.loads(client.get(base + shard, headers={"Range": f"bytes=8-{7 + n}"}).content)
        cache[shard] = (8 + n, meta)
    return cache[shard]


out = pathlib.Path(args.out)
out.mkdir(parents=True, exist_ok=True)
total = 0
for name in wanted:
    offset, meta = header(index["weight_map"][name])
    info = meta[name]
    start, end = info["data_offsets"]
    total += end - start
    if args.list:
        print(name, info["dtype"], info["shape"])
        continue
    path = out / f"{name}.npy"
    if path.exists():
        continue
    data = client.get(base + index["weight_map"][name],
                      headers={"Range": f"bytes={offset + start}-{offset + end - 1}"}).raise_for_status().content
    assert len(data) == end - start, (name, len(data), end - start)
    np.save(path, np.frombuffer(data, dtype=RAW[info["dtype"]]).reshape(info["shape"]))
    print(f"{name} {info['dtype']} {info['shape']}")
print(f"{total / 1e9:.2f} GB")
