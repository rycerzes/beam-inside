"""Read individual tensors from a safetensors checkpoint, on the Hub (HTTP range requests) or on local disk."""

import json
import pathlib
import struct
import time

import httpx
from huggingface_hub import get_token


class _Shards:
    """Tensor lookup over sharded safetensors; subclasses supply _read(shard, start, end) -> bytes."""

    def header(self, shard):
        if shard not in self._headers:
            n = struct.unpack("<Q", self._read(shard, 0, 8))[0]
            self._headers[shard] = (8 + n, json.loads(self._read(shard, 8, 8 + n)))
        return self._headers[shard]

    def info(self, name):
        """(dtype, shape, absolute byte range) of one tensor."""
        offset, meta = self.header(self.weight_map[name])
        start, end = meta[name]["data_offsets"]
        return meta[name]["dtype"], meta[name]["shape"], (offset + start, offset + end)

    def raw(self, name):
        dtype, shape, (start, end) = self.info(name)
        data = self._read(self.weight_map[name], start, end)
        assert len(data) == end - start, (name, len(data), end - start)
        return dtype, shape, data


class Checkpoint(_Shards):
    def __init__(self, repo):
        self.base = f"https://huggingface.co/{repo}/resolve/main/"
        token = get_token()
        self.client = httpx.Client(follow_redirects=True, timeout=300,
                                   headers={"Authorization": f"Bearer {token}"} if token else {},
                                   limits=httpx.Limits(max_connections=64, max_keepalive_connections=64))
        self.weight_map = self.client.get(self.base + "model.safetensors.index.json").raise_for_status().json()["weight_map"]
        self._headers = {}

    def _read(self, shard, start, end, tries=5):
        for attempt in range(tries):  # long streams hit the odd dropped connection; retry those
            try:
                return self.client.get(self.base + shard, headers={"Range": f"bytes={start}-{end - 1}"}).raise_for_status().content
            except (httpx.TransportError, httpx.HTTPStatusError):
                if attempt == tries - 1:
                    raise
                time.sleep(2 ** attempt)


class LocalCheckpoint(_Shards):
    def __init__(self, path):
        self.path = pathlib.Path(path)
        self.weight_map = json.loads((self.path / "model.safetensors.index.json").read_text())["weight_map"]
        self._headers = {}

    def _read(self, shard, start, end):
        with open(self.path / shard, "rb") as f:
            f.seek(start)
            return f.read(end - start)
