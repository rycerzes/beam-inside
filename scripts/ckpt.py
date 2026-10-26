"""Read individual tensors from a Hub safetensors checkpoint with HTTP range requests."""

import json
import struct

import httpx
from huggingface_hub import get_token


class Checkpoint:
    def __init__(self, repo):
        self.base = f"https://huggingface.co/{repo}/resolve/main/"
        token = get_token()
        self.client = httpx.Client(follow_redirects=True, timeout=300,
                                   headers={"Authorization": f"Bearer {token}"} if token else {})
        self.weight_map = self.client.get(self.base + "model.safetensors.index.json").raise_for_status().json()["weight_map"]
        self._headers = {}

    def header(self, shard):
        if shard not in self._headers:
            n = struct.unpack("<Q", self.client.get(self.base + shard, headers={"Range": "bytes=0-7"}).content)[0]
            meta = json.loads(self.client.get(self.base + shard, headers={"Range": f"bytes=8-{7 + n}"}).content)
            self._headers[shard] = (8 + n, meta)
        return self._headers[shard]

    def info(self, name):
        """(dtype, shape, absolute byte range) of one tensor."""
        shard = self.weight_map[name]
        offset, meta = self.header(shard)
        start, end = meta[name]["data_offsets"]
        return meta[name]["dtype"], meta[name]["shape"], (offset + start, offset + end)

    def raw(self, name):
        dtype, shape, (start, end) = self.info(name)
        data = self.client.get(self.base + self.weight_map[name],
                               headers={"Range": f"bytes={start}-{end - 1}"}).raise_for_status().content
        assert len(data) == end - start, (name, len(data), end - start)
        return dtype, shape, data
