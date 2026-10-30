# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pyarrow", "umap-learn==0.5.12"]
# ///
"""A 2D map of Beam's 12,544 routed experts, placed by what they write, for the explorer.

    uv run scripts/expert_map.py OUT.json [--units data/sweep/units.npz] [--atlas data/export/atlas]

Each expert is its sweep direction (the top write direction of its down-projection, signed to its + pole), embedded
with UMAP (cosine, 20 neighbours) into 2D; coordinates are scaled to [0, 1]. Directions of every layer share the
residual space, so experts that write similar things land together whatever their depth. Regions are labelled by
k-means on the map (48 clusters): each takes the most common subject among its coherent Beam labels and the most
frequent words in those labels. Also reported: how often a point's 10 nearest map neighbours share its Beam subject,
against shuffled subjects.
"""

import argparse
import collections
import json
import pathlib
import re

import numpy as np
import pyarrow.parquet as pq
import umap

parser = argparse.ArgumentParser()
parser.add_argument("out", type=pathlib.Path)
parser.add_argument("--units", type=pathlib.Path, default=pathlib.Path("data/sweep/units.npz"))
parser.add_argument("--meta", type=pathlib.Path, default=pathlib.Path("data/sweep/units.jsonl"))
parser.add_argument("--atlas", type=pathlib.Path, default=pathlib.Path("data/export/atlas"))
parser.add_argument("--clusters", type=int, default=48)
args = parser.parse_args()
rng = np.random.default_rng(0)

meta = [json.loads(line) for line in open(args.meta)]
ids = [m["unit"] for m in meta if m["kind"] == "expert_dir"]
X = np.load(args.units)["dirs"][ids].astype(np.float32)
layer = np.array([meta[u]["layer"] for u in ids])
expert = np.array([meta[u]["expert"] for u in ids])
print(f"{len(ids)} experts, embedding {X.shape}")
Y = umap.UMAP(n_neighbors=20, min_dist=0.08, metric="cosine", random_state=0).fit_transform(X)
Y = (Y - Y.min(0)) / (Y.max(0) - Y.min(0))

poles = {(p["layer"], p["expert"]): p for p in pq.read_table(args.atlas / "experts/poles.parquet").to_pylist()
         if p["pole"] == "+"}
subj = np.array([(poles.get((L, e)) or {}).get("beam_subject") or "" for L, e in zip(layer, expert)])
coh = np.array([bool((poles.get((L, e)) or {}).get("beam_coherent")) for L, e in zip(layer, expert)])
names = [(poles.get((L, e)) or {}).get("beam_name") or "" for L, e in zip(layer, expert)]

topical = coh & ~np.isin(subj, ["", "none", "general"])
T = Y[topical]
d = ((T[:, None, :] - T[None, :, :]) ** 2).sum(-1)
np.fill_diagonal(d, np.inf)
nn = np.argsort(d, 1)[:, :10]
s = subj[topical]
sh = s.copy()
rng.shuffle(sh)
print(f"map 10-NN same subject {np.mean(s[nn] == s[:, None]):.3f} vs shuffled {np.mean(sh[nn] == sh[:, None]):.3f}")

# k-means on the map for region labels
C = Y[rng.choice(len(Y), args.clusters, replace=False)]
for _ in range(60):
    a = ((Y[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
    C = np.array([Y[a == k].mean(0) if (a == k).any() else C[k] for k in range(args.clusters)])
STOP = set("and the of in or to for with a an as by on from tokens token words word related terms term forms form "
           "concept concepts various multiple".split())
clusters = []
for k in range(args.clusters):
    m = a == k
    subs = collections.Counter(x for x, c in zip(subj[m], coh[m]) if c and x not in ("", "none"))
    words = collections.Counter(w for i in np.flatnonzero(m & coh) for w in set(re.findall(r"[a-z]{3,}", names[i].lower()))
                                if w not in STOP)
    top = subs.most_common(1)
    clusters.append({"x": round(float(C[k, 0]), 4), "y": round(float(C[k, 1]), 4), "n": int(m.sum()),
                     "subject": top[0][0] if top else "", "words": [w for w, _ in words.most_common(3)],
                     "layers": [int(np.percentile(layer[m], 10)), int(np.percentile(layer[m], 90))]})

points = [[int(L), int(e), round(float(x), 4), round(float(y), 4)] for L, e, (x, y) in zip(layer, expert, Y)]
args.out.write_text(json.dumps({"points": points, "clusters": clusters}, separators=(",", ":")))
print(f"-> {args.out} ({args.out.stat().st_size / 1e3:.0f} kB); clusters:",
      "; ".join(f"{c['subject']}:{'/'.join(c['words'])}" for c in clusters[:12]))
