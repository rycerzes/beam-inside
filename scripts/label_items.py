# /// script
# requires-python = ">=3.11"
# ///
"""Swap judge items' token labels for Beam's pole labels (self_label.py on pole_rows.py rows).

    uv run scripts/label_items.py ITEMS.jsonl KEY.jsonl LABELS.jsonl OUT.jsonl
    -> the items whose pole Beam labelled, with label "<name>: <description>", for judge_router.py
       --label-kind description
"""

import argparse
import json
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument("items", type=pathlib.Path)
parser.add_argument("key", type=pathlib.Path)
parser.add_argument("labels", type=pathlib.Path)
parser.add_argument("out", type=pathlib.Path)
args = parser.parse_args()

key = {k["id"]: f"L{k['layer']}_E{k['expert']}_{k['pole']}" for k in map(json.loads, open(args.key))}
labels = {r["id"]: r for r in map(json.loads, open(args.labels)) if r["name"]}
n = 0
with open(args.out, "w") as f:
    for item in map(json.loads, open(args.items)):
        lab = labels.get(key[item["id"]])
        if lab:
            f.write(json.dumps({**item, "label": f"{lab['name']}: {lab['description']}"}, ensure_ascii=False) + "\n")
            n += 1
print(f"{n} items with Beam labels -> {args.out}")
