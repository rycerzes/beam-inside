# /// script
# requires-python = ">=3.11"
# ///
"""List every testable expert pole of a ranked v2 sweep with its token label, for self_label.py.

    uv run scripts/pole_rows.py RUN_DIR KEY.jsonl OUT_DIR
    -> OUT_DIR/poles.jsonl ({"id": "L{layer}_E{expert}_{pole}", "layer", "expert", "pole", "label"})
       OUT_DIR/ids.txt (pole ids, judge-set poles from KEY first, in KEY order, then the rest by layer and expert)

The label is the one judge_items.py shows a judge (first 12 lens tokens, decoded), so a pole's Beam label can stand
in for its token label in the three-way test.
"""

import argparse
import json
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument("run", type=pathlib.Path)
parser.add_argument("key", type=pathlib.Path)
parser.add_argument("out", type=pathlib.Path)
args = parser.parse_args()

src = (pathlib.Path(__file__).parent / "judge_items.py").read_text()  # the judge's own label code
helpers = {}
exec(src[src.index("def _byte_decoder"):src.index("def stems")], helpers)
label = helpers["label"]

rows = {}
for r in map(json.loads, open(args.run / "ranked_matched.jsonl")):
    if r["kind"] == "expert_dir" and r.get("null") != "untestable":
        pid = f"L{r['layer']}_E{r['expert']}_{r['pole']}"
        rows[pid] = {"id": pid, "layer": r["layer"], "expert": r["expert"], "pole": r["pole"], "label": label(r["lens"])}
judged = [f"L{k['layer']}_E{k['expert']}_{k['pole']}" for k in map(json.loads, open(args.key))]
missing = [p for p in judged if p not in rows]
assert not missing, f"{len(missing)} judge-set poles are not testable rows, e.g. {missing[:3]}"
order = list(dict.fromkeys(judged)) + sorted(set(rows) - set(judged), key=lambda p: (rows[p]["layer"],
                                                                                      rows[p]["expert"], p))
args.out.mkdir(parents=True, exist_ok=True)
with open(args.out / "poles.jsonl", "w") as f:
    for p in order:
        f.write(json.dumps(rows[p], ensure_ascii=False) + "\n")
(args.out / "ids.txt").write_text("\n".join(order) + "\n")
print(f"{len(order)} poles ({len(set(judged))} judge-set first) -> {args.out}")
