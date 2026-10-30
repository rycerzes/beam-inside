"""Compact figure data for the rycerz.es write-up, built from the public atlas export.

    uv run --no-project --with pandas --with pyarrow scripts/blog_data.py ATLAS_DIR SPACE_DATA_DIR OUT_DIR

ATLAS_DIR is the export of export_atlas.py (the rycerzes/beam-expert-atlas layout), SPACE_DATA_DIR the
inside-beam site's public/data (routes.json, expert_map.json, bans.json, recipes.json, baseline.json), OUT_DIR the
website's src/lib/figures/data/inside-beam.
"""

import json
import sys
from pathlib import Path

import pandas as pd

atlas, space, out = (Path(a) for a in sys.argv[1:4])
out.mkdir(parents=True, exist_ok=True)


def dump(name, obj):
    (out / name).write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
    print(name, (out / name).stat().st_size)


def r(x, n=4):
    return None if x is None or x != x else round(float(x), n)


summary = json.loads((atlas / "summary.json").read_text())

# calibration: ΔNLL per layer per dose (random directions)
calib = json.loads((atlas / "calibration/calib.json").read_text())
dump("calib.json", {
    "doses": calib["doses"],
    "tau": r(calib["tau_star"]),
    "dnll": [[r(v) for v in calib["dnll_by_layer"][str(l)]] for l in range(52)],
})

# depth bands
dump("depth.json", [
    {
        "band": band,
        "poles": d["poles"],
        "testable": d["testable"],
        "sig": r(d["significant"]["rate"]),
        "coherent": r(d["beam_coherent"]["rate"]),
        "judge_sig": r(d["judge_sig_token"]["rate"]), "judge_sig_n": d["judge_sig_token"]["n"],
        "judge_rand": r(d["judge_random_token"]["rate"]), "judge_rand_n": d["judge_random_token"]["n"],
    }
    for band, d in summary["depth"].items()
])

# bans: per-subject change vs the matched control, plus banned experts per layer
bans = json.loads((space / "bans.json").read_text())
dump("bans.json", {
    arm: {
        "n_experts": len(v["experts"]),
        "per_layer": {str(l): sum(1 for e in v["experts"] if e[0] == l) for l in range(3, 52)},
        "subjects": {k: {"diff": c["diff"], "p": c["p"], "ban": c["ban"], "ctrl": c["ctrl"], "n": c["n"]}
                     for k, c in v["by_category"].items()},
    }
    for arm, v in bans.items()
})

# steered examples: curated recipe experts, every prompt, every dose
PICK = ["L26_E174_+", "L28_E45_+", "L25_E121_+", "L43_E110_+", "L31_E193_+", "L28_E155_+", "L29_E78_+",
        "L40_E209_+", "L27_E187_+", "L31_E91_+"]
recipes = {x["id"]: x for x in json.loads((space / "recipes.json").read_text())}
baseline = json.loads((space / "baseline.json").read_text())
prompts = json.loads((space / "prompts.json").read_text())


def clip(s, n=520):
    s = (s or "").strip()
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + " …"


examples = []
for pid in PICK:
    x = recipes[pid]
    p = x["poles"][x["pole"]]
    outs = {}
    for o in p["out"]:
        outs.setdefault(o["prompt"], {})[str(o["dose"])] = {"final": clip(o.get("final")), "status": o.get("status")}
    examples.append({"id": pid, "layer": x["layer"], "expert": x["expert"], "name": p["name"], "desc": p["desc"],
                     "subject": p["subject"], "tokens": p["tokens"][:8], "theme": p["theme"][:8],
                     "pct": p["pct"], "judge": p["judge"], "judge_beam": p["judge_beam"], "out": outs})
dump("steered.json", {
    "prompts": prompts,
    "baseline": {k: {"final": clip(v["final"]), "status": v["status"]} for k, v in baseline.items()},
    "examples": examples,
})

# expert map: UMAP of write directions, + pole label per expert
poles = pd.read_parquet(atlas / "experts/poles.parquet")
plus = poles[poles.pole == "+"].set_index(["layer", "expert"])
emap = json.loads((space / "expert_map.json").read_text())
subjects = sorted(plus.beam_subject.dropna().unique().tolist())
sidx = {s: i for i, s in enumerate(subjects)}
pts = []
for layer, expert, x, y in emap["points"]:
    row = plus.loc[(layer, expert)] if (layer, expert) in plus.index else None
    subj = row.beam_subject if row is not None else None
    coh = bool(row.beam_coherent) if row is not None and row.beam_coherent is not None else False
    sig = bool(row.significant) if row is not None else False
    # names only where the label was tested and held (keeps the file small)
    name = row.beam_name if row is not None and coh and sig and isinstance(row.beam_name, str) else ""
    pts.append([layer, expert, round(x, 4), round(y, 4), sidx.get(subj, -1), int(sig), int(coh), name])
dump("map.json", {"subjects": subjects, "clusters": emap["clusters"], "points": pts})

# effort: readout by layer and position, off-dial reads, competition-math arms
an = json.loads((atlas / "effort/analysis.json").read_text())
cm = pd.read_parquet(atlas / "effort/competition_math.parquet")
ARMS = ["dial8", "dial32", "dial128", "dial512", "swap", "swap_ctrl", "route_up_prompt", "route_ctrl_prompt",
        "dir_spread"]
base = cm[cm.arm == "dial8"].set_index("q")
arms = []
for a in ARMS:
    s = cm[cm.arm == a].set_index("q")
    ratio = (s.think_tokens / base.think_tokens.reindex(s.index)).median()
    arms.append({"arm": a, "acc": r(s.correct.mean(), 3), "think_median": int(s.think_tokens.median()),
                 "think_ratio": r(ratio, 3), "cap_hits": int((s.finish == "length").sum()),
                 "won": int((s.correct & ~base.correct.reindex(s.index)).sum()),
                 "lost": int((~s.correct & base.correct.reindex(s.index)).sum())})
dump("effort.json", {
    "readout": {k: [r(v) for v in d["r2"]] for k, d in an["readout"].items()},
    "best_layer": {k: d["best_layer"] for k, d in an["readout"].items()},
    "off_dial_l12": {k: r(v, 1) for k, v in an["off_dial"]["n_last"]["12"].items()},
    "arms": arms,
})

# one token's route at the start of thinking, three questions, every effort
routes = json.loads((space / "routes.json").read_text())
QS = [0, 1, 2]
dump("routes.json", {
    "layers": routes["layers"], "efforts": routes["efforts"],
    "questions": [routes["questions"][q]["text"] for q in QS],
    "routes": {str(e): [routes["routes"]["think_start"][str(e)][q] for q in QS] for e in routes["efforts"]},
})
