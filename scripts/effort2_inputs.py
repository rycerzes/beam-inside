# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "torch"]
# ///
"""Interventions for the effort2 phase, from a v2 session's effort outputs (after effort_analysis.py).

    uv run scripts/effort2_inputs.py RUN_DIR [--out data/effort2/inputs.pt]

  r            [layers, hidden] effort direction: residual change per octave of N at the think start
  route_up     [(layer, expert, delta)]: think-start effort experts whose selection rate moves by >= 0.5 between
               N = 8 and N = 512 (significant against shuffled N), forced in (+8) if chosen more at 512, banned (-30)
               if chosen more at 8. Applied to an N = 8 prompt, it imitates N = 512's routing.
  route_down   the same experts with the signs swapped (imitates N = 8's routing on an N = 512 prompt)
  route_ctrl   for each entry of route_up, a random expert of the same layer and the same delta, outside the effort set
  swap         {"layer", "delta"}: mean residual at the last digit of N = 512 minus at N = 8, at the layer where the
               digit's readout peaks; added at the effort number's position of an N = 8 prompt
"""

import argparse
import json
import pathlib

import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("run", type=pathlib.Path)
parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("data/effort2/inputs.pt"))
parser.add_argument("--min-rd", type=float, default=0.5)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
rng = np.random.default_rng(args.seed)
config = json.loads(pathlib.Path("meta/config.json").read_text())

r = torch.load(args.run / "effort_direction.pt", weights_only=True)["r"].float()
analysis = json.loads((args.run / "effort_analysis.json").read_text())
think = analysis["experts"]["think_start"]
chosen = [t for t in think["top"] if abs(t["rd"]) >= args.min_rd and abs(t["rd"]) > think["threshold"]]
route_up = [(t["layer"], t["expert"], 8.0 if t["rd"] > 0 else -30.0) for t in chosen]
route_down = [(L, e, -30.0 if d > 0 else 8.0) for L, e, d in route_up]
effort_set = {(L, e) for L, e, _ in route_up}
route_ctrl = []
for L, _, d in route_up:
    pool = [e for e in range(config["num_experts"]) if (L, e) not in effort_set and (L, e) not in {(a, b) for a, b, _ in route_ctrl}]
    route_ctrl.append((L, int(rng.choice(pool)), d))

# Swap vector from the capture files: rows are (question, N) with 5 positions in prompt order; the first is N's last digit.
layer = analysis["readout"]["n_last"]["best_layer"]
sums, counts = {8: 0, 512: 0}, {8: 0, 512: 0}
for path in sorted(args.run.glob("effort_capture_*.pt")):
    cap = torch.load(path, weights_only=True, mmap=True)
    rows = {}
    for i, (key, pos, _, _) in enumerate(cap["index"]):
        rows.setdefault(tuple(key), []).append((pos, i))
    for (q, n), rs in rows.items():
        if n in sums:
            first = min(rs)[1]
            sums[n] = sums[n] + cap["resid"][layer, first].float()
            counts[n] += 1
swap = {"layer": layer, "delta": sums[512] / counts[512] - sums[8] / counts[8]}

args.out.parent.mkdir(parents=True, exist_ok=True)
torch.save({"r": r, "route_up": route_up, "route_down": route_down, "route_ctrl": route_ctrl, "swap": swap}, args.out)
by_layer = {}
for L, e, d in route_up:
    by_layer.setdefault(L, []).append(f"e{e}{'+' if d > 0 else '-'}")
print(f"{len(route_up)} effort experts ({sum(d > 0 for *_, d in route_up)} forced in, {sum(d < 0 for *_, d in route_up)} banned) "
      f"over {len(by_layer)} layers: " + "; ".join(f"L{L} {' '.join(v)}" for L, v in sorted(by_layer.items())))
print(f"swap at L{layer}: ‖delta‖ {float(swap['delta'].norm()):.2f} from {counts[512]} / {counts[8]} prompts; "
      f"‖r‖ per octave at L20 {float(r[20].norm()):.3f}  -> {args.out}")
