# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "safetensors", "pyarrow", "numpy"]
# ///
"""Convert the effort activation captures to safetensors with a Parquet index, for the public dataset.

    uv run scripts/export_captures.py OUT_DIR [--runs data/runs]

Each capture file from beam_steer_capture_dump holds resid [52 layers, rows, 5120] (BF16, the residual leaving each
layer), router [52, rows, 256] (FP32 router logits; zero for the dense layers 0-2) and an index of (capture_key,
position, token_id, slot) per row. Here each becomes OUT/captures/<name>_NN.safetensors with the two tensors, and
one OUT/captures/<name>_index.parquet per set lists every row: file, row, question, effort, position, token id and
the position's kind.

  gsm8k_effort        main session: 200 GSM8K test questions (row index in openai/gsm8k main/test) x efforts
                      2, 8, 16, 32, 64, 128, 256, 512, 1000; kinds n_last (last digit of the effort number), sys_end,
                      think_start, thought1, thought2
  competition_math    effort2 transfer capture: 100 problems (AIME 2024, AIME 2025, AMC 2023 in that order; see
                      effort/competition_math for the source rows) x the dial; kinds n_last, prompt_end
"""

import argparse
import pathlib

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from safetensors.torch import save_file

parser = argparse.ArgumentParser()
parser.add_argument("out", type=pathlib.Path)
parser.add_argument("--runs", type=pathlib.Path, default=pathlib.Path("data/runs"))
args = parser.parse_args()
SETS = {"gsm8k_effort": (args.runs / "20261027-145310", "effort_capture_*.pt",
                         ("n_last", "sys_end", "think_start", "thought1", "thought2")),
        "competition_math": (args.runs / "20261028-044229", "effort2_capture_*.pt", ("n_last", "prompt_end"))}
out = args.out / "captures"
out.mkdir(parents=True, exist_ok=True)

for name, (run, pattern, kinds) in SETS.items():
    rows = []
    for f, path in enumerate(sorted(run.glob(pattern))):
        cap = torch.load(path, weights_only=True, mmap=True)
        target = out / f"{name}_{f:02d}.safetensors"
        save_file({"resid": cap["resid"].contiguous(), "router": cap["router"].contiguous()}, target,
                  metadata={"source": path.name, "layers": "52", "note": "router is zero for dense layers 0-2"})
        for r, (key, pos, tok, slot) in enumerate(cap["index"]):
            rows.append({"file": target.name, "row": r, "question": key[0], "effort": key[1], "position": pos,
                         "token_id": tok, "slot": slot})
        print(f"{target.name}: {len(cap['index'])} rows, dropped {cap['dropped']}, {target.stat().st_size / 1e9:.2f} GB")
    # positions of one (question, effort) request may span two files; their order in the prompt gives the kind
    by_key = {}
    for r in rows:
        by_key.setdefault((r["question"], r["effort"]), []).append(r)
    complete = 0
    for group in by_key.values():
        group.sort(key=lambda r: r["position"])
        if len(group) == len(kinds):
            complete += 1
            for r, kind in zip(group, kinds):
                r["kind"] = kind
        else:
            for r in group:
                r["kind"] = None
    pq.write_table(pa.Table.from_pylist(rows), out / f"{name}_index.parquet", compression="zstd")
    print(f"{name}: {len(rows)} rows, {complete}/{len(by_key)} requests complete")
