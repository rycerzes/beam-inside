# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx", "huggingface_hub"]
# ///
"""Score three-way label tests (judge_items.py) with an open model on the HF Inference router.

    uv run scripts/judge_router.py ITEMS.jsonl OUT.jsonl [--model Qwen/Qwen3.8-Flash-Next-FP8:nscale] [--limit N]
        [--reasoning-effort low] [--label-kind tokens|description]

Temperature 0, one call per item, the same instructions the Claude reference judges got (with --label-kind
description, the one sentence that says what a LABEL is describes a written concept label instead of tokens). Writes one row per item
({"id", "choice", "confidence", "usage"}) and resumes: items already in OUT are skipped. The HF token comes from the
local login and is only sent to router.huggingface.co.
"""

import argparse
import concurrent.futures as cf
import json
import pathlib
import random
import re
import time

import httpx
from huggingface_hub import get_token

parser = argparse.ArgumentParser()
parser.add_argument("items", type=pathlib.Path)
parser.add_argument("out", type=pathlib.Path)
parser.add_argument("--model", default="Qwen/Qwen3.8-Flash-Next-FP8:nscale")
parser.add_argument("--limit", type=int, default=0)
parser.add_argument("--workers", type=int, default=8)
parser.add_argument("--max-tokens", type=int, default=4096, help="thinking counts against it")
parser.add_argument("--reasoning-effort", help="sent only when set (e.g. low for Beam, whose unset effort is max)")
parser.add_argument("--label-kind", choices=("tokens", "description"), default="tokens")
args = parser.parse_args()

SYSTEM = """You are a blind judge in an interpretability experiment on a language model (Beam). Researchers pushed the \
model's internal state along the write direction of one "expert" and recorded its answers to three fixed prompts \
("Who are you?", "Tell me something about snakes.", "Describe an orange in two sentences."). Each expert has a LABEL: \
the vocabulary tokens its direction promotes, computed from weights only.

You get one LABEL and three options (A, B, C). Each option is the model's answers to the three prompts while steered \
by a different expert; texts may show the model's reasoning ("[analysis]") and its answer ("[final]"), truncated. \
Exactly one option was produced by the labelled expert. The other two come from other experts with a similar overall \
effect size. Pick the option whose answers best reflect the LABEL's theme (words, topics, languages, formats, or \
concepts related to the label tokens intruding into the answers). If it is ambiguous, still pick your best guess and \
lower your confidence.

Reply with exactly one line of JSON and nothing else: {"choice": "A"|"B"|"C", "confidence": 1|2|3} \
(3 = clear match, 1 = guess)."""


if args.label_kind == "description":
    SYSTEM = SYSTEM.replace("Each expert has a LABEL: \
the vocabulary tokens its direction promotes, computed from weights only.",
                            "Each expert has a LABEL: \
a short written description of the concept its direction promotes, derived from weights only.")
    SYSTEM = SYSTEM.replace("concepts related to the label tokens intruding", "concepts related to the label intruding")
    assert "a short written description" in SYSTEM and "label intruding" in SYSTEM


def render(item):
    kind = "tokens this unit promotes" if args.label_kind == "tokens" else "concept this unit promotes"
    lines = [f"LABEL ({kind}): {item['label']}"]
    for letter in "ABC":
        lines.append(f"--- Option {letter} ---")
        lines += [f"  [{k}] {v}" for k, v in item["options"][letter].items()]
    return "\n".join(lines)


def post(client, body, retries=8):
    """POST with exponential backoff and jitter on 429, 5xx and network errors; returns (response or None, error)."""
    error = None
    for attempt in range(retries):
        try:
            r = client.post("https://router.huggingface.co/v1/chat/completions", json=body, timeout=300)
        except httpx.HTTPError as e:
            error, r = type(e).__name__, None
        else:
            if r.status_code == 200:
                return r, None
            error = f"HTTP {r.status_code}"
            if r.status_code != 429 and r.status_code < 500:
                return None, f"{error}: {r.text[:200]}"
        wait = min(120.0, 2.0 ** attempt) * random.uniform(0.5, 1.5)
        if r is not None and r.headers.get("retry-after", "").isdigit():
            wait = max(wait, float(r.headers["retry-after"]))
        time.sleep(wait)
    return None, error


def judge(client, item):
    body = {"model": args.model, "temperature": 0, "max_tokens": args.max_tokens,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": render(item)}]}
    if args.reasoning_effort:
        body["reasoning_effort"] = args.reasoning_effort
    r, error = post(client, body)
    if r is None:
        return {"id": item["id"], "choice": None, "confidence": None, "raw": error, "usage": None}
    data = r.json()
    text = data["choices"][0]["message"].get("content") or ""
    m = re.search(r'"choice"\s*:\s*"([ABC])".*?"confidence"\s*:\s*([123])', text, re.S)
    return {"id": item["id"], "choice": m.group(1) if m else None, "confidence": int(m.group(2)) if m else None,
            "raw": None if m else text[:300], "usage": data.get("usage")}


items = [json.loads(line) for line in open(args.items)]
if args.out.exists():  # keep parsed rows, retry the rest
    kept = [line for line in open(args.out) if json.loads(line)["choice"] is not None]
    args.out.write_text("".join(kept))
done = {json.loads(line)["id"] for line in open(args.out)} if args.out.exists() else set()
todo = [it for it in items if it["id"] not in done][: args.limit or None]
with httpx.Client(headers={"Authorization": f"Bearer {get_token()}"}) as client, open(args.out, "a") as f, \
        cf.ThreadPoolExecutor(args.workers) as pool:
    for row in pool.map(lambda it: judge(client, it), todo):
        f.write(json.dumps(row) + "\n")
        f.flush()
rows = [json.loads(line) for line in open(args.out)]
tok_in = sum((r["usage"] or {}).get("prompt_tokens", 0) for r in rows)
tok_out = sum((r["usage"] or {}).get("completion_tokens", 0) for r in rows)
print(f"{len(rows)} judged, {sum(r['choice'] is None for r in rows)} unparsed; tokens in {tok_in}, out {tok_out}")
