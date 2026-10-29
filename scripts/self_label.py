# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx", "huggingface_hub"]
# ///
"""Beam names its own experts: turn each pole's weights-only token label into a short concept label.

    uv run scripts/self_label.py ROWS.jsonl OUT_DIR [--ids IDS.txt] [--max-usd 34] [--workers 8]
    -> OUT_DIR/labels.jsonl ({"id", "name", "description", "subject", "coherent", "usage"})
       OUT_DIR/items_beam.jsonl, when ROWS are judge items (have "options"): the same items with the token label
       replaced by Beam's label, for judge_router.py --label-kind description

ROWS are judge items (judge_items.py) or pole rows (pole_rows.py); only "id" and "label" are sent. Beam sees only the
token label, never steered outputs or the answer key, so the three-way test of its label stays blind.

Rate limits and server errors back off exponentially with jitter (honouring Retry-After); a row that still fails is
written with name null and retried on the next run, which skips rows already labelled. Spend is estimated from the
returned token usage at the router's listed price times --bill-factor (the pilot's bill / token-priced cost), counting
rows from earlier runs, and no new request starts once it reaches --max-usd. The HF token is only sent to
router.huggingface.co.
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
parser.add_argument("rows", type=pathlib.Path)
parser.add_argument("out", type=pathlib.Path)
parser.add_argument("--ids", type=pathlib.Path, help="one row id per line, in order; default all rows")
parser.add_argument("--model", default="reflection/Beam-1.0-501B-A23B:baseten")
parser.add_argument("--reasoning-effort", default="low")
parser.add_argument("--max-tokens", type=int, default=4096, help="thinking counts against it")
parser.add_argument("--workers", type=int, default=8)
parser.add_argument("--max-usd", type=float, required=True)
parser.add_argument("--price-in", type=float, default=0.40, help="USD per M input tokens")
parser.add_argument("--price-out", type=float, default=1.20, help="USD per M output tokens")
parser.add_argument("--bill-factor", type=float, default=1.22)
parser.add_argument("--retries", type=int, default=8)
args = parser.parse_args()

SUBJECTS = ("biology", "business", "chemistry", "computer science", "economics", "engineering", "health", "history",
            "law", "math", "philosophy", "physics", "psychology", "other knowledge", "general", "code",
            "language or script", "format or syntax", "none")

PROMPT = """Below are the vocabulary tokens most promoted by one internal unit (an expert's write direction) inside \
you, Beam, computed from your weights alone. Tokens can be subword pieces and can mix languages and scripts.

Tokens: {label}

Name the concept this unit writes, as a human-readable label for an atlas of your experts. Prefer the shared meaning \
across languages over surface form; if the tokens share only a form (a suffix, a script, punctuation, code syntax), \
name that. Also pick the one subject it belongs to from: {subjects}. The subject is the topic of the concept, \
whatever languages the tokens are in: an academic or professional field if it belongs to one ("other knowledge" for \
fields not listed), "general" for everyday concepts, function words and discourse markers, "code" for programming, \
"language or script" only when the unit is about one language or writing system itself rather than a concept, \
"format or syntax" for punctuation, casing or layout, "none" if the tokens are incoherent. Reply with exactly one \
line of JSON and nothing else:
{{"name": "<2-6 words>", "description": "<one sentence>", "subject": "<one subject from the list>", \
"coherent": true|false}}"""


def cost(usage):
    u = usage or {}
    return (u.get("prompt_tokens", 0) * args.price_in + u.get("completion_tokens", 0) * args.price_out) \
        / 1e6 * args.bill_factor


def post(client, body):
    """POST with exponential backoff and jitter on 429, 5xx and network errors; returns (response or None, error)."""
    error = None
    for attempt in range(args.retries):
        try:
            r = client.post("https://router.huggingface.co/v1/chat/completions", json=body, timeout=300)
        except httpx.HTTPError as e:
            error, r = f"{type(e).__name__}", None
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


def ask(client, row):
    body = {"model": args.model, "temperature": 0, "max_tokens": args.max_tokens,
            "reasoning_effort": args.reasoning_effort,
            "messages": [{"role": "user", "content": PROMPT.format(label=row["label"], subjects=", ".join(SUBJECTS))}]}
    r, error = post(client, body)
    if r is None:
        return {"id": row["id"], "name": None, "raw": error, "usage": None}
    data = r.json()
    text = data["choices"][0]["message"].get("content") or ""
    m = re.search(r"\{.*\}", text, re.S)
    try:
        parsed = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        parsed = None
    if not parsed or not parsed.get("name"):
        return {"id": row["id"], "name": None, "raw": text[:300], "usage": data.get("usage")}
    subject = str(parsed.get("subject", "")).strip().lower()
    return {"id": row["id"], "name": parsed["name"], "description": parsed.get("description", ""),
            "subject": subject if subject in SUBJECTS else None, "subject_raw": parsed.get("subject"),
            "coherent": parsed.get("coherent"), "usage": data.get("usage")}


rows = {str(r["id"]): r for r in map(json.loads, open(args.rows))}
ids = open(args.ids).read().split() if args.ids else list(rows)
args.out.mkdir(parents=True, exist_ok=True)
path = args.out / "labels.jsonl"
previous = [json.loads(line) for line in open(path)] if path.exists() else []
spent = sum(cost(r["usage"]) for r in previous)  # failed rows were billed too
path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in previous if r["name"] is not None))
done = {str(r["id"]) for r in previous if r["name"] is not None}
todo = [rows[i] for i in ids if i not in done]
print(f"{len(done)} already labelled, {len(todo)} to do, ${spent:.2f} spent so far, cap ${args.max_usd:.2f}", flush=True)

start, n, failed, stopped = time.time(), 0, 0, False
with httpx.Client(headers={"Authorization": f"Bearer {get_token()}"}) as client, open(path, "a") as f, \
        cf.ThreadPoolExecutor(args.workers) as pool:
    queue, pending = iter(todo), set()
    while True:
        while not stopped and len(pending) < 2 * args.workers:
            if spent >= args.max_usd:
                stopped = True
                print(f"cap reached at ${spent:.2f}; not starting new requests", flush=True)
                break
            row = next(queue, None)
            if row is None:
                break
            pending.add(pool.submit(ask, client, row))
        if not pending:
            break
        finished, pending = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
        for fut in finished:
            res = fut.result()
            spent += cost(res["usage"])
            n += 1
            failed += res["name"] is None
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush()
            if n % 250 == 0:
                rate = n / (time.time() - start)
                print(f"{n}/{len(todo)}  ${spent:.2f}  {rate:.2f}/s  failed {failed}", flush=True)

labels = {str(r["id"]): r for r in map(json.loads, open(path)) if r["name"]}
if any("options" in rows[i] for i in ids[:1]):
    with open(args.out / "items_beam.jsonl", "w") as f:
        for i in ids:
            if i in labels:
                f.write(json.dumps({**rows[i], "label": f"{labels[i]['name']}: {labels[i]['description']}"},
                                   ensure_ascii=False) + "\n")
print(f"done: {len(labels)} of {len(ids)} labelled, {failed} failed this run, ${spent:.2f} estimated in total, "
      f"{n / max(time.time() - start, 1e-9):.2f} rows/s", flush=True)
