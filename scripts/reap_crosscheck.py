# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Does the expert atlas predict what REAP pruning removed? (mrfakename/Beam-1.0-REAP-255B-A23B)

    uv run scripts/reap_crosscheck.py RUN_DIR REAP_SALIENCY.json [--verdicts VERDICTS.jsonl] [--out OUT.json]

REAP kept the 16 highest-saliency experts in each 32-expert routing group (saliency from ~925k calibration tokens
"weighted toward coding and agent use"); its card predicts losses on "multilingual text or niche knowledge". Each swept
expert gets categories from its steered direction's two poles (24 lens tokens, from the weights alone; plus the
words its pushes made Beam say, for the topical categories). Categories were fixed before looking at `kept`:

- multilingual: >= 50% of the lens tokens contain a letter outside Latin script
- code: >= 50% of the lens tokens look like code (brackets, operators, snake/camel case, file or markup syntax,
  common keywords)
- health / law / history (exploratory, keyword stems): >= 2 distinct stems among lens + steered-theme words

With --beam-labels (self_label.py on pole_rows.py rows), each expert also gets the subjects Beam gave its coherent
poles, and every subject is tested the same way.

Kept rate per category against a null that shuffles `kept` within each (layer, group) block, which preserves REAP's
exact 16-of-32 constraint. Also the within-layer rank correlation of REAP saliency with each token fraction.
"""

import argparse
import collections
import json
import math
import pathlib
import re
import unicodedata

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("run", type=pathlib.Path)
parser.add_argument("reap", type=pathlib.Path)
parser.add_argument("--verdicts", type=pathlib.Path, help="judge verdicts (judge_score.py --out) for a trusted subset")
parser.add_argument("--beam-labels", type=pathlib.Path, help="Beam's pole labels with subjects (self_label.py)")
parser.add_argument("--shuffles", type=int, default=10000)
parser.add_argument("--out", type=pathlib.Path)
args = parser.parse_args()
rng = np.random.default_rng(0)


def _byte_decoder():
    """GPT-2 byte-level BPE: printable stand-in characters -> raw bytes."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


DECODE = _byte_decoder()


def readable(token):
    try:
        return bytes(DECODE[c] for c in token).decode("utf-8", errors="replace")
    except KeyError:
        return token


def non_latin(text):
    return any(ch.isalpha() and not unicodedata.name(ch, "").startswith("LATIN") for ch in text)


CODE_WORDS = {"def", "return", "import", "const", "var", "let", "self", "null", "none", "true", "false", "int", "str",
              "void", "public", "private", "static", "class", "func", "function", "html", "json", "http", "https", "div",
              "span", "src", "url", "api", "println", "printf", "elif", "endif", "lambda", "async", "await", "struct"}


def code_like(text):
    t = text.strip()
    return bool(re.search(r"[(){}\[\];=<>$#\\|`]|::|->|\.\w|\w_\w|^_|[a-z][A-Z]", t)) or t.lower() in CODE_WORDS


TOPICS = {
    "health": ("health", "medic", "patient", "disease", "clinic", "symptom", "hospital", "doctor", "drug", "treatment",
               "therap", "cancer", "infect", "virus", "diagnos", "surg", "nurse", "vaccin", "pharma", "dose", "blood",
               "cardi", "diabet", "respirat", "pandemic", "mental", "illness", "medicat"),
    "law": ("legal", "court", "judge", "attorney", "lawyer", "statut", "plaintiff", "defendant", "contract",
            "constitution", "legislat", "regulat", "liabil", "criminal", "justice", "trial", "verdict", "jurisdic",
            "lawsuit", "tort", "crime", "prosecut", "sentenc", "testimon"),
    "history": ("histor", "ancient", "century", "centuries", "empire", "dynast", "medieval", "revolution", "colonial",
                "kingdom", "emperor", "battle", "roman", "civiliz", "reign", "archaeolog", "monarch", "pharaoh",
                "feudal", "conquest"),
}


def topic_hits(words):
    words = [w.lower() for w in words]
    return {t: len({s for s in stems for w in words if w.startswith(s)}) for t, stems in TOPICS.items()}


# experts: both poles of the steered direction
experts = collections.defaultdict(lambda: {"lens": [], "theme": [], "sig": False})
for r in map(json.loads, open(args.run / "ranked_matched.jsonl")):
    if r["kind"] != "expert_dir":
        continue
    e = experts[(r["layer"], r["expert"])]
    e["lens"] += [readable(t) for t in r["lens"][:12]]
    e["theme"] += r.get("theme") or []
    e["sig"] |= (r.get("pct_matched") or 0) >= 0.95

reap = json.load(open(args.reap))["layers"]
keys = sorted(k for k in experts if reap[str(k[0])].get("type") != "dense")
kept = np.array([k[1] in set(reap[str(k[0])]["kept"]) for k in keys])
saliency = np.array([reap[str(k[0])]["reap"][k[1]] for k in keys])
layer = np.array([k[0] for k in keys])
block = np.array([k[0] * 8 + k[1] // 32 for k in keys])

frac_nonlatin, frac_code, cats = [], [], collections.defaultdict(list)
for i, k in enumerate(keys):
    e = experts[k]
    toks = [t for t in e["lens"] if t.strip()]
    fn = np.mean([non_latin(t) for t in toks]) if toks else 0.0
    fc = np.mean([code_like(t) for t in toks]) if toks else 0.0
    frac_nonlatin.append(fn)
    frac_code.append(fc)
    words = re.findall(r"[A-Za-z]{4,}", " ".join(toks + e["theme"]))
    hits = topic_hits(words)
    labels = ["multilingual"] * int(fn >= 0.5) + ["code"] * int(fc >= 0.5) + [t for t, n in hits.items() if n >= 2]
    for c in labels:
        cats[c].append(i)
frac_nonlatin, frac_code = np.array(frac_nonlatin), np.array(frac_code)
beam_subjects = collections.defaultdict(list)
if args.beam_labels:
    by_expert = collections.defaultdict(set)
    for r in map(json.loads, open(args.beam_labels)):
        if r["name"] and r.get("coherent") is True and r.get("subject"):
            L, E, _ = r["id"].split("_")
            by_expert[(int(L[1:]), int(E[1:]))].add(r["subject"])
    for i, k in enumerate(keys):
        for subject in by_expert.get(k, ()):
            beam_subjects[subject].append(i)

trusted = None
if args.verdicts:
    good = {(v["layer"], v["expert"]) for v in map(json.loads, open(args.verdicts))
            if v["stratum"] == "sig" and v["correct"] and v["confidence"] == 3}
    trusted = np.array([k in good for k in keys])

# null: shuffle kept within each (layer, group) block
blocks = [np.flatnonzero(block == b) for b in np.unique(block)]


def shuffled():
    out = kept.copy()
    for idx in blocks:
        out[idx] = rng.permutation(kept[idx])
    return out


null = np.stack([shuffled() for _ in range(args.shuffles)])


def test(idx):
    idx = np.asarray(idx, dtype=int)
    if len(idx) == 0:
        return None
    obs = kept[idx].mean()
    dist = null[:, idx].mean(1)
    p = (1 + min((dist <= obs).sum(), (dist >= obs).sum())) / (1 + len(dist)) * 2
    return {"n": int(len(idx)), "kept": round(float(obs), 3), "null_mean": round(float(dist.mean()), 3),
            "p_two_sided": round(min(p, 1.0), 4)}


def within_layer_spearman(x):
    rs = []
    for L in np.unique(layer):
        m = layer == L
        a, b = np.argsort(np.argsort(saliency[m])), np.argsort(np.argsort(x[m]))
        if a.std() and b.std():
            rs.append(np.corrcoef(a, b)[0, 1])
    return round(float(np.mean(rs)), 3)


report = {"experts": len(keys), "kept_overall": round(float(kept.mean()), 3), "categories": {}, "trusted": {},
          "spearman_saliency_vs": {"frac_nonlatin": within_layer_spearman(frac_nonlatin),
                                   "frac_code": within_layer_spearman(frac_code)}}
print(f"{len(keys)} swept MoE experts, kept {kept.mean():.1%}")
print(f"{'category':14s} {'n':>5s} {'kept':>6s} {'null':>6s}  p")
for c in ("multilingual", "code", "health", "law", "history"):
    res = test(cats[c])
    report["categories"][c] = res
    if res:
        print(f"{c:14s} {res['n']:5d} {res['kept']:6.1%} {res['null_mean']:6.1%}  {res['p_two_sided']}")
    if trusted is not None:
        t = test([i for i in cats[c] if trusted[i]])
        report["trusted"][c] = t
        if t:
            print(f"  trusted only {t['n']:5d} {t['kept']:6.1%} {t['null_mean']:6.1%}  {t['p_two_sided']}")
if beam_subjects:
    report["beam_subjects"] = {}
    print("by Beam's subject (coherent poles):")
    for c, idx in sorted(beam_subjects.items(), key=lambda kv: -len(kv[1])):
        res = test(idx)
        report["beam_subjects"][c] = res
        print(f"  {c:20s} {res['n']:5d} {res['kept']:6.1%} {res['null_mean']:6.1%}  {res['p_two_sided']}")
by_band = {}
for c in ("multilingual", "code"):
    by_band[c] = {f"L{8 * b}-{8 * b + 7}": test([i for i in cats[c] if layer[i] // 8 == b]) for b in range(7)}
report["by_band"] = by_band
print("within-layer Spearman(REAP saliency, fraction):", report["spearman_saliency_vs"])
for c, bands in by_band.items():
    print(c, {b: (r["n"], r["kept"]) for b, r in bands.items() if r})
for c in ("health", "law", "history"):
    ex = [keys[i] for i in cats[c]][:6]
    print(f"examples {c}:", [(L, e, "kept" if kept[keys.index((L, e))] else "pruned") for L, e in ex])
if args.out:
    args.out.write_text(json.dumps(report, indent=1))
