// Inside Beam explorer: static page, data from the beam-expert-atlas dataset (or ?data=URL for a local copy).
"use strict";

const DATA = new URLSearchParams(location.search).get("data") ||
  "https://huggingface.co/datasets/rycerzes/beam-expert-atlas/resolve/main/explorer/";
const SUBJECT_COLORS = {
  "code": "#4c6ef5", "computer science": "#15aabf", "math": "#f59f00", "format or syntax": "#868e96",
  "language or script": "#be4bdb", "general": "#adb5bd", "other knowledge": "#82c91e", "business": "#e8590c",
  "health": "#e03131", "law": "#5c3d99", "biology": "#2b8a3e", "engineering": "#1971c2", "physics": "#0b7285",
  "economics": "#d9480f", "psychology": "#c2255c", "history": "#a07c3b", "chemistry": "#5f3dc4", "philosophy": "#495057",
};
const DOSE_LABEL = { "0.4": "gentle push", "1": "full push", "-1": "full push" };

let SUMMARY, INDEX, PROMPTS, BASELINE;
const layerCache = new Map();

// ---- helpers ---------------------------------------------------------------------------------------------------
function el(tag, attrs = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined && v !== false) n.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) if (kid !== null && kid !== undefined) n.append(kid instanceof Node ? kid : String(kid));
  return n;
}
const SVGNS = "http://www.w3.org/2000/svg";
function sv(tag, attrs = {}, text) {
  const n = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (text !== undefined) n.textContent = text;
  return n;
}
const fmt = (x, d = 1) => (x > 0 ? "+" : x < 0 ? "−" : "") + Math.abs(x).toFixed(d);
const pct = (x) => (100 * x).toFixed(0) + "%";
const clip = (s, n) => (s.length > n ? s.slice(0, n) + "…" : s);
async function getJSON(name) {
  const r = await fetch(DATA + name);
  if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
  return r.json();
}
function parseId(id) {
  const m = /^L(\d+)_E(\d+)(?:_([+-]))?$/.exec(id);
  return m && { layer: +m[1], expert: +m[2], pole: m[3] };
}

// ---- front-page figures ----------------------------------------------------------------------------------------
function heatmap() {
  const bans = ["law", "health", "math"];
  const cats = Object.keys(SUMMARY.bans.law.by_category);
  const cw = 58, ch = 34, left = 70, top = 70, W = left + cats.length * cw + 40, H = top + bans.length * ch + 8;
  const s = sv("svg", { viewBox: `0 0 ${W} ${H}`, style: `min-width:${Math.min(W, 760)}px` });
  cats.forEach((c, j) => {
    const x = left + j * cw + cw / 2;
    s.append(sv("text", { x, y: top - 8, transform: `rotate(-40 ${x} ${top - 8})` }, c));
  });
  bans.forEach((b, i) => {
    const y = top + i * ch;
    s.append(sv("text", { x: left - 8, y: y + ch / 2 + 4, "text-anchor": "end", style: "font-weight:600" }, `− ${b}`));
    cats.forEach((c, j) => {
      const v = 100 * SUMMARY.bans[b].by_category[c].acc_diff;
      const a = Math.min(1, Math.abs(v) / 14);
      const x = left + j * cw;
      s.append(sv("rect", { x: x + 1, y: y + 1, width: cw - 2, height: ch - 2, rx: 3,
        fill: v < 0 ? "var(--neg)" : "var(--pos)", "fill-opacity": (0.08 + 0.85 * a).toFixed(2) }));
      if (c === b) s.append(sv("rect", { x: x + 1, y: y + 1, width: cw - 2, height: ch - 2, rx: 3, fill: "none",
        stroke: "var(--ink)", "stroke-width": 1.5 }));
      s.append(sv("text", { x: x + cw / 2, y: y + ch / 2 + 4, "text-anchor": "middle", class: "val",
        style: a > 0.55 ? "fill:#fff;font-weight:600" : "" }, fmt(v)));
    });
  });
  document.getElementById("heatmap").append(s);
}

function cards() {
  const S = SUMMARY;
  const items = [
    [S.sweep.significant.toLocaleString(), `of ${S.sweep.testable.toLocaleString()} expert directions`,
      `move Beam's outputs toward a theme beyond effect-matched random controls (${S.sweep.expected_by_chance.toLocaleString()} expected by chance).`],
    [`${pct(S.judge.sig.rate)} <small>vs ${pct(S.judge.random.rate)}</small>`, "blind judge accuracy",
      `picking a significant direction's steered output from three, given its top tokens; random directions ${pct(S.judge.random.rate)}, chance 33%.`],
    [`${(S.self_label.sig_if_coherent.rate / S.self_label.sig_if_incoherent.rate).toFixed(1)}×`, "Beam knows its readable experts",
      `directions Beam called coherent pass the steering test ${pct(S.self_label.sig_if_coherent.rate)} of the time, the rest ${pct(S.self_label.sig_if_incoherent.rate)}.`],
    [`${fmt(100 * S.reap.all.acc_diff)} <small>pts</small>`, "REAP pruning, emulated",
      `MMLU-Pro after banning the experts REAP-255B removed; health ${fmt(100 * S.reap.by_category.health.acc_diff, 0)}, engineering ${fmt(100 * S.reap.by_category.engineering.acc_diff, 0)}.`],
  ];
  const box = document.getElementById("findings");
  for (const [big, head, body] of items) {
    const b = el("div", { class: "big" });
    b.innerHTML = big; // our own numbers only
    box.append(el("div", { class: "card" }, b, el("div", { style: "font-weight:600" }, head), el("p", {}, body)));
  }
}

function depth() {
  const bands = Object.keys(SUMMARY.depth);
  const series = [
    ["Beam: coherent", (d) => d.beam_coherent, "var(--accent)", ""],
    ["judge, significant", (d) => d.judge_sig_token, "var(--pos)", ""],
    ["judge, random", (d) => d.judge_random_token, "var(--ink-3)", "4 3"],
    ["significant", (d) => d.significant, "var(--neg)", ""],
  ];
  const W = 460, H = 250, l = 34, r = 24, t = 10, b = 54;
  const x = (i) => l + (i * (W - l - r)) / (bands.length - 1), y = (v) => t + (1 - v) * (H - t - b);
  const s = sv("svg", { viewBox: `0 0 ${W} ${H}` });
  for (const v of [0, 0.25, 0.5, 0.75, 1]) {
    s.append(sv("line", { x1: l, x2: W - r, y1: y(v), y2: y(v), class: "axis" }));
    s.append(sv("text", { x: l - 4, y: y(v) + 4, "text-anchor": "end" }, pct(v)));
  }
  s.append(sv("line", { x1: l, x2: W - r, y1: y(1 / 3), y2: y(1 / 3), stroke: "var(--ink-3)", "stroke-dasharray": "2 3" }));
  s.append(sv("text", { x: W - r, y: y(1 / 3) - 3, "text-anchor": "end" }, "chance"));
  bands.forEach((bd, i) => s.append(sv("text", { x: x(i), y: H - b + 16, "text-anchor": "middle" }, bd)));
  series.forEach(([name, f, color, dash], k) => {
    const pts = bands.map((bd, i) => [i, f(SUMMARY.depth[bd])]).filter(([, v]) => v && v.n >= 20);
    s.append(sv("polyline", { points: pts.map(([i, v]) => `${x(i)},${y(v.rate)}`).join(" "), fill: "none",
      stroke: color, "stroke-width": 2, "stroke-dasharray": dash }));
    pts.forEach(([i, v]) => s.append(sv("circle", { cx: x(i), cy: y(v.rate), r: 2.8, fill: color })));
    const lx = l + (k % 2) * 200, ly = H - 18 + Math.floor(k / 2) * 14;
    s.append(sv("line", { x1: lx, x2: lx + 16, y1: ly - 4, y2: ly - 4, stroke: color, "stroke-width": 2, "stroke-dasharray": dash }));
    s.append(sv("text", { x: lx + 20, y: ly }, name));
  });
  document.getElementById("depth").append(s);
}

function reap() {
  const rows = Object.entries(SUMMARY.reap.by_category).map(([c, v]) => [c, 100 * v.acc_diff]).sort((a, b) => a[1] - b[1]);
  const W = 460, rh = 17, l = 112, H = rows.length * rh + 24, max = 35;
  const s = sv("svg", { viewBox: `0 0 ${W} ${H}` });
  const x = (v) => l + (Math.abs(v) / max) * (W - l - 44);
  rows.forEach(([c, v], i) => {
    const y = 4 + i * rh;
    s.append(sv("text", { x: l - 6, y: y + 11, "text-anchor": "end" }, c));
    s.append(sv("rect", { x: l, y: y + 2, width: x(v) - l, height: rh - 5, rx: 2, fill: "var(--neg)", "fill-opacity": 0.75 }));
    s.append(sv("text", { x: x(v) + 4, y: y + 11, class: "val" }, fmt(v)));
  });
  s.append(sv("text", { x: l, y: H - 4 }, `all subjects ${fmt(100 * SUMMARY.reap.all.acc_diff)} pts (62.3% → ${(100 * SUMMARY.reap.all.acc_a).toFixed(1)}%)`));
  document.getElementById("reap").append(s);
}

function effort() {
  const r2 = SUMMARY.effort.readout_r2;
  const lines = [["number's last digit", r2.n_last, "var(--accent)", ""], ["think start", r2.think_start, "var(--ink-3)", "4 3"]];
  const W = 460, H = 220, l = 34, r = 8, t = 22, b = 40, n = r2.n_last.length, lo = 0.4;
  const x = (i) => l + (i * (W - l - r)) / (n - 1), y = (v) => t + ((1 - Math.max(v, lo)) / (1 - lo)) * (H - t - b);
  const s = sv("svg", { viewBox: `0 0 ${W} ${H}` });
  for (const v of [0.4, 0.6, 0.8, 1]) {
    s.append(sv("line", { x1: l, x2: W - r, y1: y(v), y2: y(v), class: "axis" }));
    s.append(sv("text", { x: l - 4, y: y(v) + 4, "text-anchor": "end" }, v.toFixed(1)));
  }
  for (const L of [0, 12, 24, 36, 51]) s.append(sv("text", { x: x(L), y: H - b + 14, "text-anchor": "middle" }, `L${L}`));
  lines.forEach(([name, ys, color, dash], k) => {
    s.append(sv("polyline", { points: ys.map((v, i) => `${x(i)},${y(v)}`).join(" "), fill: "none", stroke: color,
      "stroke-width": 2, "stroke-dasharray": dash }));
    s.append(sv("line", { x1: l + k * 170, x2: l + k * 170 + 16, y1: H - 8, y2: H - 8, stroke: color, "stroke-width": 2, "stroke-dasharray": dash }));
    s.append(sv("text", { x: l + k * 170 + 20, y: H - 4 }, name));
  });
  const best = r2.n_last.indexOf(Math.max(...r2.n_last));
  s.append(sv("text", { x: x(best) + 4, y: y(r2.n_last[best]) - 6 }, `R² ${r2.n_last[best].toFixed(3)} at L${best}`));
  document.getElementById("effort").append(s);

  const A = SUMMARY.effort.competition_math;
  const rows = [
    ["effort 8 (start)", "dial8", false], ["effort 32", "dial32", false], ["effort 512", "dial512", false],
    ["digit swap toward 512", "swap", false], ["  random vector, same norm", "swap_ctrl", true],
    ["re-route 37 effort experts (prompt)", "route_up_prompt", false], ["  37 random experts (prompt)", "route_ctrl_prompt", true],
  ];
  const tb = el("table", { class: "arms" },
    el("thead", {}, el("tr", {}, el("th", {}, "arm"), el("th", { class: "num" }, "accuracy"), el("th", { class: "num" }, "median thinking"))),
    el("tbody", {}, rows.map(([name, k, ctrl]) => el("tr", { class: ctrl ? "ctrl" : "" },
      el("td", {}, name), el("td", { class: "num" }, pct(A[k].acc)), el("td", { class: "num" }, `${Math.round(A[k].median_think)} tok`)))));
  document.getElementById("arms").append(tb, el("p", { class: "note", style: "margin-top:8px" },
    "The digit swap lengthens thinking 1.2× over its control (p = 5e-8). Random re-routing lifts accuracy as much as " +
    "effort-expert re-routing: the low-effort state breaks easily, and that gain is not effort-specific."));
}

function gallery(G) {
  const box = document.getElementById("gallery");
  for (const g of G) {
    const p = g.poles[g.pole];
    const who = p.out.find((o) => o.prompt === "who" && o.dose === 1) || p.out[0];
    box.append(el("button", { class: "g", onclick: () => openCard(g.layer, g.expert, g.pole) },
      el("div", { class: "where" }, `Layer ${g.layer} · expert ${g.expert} · ${p.subject}`),
      el("div", { class: "name" }, p.name),
      el("div", {}, p.tokens.slice(0, 5).map((t) => el("span", { class: "chip" }, t.trim() || t))),
      el("blockquote", {}, clip(who.final || who.analysis, 180))));
  }
}

// ---- search and layer map --------------------------------------------------------------------------------------
function search() {
  const F = Object.fromEntries(INDEX.fields.map((f, i) => [f, i]));
  const rows = INDEX.rows;
  const hay = rows.map((r) => `${r[F.name]} ${r[F.subject]} ${r[F.tokens]} ${r[F.id]}`.toLowerCase());
  const subjects = [...new Set(rows.map((r) => r[F.subject]).filter((s) => s && s !== "none"))].sort();
  const sel = document.getElementById("f-subj");
  subjects.forEach((s) => sel.append(el("option", { value: s }, s)));
  const q = document.getElementById("q"), out = document.getElementById("results"), count = document.getElementById("count");
  const run = () => {
    const terms = q.value.toLowerCase().split(/\s+/).filter(Boolean);
    const sig = document.getElementById("f-sig").checked, ver = document.getElementById("f-ver").checked;
    const coh = document.getElementById("f-coh").checked, subj = sel.value;
    const hits = [];
    rows.forEach((r, i) => {
      if (sig && !r[F.sig]) return;
      if (ver && !r[F.verified]) return;
      if (coh && !r[F.coherent]) return;
      if (subj && r[F.subject] !== subj) return;
      if (terms.length && !terms.every((t) => hay[i].includes(t))) return;
      hits.push(r);
    });
    hits.sort((a, b) => b[F.verified] - a[F.verified] || b[F.sig] - a[F.sig] || b[F.coherent] - a[F.coherent]);
    count.textContent = `${hits.length.toLocaleString()} expert directions${hits.length > 200 ? " (showing 200)" : ""}`;
    out.replaceChildren(...hits.slice(0, 200).map((r) => {
      const { layer, expert, pole } = parseId(r[F.id]);
      return el("li", {}, el("button", { onclick: () => openCard(layer, expert, pole) },
        el("span", { class: "id" }, `L${layer} E${expert} ${pole === "+" ? "+" : "−"}`),
        el("span", {}, r[F.name] || "(no label)", " ",
          r[F.verified] ? el("span", { class: "chip ok" }, "verified") : r[F.sig] ? el("span", { class: "chip" }, "significant") : null,
          r[F.subject] && r[F.subject] !== "none" ? el("span", { class: "chip" }, r[F.subject]) : null),
        el("span", { class: "toks" }, clip(r[F.tokens], 48))));
    }));
  };
  let timer;
  q.addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(run, 120); });
  for (const id of ["f-sig", "f-ver", "f-coh", "f-subj"]) document.getElementById(id).addEventListener("change", run);
  run();
}

function layerMap() {
  const F = Object.fromEntries(INDEX.fields.map((f, i) => [f, i]));
  const layers = SUMMARY.moe_layers, cell = 4;
  const agg = new Map();
  for (const r of INDEX.rows) {
    const { layer, expert } = parseId(r[F.id]);
    const k = `${layer}_${expert}`;
    const a = agg.get(k) || { sig: 0, coh: 0, reap: r[F.reap_kept], subj: "", rank: -1, names: [] };
    a.sig = Math.max(a.sig, r[F.sig]);
    a.coh = Math.max(a.coh, r[F.coherent]);
    const rank = 2 * r[F.sig] + r[F.coherent];
    if (r[F.subject] && r[F.subject] !== "none" && rank > a.rank) { a.subj = r[F.subject]; a.rank = rank; }
    if (r[F.name]) a.names.push(r[F.name]);
    agg.set(k, a);
  }
  const canvas = document.getElementById("grid"), ctx = canvas.getContext("2d"), tip = document.getElementById("tip");
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const draw = () => {
    const mode = document.getElementById("mode").value, off = css("--cell-off"), on = css("--cell-on"), miss = css("--bg");
    layers.forEach((L, i) => {
      for (let e = 0; e < 256; e++) {
        const a = agg.get(`${L}_${e}`);
        let c = miss;
        if (a) c = mode === "sig" ? (a.sig ? on : off) : mode === "coh" ? (a.coh ? on : off)
          : mode === "reap" ? (a.reap ? on : off) : (SUBJECT_COLORS[a.subj] || off);
        ctx.fillStyle = c;
        ctx.fillRect(e * cell, i * cell, cell - (cell > 3 ? 0.5 : 0), cell - 0.5);
      }
    });
    const legend = document.getElementById("legend");
    const sw = (color, label) => el("span", {}, el("i", { style: `background:${color}` }), label);
    legend.replaceChildren(...(mode === "subj"
      ? Object.entries(SUBJECT_COLORS).map(([s, c]) => sw(c, s))
      : [sw(on, { sig: "significant (either pole)", coh: "coherent label (either pole)", reap: "kept by REAP" }[mode]),
         sw(off, "no"), sw(miss, "no sweep data")]));
  };
  const at = (ev) => {
    const r = canvas.getBoundingClientRect();
    const e = Math.floor(((ev.clientX - r.left) / r.width) * 256), i = Math.floor(((ev.clientY - r.top) / r.height) * layers.length);
    return e >= 0 && e < 256 && i >= 0 && i < layers.length ? { layer: layers[i], expert: e, x: ev.clientX - r.left, y: ev.clientY - r.top } : null;
  };
  canvas.addEventListener("mousemove", (ev) => {
    const p = at(ev);
    if (!p) { tip.hidden = true; return; }
    const a = agg.get(`${p.layer}_${p.expert}`);
    tip.replaceChildren(el("b", {}, `L${p.layer} E${p.expert}`), el("br"), a ? a.names.join(" / ") || "(no label)" : "no sweep data");
    tip.hidden = false;
    tip.style.left = Math.min(p.x + 12, canvas.clientWidth - 200) + "px";
    tip.style.top = p.y + 14 + "px";
  });
  canvas.addEventListener("mouseleave", () => { tip.hidden = true; });
  canvas.addEventListener("click", (ev) => { const p = at(ev); if (p) openCard(p.layer, p.expert); });
  document.getElementById("mode").addEventListener("change", draw);
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);
  draw();
}

// ---- expert card -----------------------------------------------------------------------------------------------
async function layerData(layer) {
  if (!layerCache.has(layer)) layerCache.set(layer, getJSON(`L${layer}.json`));
  return layerCache.get(layer);
}

function poleView(p, sign) {
  if (!p) return el("div", { class: "pole" }, el("h4", {}, `${sign === "+" ? "+" : "−"} pole`), el("p", { class: "note" }, "No sweep data."));
  const verdict = p.judge === null ? null : el("span", { class: `chip ${p.judge ? "ok" : "bad"}` }, p.judge ? "judge matched" : "judge missed");
  const sigChip = p.null === "untestable" ? el("span", { class: "chip" }, "no fair control")
    : el("span", { class: `chip ${p.sig ? "ok" : ""}` }, p.sig ? "significant" : "not significant");
  const prompts = [...new Set(p.out.map((o) => o.prompt))];
  const view = el("div");
  const tabs = el("div", { class: "tabs", role: "tablist" });
  const show = (prompt) => {
    for (const b of tabs.children) b.setAttribute("aria-selected", b.dataset.p === prompt);
    const base = BASELINE[prompt];
    const block = (label, o) => el("div", { class: "out" }, el("span", { class: "lab" }, label),
      o.analysis ? el("details", {}, el("summary", {}, "thinking"), el("div", {}, o.analysis)) : null, o.final || "(no final answer)");
    view.replaceChildren(
      el("p", { class: "note", style: "margin:4px 0" }, `Prompt: “${PROMPTS[prompt]}”`),
      base ? block("unsteered", { analysis: clip(base.analysis, 360), final: clip(base.final, 480) }) : null,
      ...p.out.filter((o) => o.prompt === prompt).map((o) => block(`${DOSE_LABEL[String(o.dose)] || o.dose} (${o.dose > 0 ? "+" : "−"})${o.finish === "length" ? " · cut at token limit" : ""}`, o)));
  };
  prompts.forEach((pr) => tabs.append(el("button", { role: "tab", "data-p": pr, onclick: () => show(pr) }, pr)));
  if (prompts.length) show(prompts[0]);
  return el("div", { class: "pole" },
    el("h4", {}, `${sign === "+" ? "+" : "−"} pole: ${p.name || (sign === "-" ? "pushed the other way" : "(no label)")}`),
    p.desc ? el("p", { class: "desc" }, p.desc)
      : sign === "-" ? el("p", { class: "desc" }, "Beam labelled the + side of each direction only. These are the tokens the − push promotes, read from the weights.") : null,
    el("div", {}, p.subject && p.subject !== "none" ? el("span", { class: "chip" }, p.subject) : null,
      p.coherent === null ? null : el("span", { class: `chip ${p.coherent ? "ok" : ""}` }, p.coherent ? "Beam: coherent" : "Beam: incoherent"),
      sigChip, verdict),
    el("div", { class: "stats" }, `above ${p.pct === null ? "–" : pct(p.pct)} of matched controls · ΔNLL ${p.dnll.toFixed(2)} nats/token`),
    el("div", {}, el("span", { class: "note" }, "top tokens "), p.tokens.map((t) => el("span", { class: "chip" }, t))),
    el("div", { class: "stats" }, `theme of steered outputs: ${p.theme.join(", ")}`),
    tabs, view);
}

async function openCard(layer, expert, pole) {
  const dlg = document.getElementById("card"), body = document.getElementById("card-body");
  body.replaceChildren(el("p", { class: "loading" }, `Loading layer ${layer}…`));
  if (!dlg.open) dlg.showModal();
  history.replaceState(null, "", `#L${layer}-E${expert}`);
  try {
    const e = (await layerData(layer)).find((x) => x.expert === expert);
    if (!e) { body.replaceChildren(el("h3", {}, `Layer ${layer} · expert ${expert}`), el("p", {}, "No sweep data for this expert.")); return; }
    const flags = [e.reap_kept ? "kept by REAP" : "pruned by REAP",
      ...e.banned_in.filter((a) => a !== "reap").map((a) => `in the ${a.replace("_ctrl", " control")} ban`),
      e.effort_rd !== null ? `effort expert (selection ${fmt(100 * e.effort_rd, 0)} pts at effort 512 vs 8)` : null].filter(Boolean);
    const order = pole === "-" ? ["-", "+"] : ["+", "-"];
    body.replaceChildren(
      el("h3", {}, `Layer ${layer} · expert ${expert}`),
      el("div", {}, flags.map((f) => el("span", { class: "chip" }, f))),
      el("div", { class: "poles" }, order.map((s) => poleView(e.poles[s], s))));
  } catch (err) {
    body.replaceChildren(el("p", {}, `Could not load layer ${layer}: ${err.message}`));
  }
}

// ---- boot ------------------------------------------------------------------------------------------------------
async function main() {
  let G;
  try {
    [SUMMARY, INDEX, PROMPTS, BASELINE, G] = await Promise.all(
      ["summary.json", "index.json", "prompts.json", "baseline.json", "gallery.json"].map(getJSON));
  } catch (err) {
    document.querySelector("main").prepend(el("div", { class: "panel" }, `Could not load the atlas data (${err.message}).`));
    return;
  }
  heatmap(); cards(); depth(); reap(); effort(); gallery(G); search(); layerMap();
  document.getElementById("card").addEventListener("close", () => history.replaceState(null, "", location.pathname + location.search));
  const m = /^#L(\d+)-E(\d+)$/.exec(location.hash);
  if (m) openCard(+m[1], +m[2]);
}
main();
