/* Data access for the Models page.

   Everything comes from ServingStudio Sim's public API, through the
   same origin and proxy as the kernel library (kernelData.js):
   /models lists every checkpoint with its public presets, their axes and
   members; /models/{checkpoint}/{arch}/tree gives one member's kernels by
   section and slot; POST /predict times a reader's batch on one member and
   gives each section's cost tree. */

import { load, presetCheckpoint, send } from "../kernels/kernelData";
import { pageHref } from "../../url";

export const loadModels = () => load("models");

/* A public preset as a reader names its deployment ("vLLM, TP and EP"): the
   arch's `arch_name`, else its arch tag. `archNames` maps every preset of
   /models (absent: none) to it, by id. */
export const archLabel = (preset) => preset.arch_name ?? preset.arch;
export const archNames = (models) =>
  new Map(
    (models?.checkpoints ?? []).flatMap((c) =>
      c.presets.map((p) => [p.id, archLabel(p)]),
    ),
  );

/* A member is named by its preset's id ("<checkpoint>/<arch>") and one value
   per axis of the preset. The service rejects anything it would have to guess. */
const queryString = (params) =>
  new URLSearchParams(
    Object.entries(params).map(([key, value]) => [key, String(value)]),
  ).toString();
const presetPath = (preset) => preset.split("/").map(encodeURIComponent).join("/");
export const loadTree = (preset, params) => {
  const query = queryString(params);
  return load(`models/${presetPath(preset)}/tree${query ? `?${query}` : ""}`);
};

/* The Models page at a member: the preset's id, then a value per axis. A
   plain link, so the Kernels page can link here too. */
export const memberHref = (preset, params) =>
  pageHref("models", { preset, ...params });

/* Each case's time on one member. Not cached: a reader edits the batch.
   `signal` cancels a request a newer edit made stale. */
export const predict = (body, signal) =>
  send("predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });

/* ---------- Members ---------- */

/* Axis values arrive from the URL as text; a member's params hold numbers
   and booleans, so they are compared as text, as the service compares them. */
export const sameValue = (a, b) => String(a) === String(b);

export const memberMatches = (member, params, names) =>
  names.every((name) => sameValue(member.params[name], params[name]));

/* What the service says blocks a configuration, in a reader's words: its
   build's `error`, or the measurements it lacks (`missing`, {kind: rows});
   `notYet` begins the latter ("Cannot be timed yet"). The service's "not
   checked", and a missing `missing`, mean the check has not run. Null when
   nothing blocks. Models and Simulate both say it this way. */
export function blockerText({ error, missing }, notYet) {
  if (error === "not checked" || (!error && missing == null))
    return "Its measurements are not checked yet";
  if (error) return `The simulator cannot build it: ${error}`;
  if (Object.keys(missing).length)
    return `${notYet}; measurements missing for ${missingText(missing)}`;
  return null;
}

/* Why a member cannot be predicted, or null when it can: the simulator
   built it and every kernel it calls is measured. */
export const blockedReason = (member) => blockerText(member, "Cannot be timed yet");
export const predictable = (member) => !blockedReason(member);

/* The member a link names; one it does not fully name opens on the first
   `usable` member that has the values it gives, and says so (`unmatched`). */
export function memberFromQuery(preset, query, usable) {
  const names = preset.axes.map((axis) => axis.name);
  const named = preset.members.find((m) => memberMatches(m, query, names));
  if (named) return { member: named };
  const given = names.filter((name) => query[name] != null);
  const fits = preset.members.filter((m) => memberMatches(m, query, given));
  const pool = fits.length ? fits : preset.members;
  return {
    member: pool.find(usable) ?? pool[0],
    unmatched: given.length > 0,
  };
}

/* The member a picked value moves to: of those that have it, the one that
   keeps the most of the other current values, earlier axes weighing more. */
export function closestMember(members, names, current, name, value) {
  const score = (member) =>
    names.reduce(
      (sum, other, index) =>
        other !== name && sameValue(member.params[other], current[other])
          ? sum + 2 ** (names.length - index)
          : sum,
      0,
    );
  const candidates = members.filter((m) => sameValue(m.params[name], value));
  return candidates.reduce(
    (best, m) => (best == null || score(m) > score(best) ? m : best),
    null,
  );
}

/* "ep_size 4, max_model_len 8,192, workload c32_long". */
export const paramsText = (params) =>
  Object.entries(params)
    .map(([name, value]) => `${name} ${paramValue(value)}`)
    .join(", ");

/* "grouped_gemm 998, moe_finalize_routing 50": the rows a member lacks. */
const missingText = (missing) =>
  Object.entries(missing)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([kind, count]) => `${kind} ${count.toLocaleString("en-US")}`)
    .join(", ");

/* A param's value as a reader reads it: numbers grouped, booleans as words. */
export const paramValue = (value) =>
  typeof value === "number"
    ? value.toLocaleString("en-US")
    : typeof value === "boolean"
      ? value
        ? "on"
        : "off"
      : String(value);

/* A hub reference keeps its repository and path but shortens the commit,
   which a title gives whole. */
export const shortReference = (reference) =>
  String(reference).replace(/@([0-9a-f]{7})[0-9a-f]+\//, "@$1/");

/* What each contract means for a reader. An iter-wise arch is one model that
   costs a whole iteration; the layer-wise pair is the two sides of an
   attention-FFN disaggregated deployment. */
export const CONTRACTS = {
  iter_wise: "Whole iteration",
  layer_wise_attn: "Disaggregated attention side",
  layer_wise_ffn: "Disaggregated FFN side",
};

/* ---------- The cost tree ---------- */

/* A predicted section's tree is the Analyzer's (`nodes`: root first, in
   display order, each with its `depth`); the page only folds it. A row is a
   parent when the next row is deeper. */
const isParent = (nodes, index) => nodes[index + 1]?.depth > nodes[index].depth;

/* The rows a reader sees, each parent marked, with only the parents in
   `open` (by `node`) expanded. */
export function visibleRows(nodes, open) {
  const rows = [];
  let folded = Infinity;
  nodes.forEach((node, index) => {
    if (node.depth > folded) return;
    folded = Infinity;
    const parent = isParent(nodes, index);
    rows.push({ ...node, parent });
    if (parent && !open.has(node.node)) folded = node.depth;
  });
  return rows;
}

// Open to this depth at first: the model, its sections and their blocks.
export const parents = (nodes, depth = Infinity) =>
  nodes
    .filter((node, index) => node.depth < depth && isParent(nodes, index))
    .map((node) => node.node);

/* The kernel page for a leaf, open on the simulator grid at the leaf's
   config: `config` is the id /kernels/{kind}/configs/{id} answers. Only
   documented kinds have a page. */
export function kernelLink(kernel, slot, tree) {
  if (!kernel?.documented) return null;
  return pageHref("kernels", {
    kind: slot.kernel,
    view: "grid",
    cgpu: tree.gpu,
    cmodel: presetCheckpoint(tree.preset),
    config: slot.config,
  });
}

/* A section of the cost tree as a reader reads its name ("lm_head" → "lm head"). */
export const sectionName = (section) => section.replaceAll("_", " ");

/* A share in percent as the page prints it. */
export const formatPercent = (pct) =>
  pct > 0 && pct < 0.1 ? "<0.1%" : `${pct.toFixed(1)}%`;

/* A time in milliseconds as the page prints it. */
export const formatMs = (ms) =>
  ms > 0 && ms < 0.001
    ? "<0.001"
    : ms.toLocaleString("en-US", {
        minimumFractionDigits: ms >= 100 ? 1 : 3,
        maximumFractionDigits: ms >= 100 ? 1 : 3,
      });
