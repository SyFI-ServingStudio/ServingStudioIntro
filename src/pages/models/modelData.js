/* Data access for the Models page.

   Everything comes from ServingStudio Sim's read-only public API, through the
   same origin and proxy as the kernel library (kernelData.js):
   /models lists every checkpoint with its public presets, their axes and
   members; /models/{checkpoint}/{arch}/tree gives one member's cost tree,
   structure only; POST /predict times a reader's batch on one member. */

import { API_BASE, load, responseError } from "../kernels/kernelData";

export const loadModels = () => load("models");

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

/* The Models page, at a member when given one: the preset's id, then a
   value per axis. Plain links, so the Kernels page can link here too. */
export function modelHref(params) {
  const text = new URLSearchParams(params).toString();
  return `${import.meta.env.BASE_URL}models.html${text ? `?${text}` : ""}`;
}
export const memberHref = (preset, params) => modelHref({ preset, ...params });

/* Each case's time on one member. Not cached: a reader edits the batch.
   `signal` cancels a request a newer edit made stale. */
export async function predict(body, signal) {
  const response = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok) throw await responseError("predict", response);
  return response.json();
}

/* ---------- Members ---------- */

/* Axis values arrive from the URL as text; a member's params hold numbers
   and booleans, so they are compared as text, as the service compares them. */
export const sameValue = (a, b) => String(a) === String(b);

export const memberMatches = (member, params, names) =>
  names.every((name) => sameValue(member.params[name], params[name]));

// A member the simulator built and profile.db has every row for.
export const predictable = (member) =>
  !member.error && member.missing != null && !Object.keys(member.missing).length;

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
export const missingText = (missing) =>
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

/* A section's flat nodes as a nested tree. Node `id` is the flat index, the
   index a prediction's `node_ms` uses; a leaf keeps its slot's index too. A
   composite's `path` is the dotted role every leaf under it shares. */
function commonPath(paths) {
  const split = paths.filter(Boolean).map((p) => p.split("."));
  if (!split.length) return null;
  const common = [];
  for (let i = 0; i < Math.min(...split.map((p) => p.length)); i += 1) {
    if (split.some((p) => p[i] !== split[0][i])) break;
    common.push(split[0][i]);
  }
  return common.join(".") || null;
}

export function nestSection({ nodes, slots }) {
  const build = (id) => {
    const node = nodes[id];
    if (node.kind === "leaf") {
      const slot = { index: node.slot, ...slots[node.slot] };
      return { id, kind: "leaf", label: node.label, slot, path: slot.name };
    }
    const children = node.children.map(build);
    return {
      id,
      kind: node.kind,
      label: node.label,
      n: node.n,
      overlap: node.overlap,
      children,
      path: commonPath(children.map((child) => child.path)),
    };
  };
  return build(0);
}

/* Cost-tree nodes, from the nested form to rows the tree draws.

   A composite's name is its role relative to its parent (`attention` under
   `unified.body.dense_full_index`); Rust's composite line, when there is one,
   becomes the note beside it. A leaf is named the same way. */
const relative = (path, parent) => {
  if (!path) return null;
  if (parent && path.startsWith(`${parent}.`)) return path.slice(parent.length + 1);
  return path.split(".").at(-1);
};

// Rust's composite line reads "<path> (<Worklet>) [<partition>]" or a free
// description ("layers 0..2: dense + full index (3 layers)").
function splitLabel(label, path) {
  if (!label) return { title: null, note: null };
  if (path && label.startsWith(path)) {
    const rest = label.slice(path.length).trim();
    return { title: null, note: rest || null };
  }
  if (label.startsWith("unified") || label.startsWith("afd")) {
    const space = label.indexOf(" ");
    return { title: null, note: space < 0 ? null : label.slice(space + 1) };
  }
  return { title: label, note: null };
}

/* Two subtrees are alike when they compose the same kernels the same way: the
   rank copies under a Max usually are, and one of them stands for all. */
function signature(node) {
  if (node.kind === "leaf") return `L:${node.slot.kernel}:${node.slot.config}`;
  const own =
    node.kind === "scale" ? node.n : node.kind === "max" ? node.overlap : "";
  return `${node.kind}${own}(${node.children.map(signature).join(",")})`;
}

/* The tree as nodes the view walks: each with its display name, note, depth,
   and for a Max whose children are alike, how many there are (`copies`) with
   only the first kept. Leaves count every kernel they stand for. */
export function prepareTree(root) {
  // `repeated`: inside a Scale, where a node's time is one repeat's.
  const walk = (node, parentPath, depth, repeated = false) => {
    const { title, note } = splitLabel(node.label, node.path);
    const base = {
      id: node.id,
      kind: node.kind,
      depth,
      name: title ?? relative(node.path, parentPath) ?? node.kind,
      path: node.path,
      note,
      repeated,
    };
    if (node.kind === "leaf") return { ...base, slot: node.slot, leaves: 1 };
    let children = node.children;
    let copies = null;
    if (
      node.kind === "max" &&
      children.length > 1 &&
      children.every((child) => signature(child) === signature(children[0]))
    ) {
      copies = children.length;
      children = [children[0]];
    }
    const next = node.path ?? parentPath;
    const inner = repeated || node.kind === "scale";
    const kids = children.map((child) => walk(child, next, depth + 1, inner));
    return {
      ...base,
      n: node.n,
      overlap: node.overlap,
      copies,
      width: node.children.length,
      children: kids,
      // Every kernel call the node stands for, the copies not shown included.
      leaves: (copies ?? 1) * kids.reduce((sum, kid) => sum + kid.leaves, 0),
    };
  };
  return walk(root, null, 0);
}

export function* visible(node, open) {
  yield node;
  if (node.children && open.has(node.id))
    for (const child of node.children) yield* visible(child, open);
}

export function allIds(node, out = []) {
  if (node.children) {
    out.push(node.id);
    node.children.forEach((child) => allIds(child, out));
  }
  return out;
}

// Open to this depth at first: the model, its sections and their blocks.
export function initiallyOpen(node, depth = 2, out = new Set()) {
  if (node.children && node.depth < depth) {
    out.add(node.id);
    node.children.forEach((child) => initiallyOpen(child, depth, out));
  }
  return out;
}

/* The kernel page for a leaf, open on the simulator grid at the leaf's
   config: `config` is the id /kernels/{kind}/configs/{id} answers. Only
   documented kinds have a page. */
export function kernelLink(kernel, slot, tree) {
  if (!kernel?.documented) return null;
  const params = new URLSearchParams({
    kind: slot.kernel,
    view: "grid",
    cgpu: tree.gpu,
    cmodel: tree.preset.split("/")[0],
    config: slot.config,
  });
  return `${import.meta.env.BASE_URL}kernels.html?${params}`;
}

/* A time in milliseconds as the page prints it. */
export const formatMs = (ms) =>
  ms > 0 && ms < 0.001
    ? "<0.001"
    : ms.toLocaleString("en-US", {
        minimumFractionDigits: ms >= 100 ? 1 : 3,
        maximumFractionDigits: ms >= 100 ? 1 : 3,
      });
