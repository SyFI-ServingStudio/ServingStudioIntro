/* Data access for the Models page.

   Everything comes from ServingStudio Sim's read-only public API, through the
   same origin and proxy as the kernel library (kernelData.js):
   /archs lists the archs, /archs/{arch} gives one arch's params and the
   parameter sets its #[supported] rows allow, and /archs/{arch}/cost-tree
   gives one set's cost tree. A tree is structure only: no timings. */

import { load } from "../kernels/kernelData";

export const loadArchs = () => load("archs");
export const loadArch = (arch) => load(`archs/${encodeURIComponent(arch)}`);

// A set's query names it exactly: the GPU, the model and every param the
// arch's rows choose. The service rejects anything it would have to guess.
export const queryString = (query) =>
  new URLSearchParams(
    Object.entries(query).map(([key, value]) => [key, String(value)]),
  ).toString();
export const loadCostTree = (arch, query) =>
  load(`archs/${encodeURIComponent(arch)}/cost-tree?${queryString(query)}`);

/* The page keeps its own state in the query string: `arch`, then the set's
   query under the same names the API uses. A value arrives as text; a set's
   query holds numbers and booleans, so they are compared as text. */
export function memberFromUrl(arch, url) {
  const members = arch.param_sets.flatMap((set) =>
    set.members.map((member) => ({ set, member })),
  );
  return (
    members.find(({ member }) =>
      Object.entries(member.query).every(
        ([key, value]) => url[key] === String(value),
      ),
    ) ?? null
  );
}

/* A param's value as a reader reads it: numbers grouped, booleans as words. */
export const paramValue = (value) =>
  typeof value === "number"
    ? value.toLocaleString("en-US")
    : typeof value === "boolean"
      ? value
        ? "on"
        : "off"
      : String(value);

/* A routing as a reader reads it: its kind and, for a measured routing, the
   file it was measured into. A hub reference keeps its repository and path
   but shortens the commit, which the title gives whole. */
export const shortReference = (reference) =>
  reference.replace(/@([0-9a-f]{7})[0-9a-f]+\//, "@$1/");

export function routingText(value, routing) {
  const kind = value.routing ?? "not recorded";
  const file = Object.entries(value).find(
    ([name]) => name !== "routing" && name !== "routing_seed",
  )?.[1];
  const seed = value.routing_seed != null ? `, seed ${value.routing_seed}` : "";
  const label = routing?.label ?? file;
  return label && label !== kind
    ? `${kind} ${shortReference(label)}${seed}`
    : `${kind}${seed}`;
}

/* How one registry source recorded a run: a preset, an alignment case of a
   pack, or a prediction config. A path another machine had is named by its
   file name only. */
export function sourceText(source) {
  const where = source.path ?? `${source.name} (recorded on another machine)`;
  if (source.kind === "alignment") {
    const cases = source.cases.map((name) => name.split("_")[0]).join(", ");
    return `the alignment pack ${where}, ${source.variant}${cases ? `, cases ${cases}` : ""}`;
  }
  if (source.kind === "timing_predict") return `the prediction config ${where}`;
  return where;
}

/* What each contract means for a reader. An iter-wise arch is one model that
   costs a whole iteration; the layer-wise pair is the two sides of an
   attention-FFN disaggregated deployment. */
export const CONTRACTS = {
  iter_wise: "Whole iteration",
  layer_wise_attn: "Disaggregated attention side",
  layer_wise_ffn: "Disaggregated FFN side",
};

/* Cost-tree nodes, from the API's nested form to rows the tree draws.

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
  if (node.kind === "leaf") return `L:${node.slot.config_key}`;
  const own =
    node.kind === "scale" ? node.n : node.kind === "max" ? node.overlap : "";
  return `${node.kind}${own}(${node.children.map(signature).join(",")})`;
}

/* The tree as nodes the view walks: each with its display name, note, depth,
   and for a Max whose children are alike, how many there are (`copies`) with
   only the first kept. Leaves count every kernel they stand for. */
export function prepareTree(root) {
  const walk = (node, parentPath, depth) => {
    const path = node.kind === "leaf" ? node.slot.name : node.path;
    const { title, note } = splitLabel(node.label, path);
    const base = {
      id: node.id,
      kind: node.kind,
      depth,
      name: title ?? relative(path, parentPath) ?? node.kind,
      path,
      note,
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
    const next = path ?? parentPath;
    const kids = children.map((child) => walk(child, next, depth + 1));
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

/* A leaf's config: whether profile.db's registry holds it on this GPU and how
   much of its grid is measured, by the best-covered backend. */
export function coverage(config) {
  const registry = config?.registry;
  if (!registry) return { state: "unregistered" };
  const runnable = registry.cells - registry.infeasible;
  const best = Math.max(0, ...Object.values(registry.measured));
  if (!best) return { state: "unmeasured", runnable };
  return { state: best >= runnable ? "measured" : "partial", best, runnable };
}

/* The kernel page for a leaf, open on the simulator grid at the leaf's config
   when the registry holds it there. Only documented kinds have a page. A tree
   keys a config by `config_key` (kind and hash): two kinds of one shape share
   a hash. */
export function kernelLink(tree, slot) {
  const kernel = tree.kernels[slot.kind];
  if (!kernel?.documented) return null;
  const params = new URLSearchParams({ kind: slot.kind });
  if (tree.configs[slot.config_key]?.registry) {
    params.set("view", "grid");
    params.set("cgpu", tree.gpu);
    params.set("cmodel", tree.model_config);
    params.set("config", slot.config_hash);
  }
  return `${import.meta.env.BASE_URL}kernels.html?${params}`;
}

/* Prose from the catalog marks identifiers in backticks, as the Rust docs it
   comes from do: `tp_size`. They read as code on the page. */
export function withCode(text) {
  return text
    .split("`")
    .map((part, index) => (index % 2 ? { code: part } : { text: part }));
}
