/* Data access for the kernel library.

   Everything comes from ServingStudio Sim's read-only public API. Pages request
   it from their own origin; the server in front of the site forwards
   /api/public/v1 to the service (in development, Vite's proxy; see
   vite.config.js). VITE_PUBLIC_API_BASE overrides the path. */

export const API_BASE = import.meta.env.VITE_PUBLIC_API_BASE ?? "/api/public/v1";
// The same base as an absolute URL, for code a reader copies elsewhere.
export const apiUrl = (path) =>
  new URL(`${API_BASE}/${path}`, window.location.origin).href;

/* One request per path, shared by every page that asks. A failed request
   rejects with the service's message; a detail that is an object (a query the
   service will not guess at, with the valid choices) rides on the error. */
const cache = new Map();
export function load(path) {
  if (!cache.has(path)) {
    cache.set(
      path,
      fetch(`${API_BASE}/${path}`).then(async (response) => {
        if (!response.ok) {
          const detail = await response.json().then(
            (body) => body.detail,
            () => null,
          );
          const message =
            typeof detail === "object" && detail ? detail.message : detail;
          const error = new Error(
            `${API_BASE}/${path}: ${message ?? `HTTP ${response.status}`}`,
          );
          error.status = response.status;
          error.detail = detail;
          throw error;
        }
        return response.json();
      }),
    );
  }
  return cache.get(path);
}

export const loadCatalog = () => load("kernels");
export const loadKernel = (kind) => load(`kernels/${kind}`);
export const loadRows = (kind) => load(`kernels/${kind}/rows`);
// The kernel configs the simulator registered as reading this kind's rows,
// and one config's grid on its cache axes.
export const loadConfigs = (kind) => load(`kernels/${kind}/configs`);
export const loadConfig = (kind, hash, gpu) =>
  load(`kernels/${kind}/configs/${hash}?gpu=${encodeURIComponent(gpu)}`);

/* A kernel's arguments, in profile.db column order, and the role a supported
   model gives each: "sweep" (varies with the batch), "config" (fixed by the
   model) or null when no supported model runs the kernel yet. */
export const argNames = (kernel) => kernel.args.map((arg) => arg.name);
const argInfo = (kernel, name) => kernel.args.find((arg) => arg.name === name);
export const argRole = (kernel, name) => argInfo(kernel, name)?.role ?? null;
// What one step of a numeric argument counts: tokens, bytes, GPUs, ...
export const argUnit = (kernel, name) => argInfo(kernel, name)?.unit ?? null;
// "dtype", "number", "list" or "label", from the argument's declared type.
const argType = (kernel, name) => argInfo(kernel, name)?.type;
export const isNumeric = (kernel, name) => argType(kernel, name) === "number";
// A list-valued argument, such as per-request (queries, context) pairs.
export const isList = (kernel, name) => argType(kernel, name) === "list";
// An element type (bf16, fp8_e4m3, ...): these are picked together as the precision.
export const isDtype = (kernel, name) => argType(kernel, name) === "dtype";
// The argument holding the compute dtype, which picks the throughput peak.
export const precisionArg = (kernel) =>
  kernel.args.find((arg) => arg.precision)?.name ?? null;

/* The metrics a kernel records, with the label and unit Sim gives each. */
export const metricNames = (kernel) => kernel.metrics.map((metric) => metric.name);
export const metricDoc = (kernel, name) =>
  kernel.metrics.find((metric) => metric.name === name);

/* Rows arrive column-oriented; expand them once into objects. */
export function expandRows({ columns, rows, provenance }) {
  return rows.map((row) => {
    const record = {};
    columns.forEach((column, index) => {
      record[column] = row[index];
    });
    record.provenance = provenance[record.provenance];
    return record;
  });
}

export const shortGpu = (name) => name.replace(/^NVIDIA /, "");

/* The spec-sheet ceiling Sim gives this metric on the picked GPU. A throughput
   ceiling depends on the compute dtype, so it shows only with one picked. */
export function peakFor(kernel, y, color, selection, catalog) {
  if (color === "gpu") return null;
  const peak = catalog.gpus.find((g) => g.name === selection.gpu)?.peaks[y];
  if (!peak) return null;
  const dtypeArg = precisionArg(kernel);
  const dtype = peak.by_dtype ? selection[dtypeArg] : null;
  if (peak.by_dtype && (!dtype || color === dtypeArg)) return null;
  const value = peak.by_dtype ? peak.by_dtype[dtype] : peak.value;
  if (value == null) return null;
  const { label, unit } = metricDoc(kernel, y);
  const amount = `${value.toLocaleString("en-US")} ${unit}`;
  const note = [dtype, peak.note].filter(Boolean).join(" ");
  return {
    value,
    label: `${shortGpu(selection.gpu)} spec-sheet ${label.toLowerCase()}, ${note}: ${amount} (not measured)`,
    short: `Spec-sheet ${value.toLocaleString("en-US")}`,
  };
}

export function formatNumber(value, digits = 3) {
  if (value == null) return "not measured";
  if (value === 0) return "0";
  const magnitude = Math.abs(value);
  if (magnitude >= 1000) return Math.round(value).toLocaleString("en-US");
  if (magnitude >= 1)
    return Number(value.toPrecision(digits + 1)).toLocaleString("en-US");
  return Number(value.toPrecision(digits)).toString();
}

/* An argument value in its unit: a byte count reads as "4 MiB". */
// Past this many characters a list value is cut short; the rows keep it whole.
const LIST_LABEL_LIMIT = 32;

/* A list argument arrives as JSON text, sometimes hundreds of entries long
   (one per request or query row). Runs of equal entries are written once with
   a count, "1 ×256" or "(1, 4096) ×2, (64, 8192)", so a label stays short. */
function formatList(text) {
  let items;
  try {
    items = JSON.parse(text);
  } catch {
    return text;
  }
  if (!Array.isArray(items)) return text;
  if (!items.length) return "none";
  const item = (v) =>
    Array.isArray(v)
      ? `(${v.map((x) => x.toLocaleString("en-US")).join(", ")})`
      : typeof v === "number"
        ? v.toLocaleString("en-US")
        : String(v);
  const runs = [];
  for (const v of items) {
    const label = item(v);
    const last = runs.at(-1);
    if (last && last.label === label) last.count += 1;
    else runs.push({ label, count: 1 });
  }
  const parts = runs.map(({ label, count }) =>
    count > 1 ? `${label} ×${count}` : label,
  );
  let out = parts[0];
  for (const part of parts.slice(1)) {
    if (out.length + part.length + 2 > LIST_LABEL_LIMIT) {
      return `${out}, … (${items.length} entries)`;
    }
    out = `${out}, ${part}`;
  }
  return out;
}

export function formatValue(value, unit) {
  if (typeof value === "string" && value.startsWith("[")) return formatList(value);
  if (typeof value !== "number") return String(value);
  if (unit === "bytes") return formatBytes(value);
  return value.toLocaleString("en-US");
}

export function formatBytes(bytes) {
  const units = ["B", "KiB", "MiB", "GiB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${Number(value.toPrecision(3))} ${units[unit]}`;
}

export const formatDate = (iso) =>
  iso
    ? new Date(iso).toLocaleDateString("en-GB", {
        day: "numeric",
        month: "short",
        year: "numeric",
        timeZone: "UTC",
      })
    : "unknown date";

export function toCsv(records, columns) {
  const escape = (value) => {
    if (value == null) return "";
    const text = String(value);
    return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
  };
  const provenanceColumns = [
    "profiler_git_hash",
    "profiler_run_at",
    "cuda_version",
    "driver_version",
    "backend_version",
  ];
  const header = [...columns, ...provenanceColumns];
  const lines = records.map((record) =>
    [
      ...columns.map((column) => escape(record[column])),
      ...provenanceColumns.map((column) => escape(record.provenance[column])),
    ].join(","),
  );
  return [header.join(","), ...lines].join("\n");
}

export function downloadText(filename, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

/* The kernel library keeps its own state in the query string so every view can
   be shared. The site router only knows the path, so these helpers notify
   listeners themselves. */
const listeners = new Set();
export function subscribeUrl(listener) {
  listeners.add(listener);
  window.addEventListener("popstate", listener);
  return () => {
    listeners.delete(listener);
    window.removeEventListener("popstate", listener);
  };
}
export function setQuery(params, { push = false } = {}) {
  const url = new URL(window.location.href);
  url.search = "";
  for (const [key, value] of Object.entries(params)) {
    if (value != null && value !== "") url.searchParams.set(key, value);
  }
  if (url.href === window.location.href) return;
  window.history[push ? "pushState" : "replaceState"](null, "", url);
  listeners.forEach((listener) => listener());
}
export const readQuery = () =>
  Object.fromEntries(new URLSearchParams(window.location.search));

/* A model is keyed by its model config (the file stem Sim names it by); the
   catalog gives most of them a name and a family. One it does not name reads
   as its model config and stands in a family of its own. */
export const modelEntry = (models, stem) =>
  models.find((m) => m.model_config === stem);
export const modelName = (models, stem) => modelEntry(models, stem)?.name ?? stem;
export const modelFamily = (models, stem) =>
  modelEntry(models, stem)?.family ?? modelName(models, stem);

/* The models a kernel is used by, grouped by family in catalog order:
   [{ family: "Qwen", stems: [...], names: ["Qwen3 235B-A22B", ...] }, ...]. */
export function groupModels(usedBy, models) {
  const groups = [];
  for (const { model_config: stem } of models) {
    if (!usedBy.includes(stem)) continue;
    const family = modelFamily(models, stem);
    const name = modelName(models, stem);
    const group = groups.find((g) => g.family === family);
    if (group) {
      group.stems.push(stem);
      group.names.push(name);
    } else groups.push({ family, stems: [stem], names: [name] });
  }
  return groups;
}
