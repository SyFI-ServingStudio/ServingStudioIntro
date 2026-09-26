/* Data access for the kernel library.

   The prototype reads a static snapshot exported by
   scripts/kernel-library/export.py into public/kernel-library/v1/. The planned
   read-only API serves the same documents under the same paths, so switching
   is a change to DATA_BASE only. */

export const DATA_BASE = `${import.meta.env.BASE_URL}kernel-library/v1`;
export const API_DISPLAY_BASE = "https://<api-host>/api/v1";

const cache = new Map();
function load(path) {
  if (!cache.has(path)) {
    cache.set(
      path,
      fetch(`${DATA_BASE}/${path}`).then((response) => {
        if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
        return response.json();
      }),
    );
  }
  return cache.get(path);
}

export const loadCatalog = () => load("catalog.json");
export const loadKernel = (kind) => load(`kernels/${kind}.json`);
// MOCK: which model deployment asks for each shape. A stand-in until the
// data path is designed; a kernel without it just lists its shapes.
export const loadShapeSources = (kind) =>
  load("mock/shape-sources.json").then(
    (data) => data.kinds[kind] ?? [],
    () => [],
  );

/* Rows arrive column-oriented; expand them once into objects. */
export function expandRows(kernel) {
  const { columns, rows, provenance } = kernel;
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

export const METRIC_LABELS = {
  time_ms: { label: "Time", unit: "ms" },
  tflops: { label: "Throughput", unit: "TFLOPS" },
  memory_bandwidth_gbps: { label: "Memory bandwidth", unit: "GB/s" },
  algbw_gbps: { label: "Algorithm bandwidth", unit: "GB/s" },
  busbw_gbps: { label: "Bus bandwidth", unit: "GB/s" },
  energy_j: { label: "Energy", unit: "J" },
};

export function formatNumber(value, digits = 3) {
  if (value == null) return "not measured";
  if (value === 0) return "0";
  const magnitude = Math.abs(value);
  if (magnitude >= 1000) return Math.round(value).toLocaleString("en-US");
  if (magnitude >= 1)
    return Number(value.toPrecision(digits + 1)).toLocaleString("en-US");
  return Number(value.toPrecision(digits)).toString();
}

export function formatArg(name, value) {
  if (typeof value !== "number") return String(value);
  if (name.endsWith("_bytes")) return formatBytes(value);
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

/* The models a kernel is used by, grouped by family in catalog order:
   [{ family: "Qwen", names: ["Qwen3.6", "Qwen3", "Qwen3-MoE"] }, ...]. */
export function groupModels(usedBy, models) {
  const groups = [];
  for (const { name, family } of models) {
    if (!usedBy.includes(name)) continue;
    const group = groups.find((g) => g.family === family);
    if (group) group.names.push(name);
    else groups.push({ family, names: [name] });
  }
  return groups;
}
