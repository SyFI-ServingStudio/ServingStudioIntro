/* Data access for the Simulate page.

   Everything comes from ServingStudio Sim's public API, through the same
   origin and proxy as the other pages (kernelData.js): /simulations/presets
   lists every deployment a reader can simulate and what each recording holds,
   /workloads the generator and the CSV formats the service takes, POST /simulate queues a run and GET /simulations/{id} reads
   it until it is done. What a trace holds, which columns an upload needs and
   why a deployment cannot run are the service's answers. */

import { load, send } from "../kernels/kernelData";

export const loadSimPresets = () => load("simulations/presets");
export const loadWorkloads = () => load("workloads");

export const startSimulation = (body) =>
  send("simulate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

// Not cached: a run changes until it is done.
export const readSimulation = (id, signal) =>
  send(`simulations/${encodeURIComponent(id)}`, { signal });

export const deleteSimulation = (id) =>
  send(`simulations/${encodeURIComponent(id)}`, { method: "DELETE" });

/* Keep an uploaded trace; the service reads it as the format its header
   fits, and the answer's workload_id names it in a run. */
export const uploadWorkload = (file) =>
  send("workloads", {
    method: "POST",
    headers: { "Content-Type": "text/csv" },
    body: file,
  });

// A run still moving: the page reads it again until it is not.
export const unfinished = (status) => status === "queued" || status === "running";

/* ---------- Names ---------- */

/* How a reader knows a worker type: what it does to a request. A type
   missing here shows its tag. */
const WORKERS = {
  barebone: "Whole-prompt prefill",
  hp_unified: "Whole-prompt prefill, prefix-aware placement",
  chunked_prefill: "Chunked prefill",
  speculative: "Speculative decoding",
};

const DEPLOYMENTS = {
  pd: "Prefill and decode on separate GPUs",
};

/* A sim preset as a reader names it: how its pools are deployed (the arch
   presets' names from /models) and how they schedule requests. */
export function presetName(preset, archNames) {
  const pools = Object.values(preset.pools);
  const archs = [
    ...new Set(pools.map((p) => archNames.get(p.arch_preset) ?? p.arch)),
  ];
  const workers =
    preset.deployment === "unified"
      ? [...new Set(pools.map((p) => WORKERS[p.worker] ?? p.worker))]
      : [DEPLOYMENTS[preset.deployment] ?? preset.deployment];
  return [...workers, ...archs].join(", ");
}

// The pool whose worker drafts tokens, and how many per step.
export const draftTokens = (member) =>
  Object.values(member.pools).find((p) => p.worker.draft_tokens != null)?.worker
    .draft_tokens ?? null;

/* ---------- Saved runs ---------- */

/* The runs this browser started, newest first, so a reader can compare them.
   Only their ids are kept; the service keeps each run a day and forgets it
   after, so a run it no longer has is dropped. A browser that keeps nothing
   (a private window) just shows the runs of this visit. */
const SAVED = "servingstudio.simulations";
const MAX_SAVED = 20;

export function savedRuns() {
  try {
    const ids = JSON.parse(window.localStorage.getItem(SAVED) ?? "[]");
    return Array.isArray(ids) ? ids.filter((id) => typeof id === "string") : [];
  } catch {
    return [];
  }
}

export function saveRuns(ids) {
  try {
    window.localStorage.setItem(SAVED, JSON.stringify(ids.slice(0, MAX_SAVED)));
  } catch {
    // Nothing kept; the list lasts this visit.
  }
}

/* ---------- Numbers ---------- */

/* A latency in milliseconds as the page prints it: milliseconds under a
   second, seconds above, three significant figures. */
export function latency(ms) {
  if (ms == null || !Number.isFinite(ms)) return { value: "-", unit: "" };
  const [value, unit] = ms >= 1000 ? [ms / 1000, "s"] : [ms, "ms"];
  return { value: significant(value), unit };
}

const significant = (value) =>
  value.toLocaleString("en-US", {
    maximumFractionDigits: value >= 100 ? 0 : value >= 10 ? 1 : 2,
  });

export const latencyText = (ms) => {
  const { value, unit } = latency(ms);
  return unit ? `${value} ${unit}` : value;
};

export const count = (value) =>
  value == null ? "-" : Math.round(value).toLocaleString("en-US");

export const rate = (value) => (value == null ? "-" : significant(value));

/* A mean token count: whole tokens, a thousand and up in k. */
export function tokens(value) {
  if (value == null) return "-";
  return value >= 1000 ? `${significant(value / 1000)}k` : count(value);
}

/* Simulated time: seconds, then minutes, then hours. */
export function duration(ms) {
  if (ms == null) return "-";
  const s = ms / 1000;
  if (s < 120) return `${significant(s)} s`;
  if (s < 7200) return `${significant(s / 60)} min`;
  return `${significant(s / 3600)} h`;
}
