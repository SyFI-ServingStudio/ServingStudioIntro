import test from "node:test";
import assert from "node:assert/strict";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  fetchCatalogs,
  injectStaticContent,
  readSkills,
  renderCatalogPage,
  renderRobotsText,
} from "./agent-content.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const origin = "https://example.org";
const links = {
  origin,
  api: `${origin}/api/public/v1`,
  apiPath: "/api/public/v1",
  page: (id, params = {}) => {
    const query = new URLSearchParams(params).toString();
    return `${origin}/${id}.html${query ? `?${query}` : ""}`;
  },
  skill: (name) => `${origin}/skills/${name}.md`,
};

// The smallest answers of each catalog route the renderers read.
const catalogs = {
  health: { status: "ok", sim_commit: "abc123" },
  models: {
    checkpoints: [
      {
        checkpoint: "meta-llama/Llama-3.1-8B",
        name: "Llama 3.1 8B",
        family: "Llama",
        presets: [
          {
            id: "Llama-3.1-8B/llama3_dense_tp",
            arch: "llama3_dense_tp",
            arch_name: "TP",
            gpu: "NVIDIA H200",
            axes: [{ name: "tp_size", values: [1, 2] }],
            members: [
              { params: { tp_size: 1 }, error: null, missing: {} },
              { params: { tp_size: 2 }, error: null, missing: { grouped_gemm: 3 } },
            ],
          },
        ],
      },
    ],
  },
  kernels: {
    categories: ["GEMM", "Communication"],
    gpus: [{ name: "NVIDIA H200" }],
    snapshot: { last_measured_at: "2026-11-06T17:17:20+00:00" },
    kernels: [
      {
        kind: "all_reduce",
        documented: true,
        title: "All-reduce",
        summary: "Sum one buffer across the GPUs of a tensor-parallel group.",
        category: "Communication",
        subcategory: null,
        backends: ["nccl"],
        precisions: [],
        coverage: [
          { gpu: "NVIDIA H200", backend: "nccl", precision: null, rows: 57 },
        ],
        rows: 57,
      },
    ],
  },
  simulationPresets: {
    limits: { max_running: 2, max_queued: 32, max_requests: 2000, timeout_s: 600 },
    presets: [
      {
        id: "Llama-3.1-8B/llama3_dense_tp_barebone",
        checkpoint: "meta-llama/Llama-3.1-8B",
        deployment: "unified",
        pools: {
          main: {
            arch_preset: "Llama-3.1-8B/llama3_dense_tp",
            arch: "llama3_dense_tp",
            gpu: "NVIDIA H200",
            worker: "barebone",
          },
        },
        axes: [{ name: "replicas", values: [1, 2] }],
        captures: [
          {
            name: "c32_long",
            facts: { requests: 64, prompt_tokens: 1856.4, output_tokens: 192 },
          },
        ],
        members: [{}, {}],
      },
    ],
  },
  workloads: {
    routing: "An MoE member routes as its capture does.",
    sources: {
      capture: { summary: "A recorded trace." },
      generated: {
        summary: "A trace `tracegen` draws.",
        generators: [
          {
            name: "synthetic",
            about: "Draw a session workload",
            arguments: [
              {
                name: "seed",
                default: "0",
                choices: [],
                help: "Seed for every draw",
              },
            ],
          },
        ],
      },
      upload: {
        summary: "A CSV you send.",
        max_bytes: 1048576,
        keep_s: 86400,
        formats: [
          { name: "text-generation-independent", columns: ["id"], tags: ["slo"] },
        ],
        tags: [{ name: "slo", columns: ["ttft_slo_ms"] }],
      },
    },
  },
};

const page = (id) => ({
  id,
  title: id,
  description: `The ${id} page.`,
  skill: "servingstudio-simulate",
  render: () => "",
});

test("only the catalog pages get static text; without catalogs it says why", async () => {
  const module = await import("./agent-content.mjs");
  const plugin = module.agentContent({
    origin,
    pages: [
      { id: "models", name: "Models", file: "models.html" },
      { id: "kernels", name: "Kernels", file: "kernels.html" },
      { id: "simulate", name: "Simulate", file: "simulate.html" },
    ],
  });
  plugin.configResolved({ base: "/", root });
  const html = '<html><head></head><body><div id="root"></div></body></html>';
  const render = (file) =>
    plugin.transformIndexHtml.handler(html, { filename: path.join(root, file) });

  // Without catalogs the page says why its lists are missing.
  assert.match(render("models.html"), /without the API&#x27;s catalogs/);
  assert.equal(render("index.html"), html);
});

test("renderCatalogPage writes models, kernels and simulations", () => {
  const models = renderCatalogPage(
    { ...page("models"), render: (data) => data.models.checkpoints[0].name },
    catalogs,
    links,
  );
  assert.match(models, /^# models \| ServingStudio\n/);
  assert.match(models, /Sim commit `abc123`/);
  assert.match(models, /Llama 3\.1 8B/);
  assert.match(
    models,
    /\[servingstudio-simulate\]\(https:\/\/example\.org\/skills/,
  );

  const missing = renderCatalogPage(page("models"), null, links);
  assert.match(missing, /ALLOW_NO_API_CATALOG=1/);
});

test("the real page bodies name every entry", async () => {
  const module = await import("./agent-content.mjs");
  const plugin = module.agentContent({
    origin,
    pages: [
      { id: "models", name: "Models", file: "models.html" },
      { id: "kernels", name: "Kernels", file: "kernels.html" },
      { id: "simulate", name: "Simulate", file: "simulate.html" },
    ],
    fetchCatalogs: async () => catalogs,
  });
  plugin.configResolved({ base: "/", root });
  await plugin.buildStart();
  const emitted = {};
  plugin.generateBundle.call({
    emitFile: ({ fileName, source }) => (emitted[fileName] = source),
  });

  assert.match(
    emitted["models.md"],
    /- \[TP\]\(https:\/\/example\.org\/models\.html\?preset=Llama-3\.1-8B%2Fllama3_dense_tp\): `Llama-3\.1-8B\/llama3_dense_tp`; NVIDIA H200; sweeps `tp_size` 1, 2; 1 of 2 configurations can be timed/,
  );
  assert.match(
    emitted["kernels.md"],
    /measured on NVIDIA H200, last measured 2026-11-06/,
  );
  assert.match(
    emitted["kernels.md"],
    /## Communication\n\n- \[All-reduce\]\(https:\/\/example\.org\/kernels\.html\?kind=all_reduce\) \(`all_reduce`\): Sum one buffer .*backends: `nccl`; GPUs: NVIDIA H200; 57 measured rows\./,
  );
  assert.doesNotMatch(emitted["kernels.md"], /## GEMM/);
  assert.match(
    emitted["simulate.md"],
    /main: TP on NVIDIA H200, worker `barebone`; sweeps `replicas` 1, 2; 2 configurations; captures: `c32_long` \(64 requests, mean 1,856 prompt and 192 output tokens\)/,
  );
  assert.match(
    emitted["simulate.md"],
    /- `seed` \(default `0`\): Seed for every draw/,
  );
  assert.match(emitted["llms.txt"], /^# ServingStudio\n\n> /);
  assert.match(
    emitted["llms.txt"],
    /\[Models\]\(https:\/\/example\.org\/models\.md\)/,
  );
  assert.match(emitted["llms.txt"], /api\/public\/v1\/openapi\.json/);
  assert.ok(emitted["skills/servingstudio-simulate.md"].startsWith("---\nname:"));
  assert.match(
    emitted["robots.txt"],
    /Sitemap: https:\/\/example\.org\/sitemap\.xml/,
  );
});

test("fetchCatalogs fails on an unreachable route or an empty list", async () => {
  const answers = {
    health: catalogs.health,
    models: catalogs.models,
    kernels: catalogs.kernels,
    "simulations/presets": catalogs.simulationPresets,
    workloads: catalogs.workloads,
  };
  const fetchFrom = (overrides) => async (url) => {
    const route = url.replace("https://api.test/", "");
    if (overrides[route] === "down") throw new TypeError("fetch failed");
    return {
      ok: true,
      status: 200,
      json: async () => overrides[route] ?? answers[route],
    };
  };

  const fetched = await fetchCatalogs("https://api.test", fetchFrom({}));
  assert.equal(fetched.simulationPresets.presets.length, 1);
  await assert.rejects(
    fetchCatalogs("https://api.test", fetchFrom({ kernels: "down" })),
    /https:\/\/api\.test\/kernels: fetch failed/,
  );
  await assert.rejects(
    fetchCatalogs("https://api.test", fetchFrom({ models: { checkpoints: [] } })),
    /models answered an empty list/,
  );
});

test("the plugin stops a build without catalogs unless allowed", async () => {
  const { agentContent } = await import("./agent-content.mjs");
  const unreachable = async () => {
    throw new Error("unreachable");
  };
  const quiet = { log() {}, warn() {} };
  const strict = agentContent({
    origin,
    pages: [],
    fetchCatalogs: unreachable,
    log: quiet,
  });
  strict.configResolved({ base: "/", root });
  await assert.rejects(strict.buildStart(), /ALLOW_NO_API_CATALOG=1/);

  const lenient = agentContent({
    origin,
    pages: [],
    fetchCatalogs: unreachable,
    allowMissing: true,
    log: quiet,
  });
  lenient.configResolved({ base: "/", root });
  await lenient.buildStart();
});

test("injectStaticContent puts the text in #root and hides it from scripted browsers", () => {
  const html = injectStaticContent(
    '<html><head><title>x</title></head><body><div id="root"></div></body></html>',
    { markdown: "# Title\n\n- [Item](https://example.org/)", markdownUrl: "/x.md" },
  );
  assert.match(
    html,
    /<link rel="alternate" type="text\/markdown" href="\/x\.md" \/>/,
  );
  assert.match(
    html,
    /@media \(scripting: enabled\) \{ \.static-page \{ display: none; \} \}/,
  );
  assert.match(
    html,
    /<div id="root"><main class="static-page wrap"><h1>Title<\/h1>\n<ul>\n<li><a href="https:\/\/example\.org\/">Item<\/a><\/li>/,
  );
});

test("readSkills reads every published skill with its description", () => {
  const skills = readSkills(root);
  assert.deepEqual(
    skills.map((skill) => skill.name),
    [
      "servingstudio-kernel-performance",
      "servingstudio-simulate",
      "servingstudio-timing-predict",
    ],
  );
  for (const skill of skills) assert.ok(skill.description.length > 40, skill.name);
});

test("robots.txt allows everything and names llms.txt and the sitemap", () => {
  assert.equal(
    renderRobotsText(origin),
    `# Every crawler is welcome. For a plain-text guide to the site, its
# Markdown pages, the public API and the agent skills, read https://example.org/llms.txt
User-agent: *
Allow: /

Sitemap: https://example.org/sitemap.xml
`,
  );
});
