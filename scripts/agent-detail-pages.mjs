/* One Markdown page per kernel kind (`kernels/<kind>.md`) and per public
   model deployment (`models/<checkpoint>/<arch>.md`), for crawlers and
   agents. The interactive pages show these details from a query string
   (`kernels.html?kind=…`, `models.html?preset=…`), which a static host
   answers with the list page's bytes, so a reader without JavaScript needs
   its own page per detail.

   Everything comes from the public API, read once per build by
   scripts/agent-content.mjs: `/kernels/{kind}` per kind, and one member's
   `/models/{checkpoint}/{arch}/tree` per deployment. Wording is the API's
   own fields; this file only arranges them. */

import { code, join, plural, predictable, text } from "./markdown-text.mjs";

// Requests in flight at once while reading the per-kind and per-tree routes.
const DETAIL_CONCURRENCY = 6;

/* One route's JSON, or an error naming the URL that failed. */
export async function fetchJson(url, fetchImplementation = fetch) {
  let response;
  try {
    response = await fetchImplementation(url, {
      signal: AbortSignal.timeout(60_000),
    });
  } catch (error) {
    throw new Error(`${url}: ${error.cause?.message ?? error.message}`);
  }
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response.json();
}

async function mapWithConcurrency(items, limit, work) {
  const results = new Array(items.length);
  let next = 0;
  const worker = async () => {
    while (next < items.length) {
      const index = next++;
      results[index] = await work(items[index]);
    }
  };
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker));
  return results;
}

/* The URL path of a preset's routes: each part of "<checkpoint>/<arch>". */
const presetPath = (presetId) =>
  presetId.split("/").map(encodeURIComponent).join("/");
const queryString = (params) =>
  new URLSearchParams(
    Object.entries(params).map(([name, value]) => [name, String(value)]),
  ).toString();

/* The member a deployment page shows the kernels of: the first that can be
   timed, else the first the simulator built, else none. */
export const shownMember = (preset) =>
  preset.members.find(predictable) ??
  preset.members.find((member) => !member.error) ??
  null;

export const treeUrl = (apiBase, presetId, params) => {
  const query = queryString(params);
  return `${apiBase}/models/${presetPath(presetId)}/tree${query ? `?${query}` : ""}`;
};

/* Every documented kind's `/kernels/{kind}` and every deployment's shown
   member's tree. Any failure rejects, naming the URL. */
export async function fetchDetails(apiBase, catalogs, fetchImplementation = fetch) {
  const kinds = catalogs.kernels.kernels.filter((kernel) => kernel.documented);
  const kernelDetails = await mapWithConcurrency(
    kinds,
    DETAIL_CONCURRENCY,
    (kernel) =>
      fetchJson(
        `${apiBase}/kernels/${encodeURIComponent(kernel.kind)}`,
        fetchImplementation,
      ),
  );
  const presets = catalogs.models.checkpoints.flatMap(
    (checkpoint) => checkpoint.presets,
  );
  const trees = await mapWithConcurrency(presets, DETAIL_CONCURRENCY, (preset) => {
    const member = shownMember(preset);
    return member
      ? fetchJson(treeUrl(apiBase, preset.id, member.params), fetchImplementation)
      : null;
  });
  return {
    kernels: new Map(
      kinds.map((kernel, index) => [kernel.kind, kernelDetails[index]]),
    ),
    trees: new Map(presets.map((preset, index) => [preset.id, trees[index]])),
  };
}

/* ---------- Paths ---------- */

export const kernelPagePath = (kind) => `kernels/${kind}.md`;
export const deploymentPagePath = (presetId) => `models/${presetId}.md`;

// A table cell: one line, no pipes.
const cell = (value) =>
  value == null || value === ""
    ? ""
    : String(value)
        .replace(/\s*\n\s*/g, " ")
        .replace(/\|/g, "\\|");
const table = (header, rows) =>
  [
    `| ${header.join(" | ")} |`,
    `| ${header.map(() => "---").join(" | ")} |`,
    ...rows.map((row) => `| ${row.map(cell).join(" | ")} |`),
  ].join("\n");

/* ---------- Kernel kind ---------- */

export function renderKernelPage({ kernel, detail, links, archNames }) {
  const lines = [
    `# ${text(detail.title ?? kernel.title)} (${code(kernel.kind)}) | ServingStudio Kernel Library`,
    "",
  ];
  if (detail.summary) lines.push(text(detail.summary), "");
  if (detail.description) lines.push(text(detail.description), "");
  lines.push(
    `Category: ${text(detail.category ?? kernel.category)}${kernel.subcategory ? `; group: ${text(kernel.subcategory)}` : ""}. Interactive page: ${links.page("kernels", { kind: kernel.kind })}. All kernel kinds: ${links.file("kernels.md")}.`,
  );
  if (detail.formula?.length)
    lines.push(
      "",
      "## Formula",
      "",
      ...detail.formula.map((line) => `- ${code(line)}`),
    );
  if (detail.method)
    lines.push("", "## How it is measured", "", text(detail.method));
  if (detail.caveats?.length)
    lines.push(
      "",
      "## Caveats",
      "",
      ...detail.caveats.map((line) => `- ${text(line)}`),
    );
  if (detail.args?.length)
    lines.push(
      "",
      "## Arguments",
      "",
      table(
        ["Argument", "Type", "Unit", "Meaning"],
        detail.args.map((argument) => [
          code(argument.name),
          argument.type,
          argument.unit ?? "",
          text(argument.doc ?? ""),
        ]),
      ),
    );
  if (detail.metrics?.length)
    lines.push(
      "",
      "## Metrics",
      "",
      ...detail.metrics.map(
        (metric) =>
          `- ${text(metric.label)} (${code(metric.name)}${metric.unit ? `, ${metric.unit}` : ""})`,
      ),
    );
  const backends = Object.entries(detail.backends ?? {});
  if (backends.length) {
    lines.push("", "## Backends", "");
    for (const [name, backend] of backends) {
      const supports = backend.supports ?? {};
      const requirements = [
        supports.compute?.length
          ? `dtypes ${join(supports.compute.map(code))}`
          : null,
        supports.kv?.length ? `KV dtypes ${join(supports.kv.map(code))}` : null,
        supports.min_compute_capability != null
          ? `compute capability ${supports.min_compute_capability} or newer`
          : null,
        supports.sm_targets?.length
          ? `SM targets ${join(supports.sm_targets.map(code))}`
          : null,
        supports.gpus?.length ? `GPUs ${join(supports.gpus)}` : "any CUDA GPU",
      ].filter(Boolean);
      lines.push(
        `- ${code(name)}: ${backend.summary ? `${text(backend.summary)} ` : ""}${join(requirements)}.${backend.url ? ` Source: ${backend.url}` : ""}`,
      );
    }
  }
  lines.push("", "## Measured rows", "");
  if (kernel.coverage.length)
    lines.push(
      `${plural(kernel.rows ?? 0, "row")} in all.`,
      "",
      table(
        ["GPU", "Backend", "Precision", "Rows"],
        kernel.coverage.map((entry) => [
          entry.gpu,
          code(entry.backend),
          entry.precision ? code(entry.precision) : "",
          entry.rows.toLocaleString("en-US"),
        ]),
      ),
    );
  else lines.push("None measured yet.");
  if (kernel.used_by?.length)
    lines.push(
      "",
      "## Used by these deployments",
      "",
      ...kernel.used_by.map(
        (presetId) =>
          `- [${text(archNames.get(presetId) ?? presetId)}](${links.file(deploymentPagePath(presetId))}) (${code(presetId)})`,
      ),
    );
  lines.push(
    "",
    "## API",
    "",
    `- This kind: ${links.api}/kernels/${kernel.kind}`,
    `- Measured rows: ${links.api}/kernels/${kernel.kind}/rows (filter by any column, such as \`?gpu=NVIDIA%20H200\`; \`format=csv\` for CSV)`,
    `- Configs the public deployments ask of it: ${links.api}/kernels/${kernel.kind}/configs`,
  );
  if (detail.reference?.source)
    lines.push(
      "",
      "## Torch reference",
      "",
      `What the kind computes, as Torch code (\`${detail.reference.path}\` in ServingStudio Sim):`,
      "",
      "```python",
      detail.reference.source.trimEnd(),
      "```",
    );
  return `${lines.join("\n")}\n${links.footer(kernelPagePath(kernel.kind))}`;
}

/* ---------- Model deployment ---------- */

/* What each contract means for a reader (modelData.js CONTRACTS). */
const CONTRACTS = {
  iter_wise: "Whole iteration",
  layer_wise_attn: "Disaggregated attention side",
  layer_wise_ffn: "Disaggregated FFN side",
};

const paramsText = (params) =>
  Object.keys(params).length
    ? Object.entries(params)
        .map(([name, value]) => `${name} ${value}`)
        .join(", ")
    : "(none)";

function memberStatus(member) {
  if (member.error) return `cannot build: ${member.error}`;
  if (member.missing == null) return "not checked yet";
  const missing = Object.entries(member.missing);
  return missing.length
    ? `missing rows: ${missing.map(([kind, count]) => `${kind} ${count}`).join(", ")}`
    : "yes";
}

export function renderDeploymentPage({
  checkpoint,
  preset,
  tree,
  caseFields,
  simulationPresets,
  links,
}) {
  const name = preset.arch_name ?? preset.arch;
  const lines = [
    `# ${text(checkpoint.name)}: ${text(name)} | ServingStudio Models`,
    "",
    `Deployment ${code(preset.id)} of ${text(checkpoint.name)} (${code(checkpoint.checkpoint)}, family ${text(checkpoint.family)}). Arch ${code(preset.arch)}; ${CONTRACTS[preset.contract] ?? preset.contract}; GPU ${preset.gpu}. Interactive page, with its cost tree and live prediction: ${links.page("models", { preset: preset.id })}. All models: ${links.file("models.md")}.`,
    "",
    "## Configurations",
    "",
    preset.axes.length
      ? `The preset sweeps ${join(preset.axes.map((axis) => code(axis.name)))}; each configuration (member) is one value of each.`
      : "The preset has one configuration.",
    "",
    table(
      [
        "Parameters",
        "GPUs per replica",
        "Case form",
        "Groups",
        "max_model_len",
        "Can be timed",
      ],
      preset.members.map((member) => [
        paramsText(member.params),
        member.gpus_per_replica,
        member.predict ? code(member.predict.selector) : "",
        member.predict?.groups ?? "",
        member.predict?.max_model_len?.toLocaleString("en-US") ?? "",
        memberStatus(member),
      ]),
    ),
  ];

  const selectors = [
    ...new Set(
      preset.members.map((member) => member.predict?.selector).filter(Boolean),
    ),
  ];
  const example = shownMember(preset);
  lines.push(
    "",
    "## Timing a batch",
    "",
    `\`POST ${links.api}/predict\` with the preset id, one configuration's \`params\` and 1 to 64 \`cases\`. A case is \`{"groups": [...]}\`, one object per group (the table's Groups column), each with the fields of the configuration's case form. The answer gives each case's time per section and per node of its cost tree. The servingstudio-timing-predict skill walks through it: ${links.file("skills/servingstudio-timing-predict.md")}.`,
  );
  if (example)
    lines.push(
      "",
      "```json",
      JSON.stringify({ preset: preset.id, params: example.params, cases: ["…"] }),
      "```",
    );
  for (const selector of selectors) {
    const fields = caseFields?.[selector]?.groups ?? [];
    if (!fields.length) continue;
    lines.push(
      "",
      `Fields of a group for case form ${code(selector)}:`,
      "",
      table(
        ["Field", "Type", "Required", "Meaning"],
        fields.map((field) => [
          code(field.name),
          field.type,
          field.required ? "yes" : "no",
          text(field.description ?? ""),
        ]),
      ),
    );
  }

  if (tree?.sections?.length) {
    lines.push(
      "",
      `## Kernels of configuration ${paramsText(tree.params ?? example?.params ?? {})}`,
      "",
      "Each section's leaf slots, in order: the kernel kind each calls and the backends measured for it. Other configurations share the shape but not every size.",
    );
    for (const section of tree.sections) {
      lines.push(
        "",
        `### Section ${code(section.section)}`,
        "",
        table(
          ["Slot", "Kernel", "Backends"],
          section.slots.map((slot) => [
            code(slot.name),
            `[${slot.kernel}](${links.file(kernelPagePath(slot.kernel))})`,
            join((slot.backends ?? []).map(code)),
          ]),
        ),
      );
    }
  }

  const simulations = simulationPresets.filter((simulation) =>
    Object.values(simulation.pools).some((pool) => pool.arch_preset === preset.id),
  );
  if (simulations.length)
    lines.push(
      "",
      "## Simulations that run it",
      "",
      ...simulations.map(
        (simulation) =>
          `- [${code(simulation.id)}](${links.page("simulate", { preset: simulation.id })}): ${simulation.deployment} deployment, ${plural(simulation.members.length, "configuration")}`,
      ),
    );
  lines.push(
    "",
    "## API",
    "",
    `- This deployment in the model list: ${links.api}/models`,
  );
  if (example)
    lines.push(
      `- The kernels of one configuration: ${treeUrl(links.api, preset.id, example.params)}`,
    );
  return `${lines.join("\n")}\n${links.footer(deploymentPagePath(preset.id))}`;
}
