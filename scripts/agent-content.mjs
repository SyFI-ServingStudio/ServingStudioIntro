/* What the site offers, for readers that run no JavaScript: AI crawlers and
   coding agents. Every page is a React app that fills #root in the browser,
   so its HTML alone says nothing. At build time this plugin writes

   - each page of PAGES as Markdown (`index.md`, `features.md`, …,
     `models.md`, `kernels.md`, `simulate.md`), and the same text as HTML
     inside the page's #root, with a `rel="alternate"` link to the Markdown
     (static-markup.mjs). A browser that runs scripts never shows that text:
     a `(scripting: enabled)` media query hides it, and React replaces
     #root's children on its first render anyway. The Overview, Features,
     Architecture and Blog text is the App itself, rendered at build time
     (prerender-pages.mjs); the Models, Kernels and Simulate lists come from
     ServingStudio Sim's public API;
   - one page per kernel kind and per model deployment
     (agent-detail-pages.mjs), since `kernels.html?kind=…` is the list page's
     bytes to a static host;
   - each blog post's Markdown source beside its page;
   - `llms.txt` (https://llmstxt.org/): what ServingStudio is, in the site's
     own words, and links to all of the above, the API, its docs and the
     skills;
   - the agent skills of a ServingStudio Sim checkout, every
     `public_api/skills/<name>/SKILL.md` byte for byte, as `skills/<name>.md`;
   - `robots.txt`, allowing every crawler and naming the sitemap.

   The API is read from PUBLIC_API_CATALOG_BASE (default: the public API of
   the deployed site): the catalogs, then each kind and one member's tree per
   deployment, a few requests at a time. A build that cannot read them stops,
   so a deploy never ships empty lists; ALLOW_NO_API_CATALOG=1 builds without
   the lists and detail pages and says so in the log and in the generated
   text, as ALLOW_NO_READ_MORE does for Read more (servingstudio-ui.mjs).

   The skills come from the checkout SERVINGSTUDIO_SIM_DIR names, which has no
   default, like SERVINGSTUDIO_UI_DIR. Without it, or with no skill in it, the
   build stops; ALLOW_NO_SKILLS=1 builds without them, and llms.txt says so.

   Wording is taken from the pages (their components, PageHero copy and meta
   descriptions) and from the API's own fields; add no claims here that the
   site does not make. */

import { existsSync, readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { readBlogPosts } from "./blog-content.mjs";
import {
  deploymentPagePath,
  fetchDetails,
  fetchJson,
  kernelPagePath,
  renderDeploymentPage,
  renderKernelPage,
} from "./agent-detail-pages.mjs";
import { htmlToMarkdown, prerenderPages } from "./prerender-pages.mjs";
import { injectStaticContent } from "./static-markup.mjs";
import {
  axesText,
  code,
  join,
  plural,
  predictable,
  text,
} from "./markdown-text.mjs";

export const DEFAULT_API_BASE =
  "https://servingstudio.cs.washington.edu/api/public/v1";

// The source repository and the skills, as the Overview hero and SkillInstall link them.
const GITHUB_REPO = "https://github.com/SyFI-ServingStudio/ServingStudio";
const SKILL_SOURCE =
  "https://github.com/SyFI-ServingStudio/ServingStudioSim/tree/main";
const SKILL_INSTALL = `npx skills add ${SKILL_SOURCE}/public_api`;

// The catalog routes one build reads, by the name the renderers use.
const CATALOG_ROUTES = {
  health: "health",
  models: "models",
  kernels: "kernels",
  simulationPresets: "simulations/presets",
  workloads: "workloads",
};

/* Each catalog page: its file, its PageHero title and description (copied
   from ModelCatalog.jsx, KernelCatalog.jsx and Simulate.jsx), the skill its
   hero links, and the Markdown body built from the catalogs. */
const CATALOG_PAGES = [
  {
    id: "models",
    title: "Models",
    description:
      "Each model runs as an execution graph, which the simulator costs as a tree of measured kernels. Pick a model and how it is deployed to see how an iteration's time is put together, then time a batch of your own.",
    skill: "servingstudio-timing-predict",
    render: (catalogs, links) => modelsBody(catalogs, links),
  },
  {
    id: "kernels",
    title: "Kernel Library",
    description:
      "Explore measured GPU performance across the kernels, shapes, and backends that power LLM serving.",
    skill: "servingstudio-kernel-performance",
    render: (catalogs, links) => kernelsBody(catalogs, links),
  },
  {
    id: "simulate",
    title: "Simulate",
    description:
      "Send traffic to a deployment and read the latency and throughput its clients would see.",
    skill: "servingstudio-simulate",
    render: (catalogs, links) => simulateBody(catalogs, links),
  },
];

/* ---------- Reading the API ---------- */

/* Every catalog route's JSON, or an error naming the route that failed. */
export async function fetchCatalogs(apiBase, fetchImplementation = fetch) {
  const entries = await Promise.all(
    Object.entries(CATALOG_ROUTES).map(async ([name, route]) => [
      name,
      await fetchJson(`${apiBase}/${route}`, fetchImplementation),
    ]),
  );
  const catalogs = Object.fromEntries(entries);
  if (catalogs.health.status !== "ok")
    throw new Error(`${apiBase}/health: status is ${catalogs.health.status}`);
  for (const [name, list] of [
    ["models", catalogs.models.checkpoints],
    ["kernels", catalogs.kernels.kernels],
    ["simulations/presets", catalogs.simulationPresets.presets],
  ])
    if (!Array.isArray(list) || !list.length)
      throw new Error(`${apiBase}/${name} answered an empty list`);
  return catalogs;
}

/* ---------- Page bodies ---------- */

function modelsBody({ models }, links) {
  const lines = [
    `ServingStudio publishes ${plural(models.checkpoints.length, "model")}. Each is listed with its deployments (public presets): how the model is split over GPUs, the GPU, the parameters the preset sweeps, and how many of its configurations could be timed when this text was generated. On the interactive page, a deployment shows its cost tree and times a batch you give it (\`POST ${links.apiPath}/predict\`).`,
  ];
  for (const checkpoint of models.checkpoints) {
    lines.push(
      "",
      `## ${text(checkpoint.name)}`,
      "",
      `Family: ${text(checkpoint.family)}. Checkpoint: ${code(checkpoint.checkpoint)}.`,
      "",
    );
    if (!checkpoint.presets.length) lines.push("No public deployment yet.");
    for (const preset of checkpoint.presets) {
      const timed = preset.members.filter(predictable).length;
      const details = [
        code(preset.id),
        preset.gpu,
        preset.axes.length ? `sweeps ${axesText(preset.axes)}` : null,
        `${timed} of ${plural(preset.members.length, "configuration")} can be timed`,
      ].filter(Boolean);
      lines.push(
        `- [${text(preset.arch_name ?? preset.arch)}](${links.page("models", { preset: preset.id })}): ${details.join("; ")}; details: [Markdown](${links.file(deploymentPagePath(preset.id))})`,
      );
    }
  }
  return lines.join("\n");
}

function kernelsBody({ kernels }, links) {
  const measuredAt = kernels.snapshot?.last_measured_at;
  const lines = [
    `The Kernel Library lists ${plural(kernels.kernels.length, "kernel kind")} measured on ${join(kernels.gpus.map((gpu) => gpu.name))}${measuredAt ? `, last measured ${measuredAt.slice(0, 10)}` : ""}. Each kind is one operation; its rows are measurements, one per combination of argument values, GPU and backend. Read the rows of a kind from \`${links.apiPath}/kernels/{kind}/rows\` (\`?format=csv\` for CSV).`,
  ];
  const categories = [
    ...kernels.categories,
    ...new Set(kernels.kernels.map((kernel) => kernel.category)),
  ].filter((category, index, all) => all.indexOf(category) === index);
  for (const category of categories) {
    const members = kernels.kernels.filter(
      (kernel) => kernel.category === category,
    );
    if (!members.length) continue;
    lines.push("", `## ${text(category)}`, "");
    for (const kernel of members) {
      const gpus = [...new Set(kernel.coverage.map((entry) => entry.gpu))];
      const details = [
        kernel.subcategory ? `group: ${text(kernel.subcategory)}` : null,
        kernel.backends.length
          ? `backends: ${join(kernel.backends.map(code))}`
          : null,
        kernel.precisions.length
          ? `precisions: ${join(kernel.precisions.map(code))}`
          : null,
        `GPUs: ${gpus.length ? join(gpus) : "none measured yet"}`,
        plural(kernel.rows ?? 0, "measured row"),
      ].filter(Boolean);
      const name = kernel.documented
        ? `[${text(kernel.title)}](${links.page("kernels", { kind: kernel.kind })})`
        : text(kernel.title ?? kernel.kind);
      lines.push(
        `- ${name} (${code(kernel.kind)}): ${kernel.summary ? `${text(kernel.summary)} ` : ""}${details.join("; ")}.${kernel.documented ? ` Details: [Markdown](${links.file(kernelPagePath(kernel.kind))})` : ""}`,
      );
    }
  }
  return lines.join("\n");
}

function simulateBody({ models, simulationPresets, workloads }, links) {
  const archNames = new Map(
    models.checkpoints.flatMap((checkpoint) =>
      checkpoint.presets.map((preset) => [
        preset.id,
        preset.arch_name ?? preset.arch,
      ]),
    ),
  );
  const checkpointNames = new Map(
    models.checkpoints.map((checkpoint) => [
      checkpoint.checkpoint,
      checkpoint.name,
    ]),
  );
  const order = [...checkpointNames.keys()];
  const position = (checkpoint) =>
    order.includes(checkpoint) ? order.indexOf(checkpoint) : order.length;
  const presets = [...simulationPresets.presets].sort(
    (first, second) => position(first.checkpoint) - position(second.checkpoint),
  );
  const limits = simulationPresets.limits;
  const lines = [
    `A simulation replays a stream of requests through one deployment and reports time to first token (TTFT), time per output token (TPOT), end-to-end latency and throughput. ${plural(presets.length, "deployment")} can be simulated. Start one with \`POST ${links.apiPath}/simulate\` and read it from \`${links.apiPath}/simulations/{simulation_id}\`. At most ${limits.max_running} simulations run at once and ${limits.max_queued} wait; a run may use up to ${limits.max_requests.toLocaleString("en-US")} requests and is stopped after ${limits.timeout_s / 60} minutes.`,
    "",
    "## Deployments",
  ];
  let currentCheckpoint = null;
  for (const preset of presets) {
    if (preset.checkpoint !== currentCheckpoint) {
      currentCheckpoint = preset.checkpoint;
      lines.push(
        "",
        `### ${text(checkpointNames.get(preset.checkpoint) ?? preset.checkpoint)}`,
        "",
      );
    }
    const pools = Object.entries(preset.pools).map(
      ([role, pool]) =>
        `${role}: [${text(archNames.get(pool.arch_preset) ?? pool.arch)}](${links.file(deploymentPagePath(pool.arch_preset))}) on ${pool.gpu}, worker ${code(pool.worker)}`,
    );
    const captures = preset.captures.map((capture) => {
      const facts = capture.facts;
      return facts
        ? `${code(capture.name)} (${plural(facts.requests, "request")}, mean ${Math.round(facts.prompt_tokens).toLocaleString("en-US")} prompt and ${Math.round(facts.output_tokens).toLocaleString("en-US")} output tokens)`
        : code(capture.name);
    });
    const details = [
      `${preset.deployment} deployment`,
      join(pools),
      preset.axes.length ? `sweeps ${axesText(preset.axes)}` : null,
      plural(preset.members.length, "configuration"),
      captures.length ? `captures: ${join(captures)}` : null,
    ].filter(Boolean);
    lines.push(
      `- [${code(preset.id)}](${links.page("simulate", { preset: preset.id })}): ${details.join("; ")}`,
    );
  }

  lines.push(
    "",
    "## Workloads",
    "",
    "A simulation's requests come from one of three sources:",
    "",
  );
  for (const [source, entry] of Object.entries(workloads.sources))
    lines.push(`- ${code(source)}: ${text(entry.summary)}`);
  for (const generator of workloads.sources.generated?.generators ?? []) {
    lines.push(
      "",
      `### Generator ${code(generator.name)}`,
      "",
      `${text(generator.about)}. Arguments:`,
      "",
    );
    for (const argument of generator.arguments)
      lines.push(
        `- ${code(argument.name)} (default ${code(argument.default)}${argument.choices.length ? `; one of ${join(argument.choices.map(code))}` : ""}): ${text(argument.help)}`,
      );
  }
  const upload = workloads.sources.upload;
  if (upload) {
    lines.push(
      "",
      "### Upload formats",
      "",
      `An uploaded CSV may be up to ${Math.round(upload.max_bytes / 1024)} KiB and is kept for ${upload.keep_s / 3600} hours.`,
      "",
    );
    for (const format of upload.formats)
      lines.push(
        `- ${code(format.name)}: columns ${join(format.columns.map(code))}; optional tags ${join(format.tags.map(code))}`,
      );
    for (const tag of upload.tags)
      lines.push(
        `- tag ${code(tag.name)} adds columns ${join(tag.columns.map(code))}`,
      );
  }
  if (workloads.routing) lines.push("", text(workloads.routing));
  return lines.join("\n");
}

/* ---------- Whole documents ---------- */

/* One catalog page as Markdown: title, description, where its data comes
   from, then the lists (or a note that this build has none). */
export function renderCatalogPage(page, catalogs, links) {
  const header = [
    `# ${page.title} | ServingStudio`,
    "",
    page.description,
    "",
    `Interactive page: ${links.page(page.id)}. The page reads its lists from ServingStudio Sim's public API at ${links.api} (OpenAPI: ${links.api}/openapi.json). An agent can use it through the [${page.skill}](${links.skill(page.skill)}) skill.`,
  ];
  const body = catalogs
    ? [
        `This text was generated when the site was built, from Sim commit ${code(catalogs.health.sim_commit)}; the API has the current lists.`,
        "",
        page.render(catalogs, links),
      ]
    : [
        "This build was made without the API's catalogs (ALLOW_NO_API_CATALOG=1), so the lists are not included here; read them from the API.",
      ];
  return `${[...header, "", ...body].join("\n")}\n${links.footer(`${page.id}.md`)}`;
}

/* llms.txt: the site in a few sentences of its own copy, then links. The
   summary is index.html's meta description; the paragraph joins the Overview
   hero (src/sections/Hero.jsx) and the Architecture hero
   (src/pages/Architecture.jsx). Change them together. */
export function renderLlmsText({
  links,
  pages,
  posts,
  skills,
  catalogs,
  kernelPages = [],
  deploymentPages = [],
}) {
  const lines = [
    "# ServingStudio",
    "",
    "> Use measured simulation, detailed performance analysis, and an Agent to improve LLM serving systems.",
    "",
    "Compare configurations using performance models grounded in real GPU measurements, then let an agent build and validate the strongest candidates in real serving frameworks. A kernel is measured once on the target GPU. Higher layers compose that measurement into a model, then into the workers and pools that serve requests. The top layer predicts a whole serving run.",
    "",
    `Every page below has a Markdown version, linked here. The Models, Kernels and Simulate pages read ServingStudio Sim's public JSON API at ${links.api}${catalogs ? `; their Markdown, and one page per kernel kind and per model deployment, were generated from it at build time (Sim commit ${catalogs.health.sim_commit})` : ""}. Source code: ${GITHUB_REPO}.`,
    "",
    "## Pages",
    "",
    ...pages.map(
      (page) => `- [${page.name}](${page.markdown}): ${page.description}`,
    ),
    "",
    "## Public API",
    "",
    `- [API base](${links.api}): JSON over HTTPS; \`/health\`, \`/models\`, \`/kernels\`, \`/simulations/presets\` and \`/workloads\` list what it serves; \`POST /predict\` and \`POST /simulate\` run work`,
    `- [OpenAPI schema](${links.api}/openapi.json): every route`,
    `- [Interactive API docs](${links.api}/docs)`,
    "",
    "## Agent skills",
    "",
    `Install them all with \`${SKILL_INSTALL}\`.`,
    "",
    ...(skills.length
      ? skills.map(
          (skill) => `- [${skill.name}](${skill.url}): ${skill.description}`,
        )
      : [
          "This build was made without the skills (ALLOW_NO_SKILLS=1), so they are not included here; the install command above fetches them from GitHub.",
        ]),
  ];
  if (deploymentPages.length)
    lines.push(
      "",
      "## Model deployments",
      "",
      ...deploymentPages.map((page) => `- [${page.name}](${page.url})`),
    );
  if (kernelPages.length)
    lines.push(
      "",
      "## Kernel kinds",
      "",
      ...kernelPages.map(
        (page) =>
          `- [${page.name}](${page.url})${page.summary ? `: ${page.summary}` : ""}`,
      ),
    );
  lines.push(
    "",
    "## Optional",
    "",
    ...posts.map(
      (post) => `- [${post.title}](${post.markdown}): ${post.description}`,
    ),
    `- [Source code](${GITHUB_REPO})`,
  );
  return `${lines.join("\n")}\n`;
}

export function renderRobotsText(origin) {
  return `# Every crawler is welcome. For a plain-text guide to the site, its
# Markdown pages, the public API and the agent skills, read ${origin}/llms.txt
User-agent: *
Allow: /

Sitemap: ${origin}/sitemap.xml
`;
}

/* Every skill of a ServingStudio Sim checkout: each directory of
   public_api/skills that holds a SKILL.md, with its bytes (published as they
   are) and the one-line description of its front matter (a folded
   `description: >-` block). Throws, naming the cause, when `simDir` is unset
   or holds no skill. */
export function readSkills(simDir) {
  if (!simDir)
    throw new Error(
      "SERVINGSTUDIO_SIM_DIR is not set; point it at a ServingStudio Sim checkout.",
    );
  const directory = path.join(path.resolve(simDir), "public_api", "skills");
  if (!existsSync(directory))
    throw new Error(`${path.resolve(simDir)} has no public_api/skills directory.`);
  const skills = readdirSync(directory, { withFileTypes: true })
    .filter(
      (entry) =>
        entry.isDirectory() &&
        existsSync(path.join(directory, entry.name, "SKILL.md")),
    )
    .map(({ name }) => {
      const file = path.join(directory, name, "SKILL.md");
      const bytes = readFileSync(file);
      const source = bytes.toString("utf8");
      const frontMatter = source.match(/^---\n([\s\S]*?)\n---\n/);
      if (!frontMatter) throw new Error(`${file} has no front matter`);
      const description = frontMatter[1]
        .match(/^description:\s*>-?\n((?:[ \t]+.*\n?)+)/m)?.[1]
        .split("\n")
        .map((line) => line.trim())
        .filter(Boolean)
        .join(" ");
      if (!description) throw new Error(`${file} has no description`);
      return { name, description, bytes };
    })
    .sort((first, second) => first.name.localeCompare(second.name));
  if (!skills.length) throw new Error(`${directory} holds no <name>/SKILL.md.`);
  return skills;
}

/* A page's meta description, the one sentence each page already has. */
function metaDescription(root, file) {
  const html = readFileSync(path.join(root, file || "index.html"), "utf8");
  const match = html.match(/<meta\s+name="description"\s+content="([^"]*)"/);
  return match ? match[1] : "";
}

/* ---------- The plugin ---------- */

// A page's Markdown file: index.md for the Overview, else its HTML name.
const markdownFile = (page) =>
  page.file ? page.file.replace(/\.html$/, ".md") : "index.md";

/* `fetchCatalogs`, `fetchDetails` and `prerender` are replaceable for
   tests; a build reads the API and renders the App. */
export function agentContent({
  origin,
  pages,
  apiBase,
  allowMissing = false,
  simDir,
  allowMissingSkills = false,
  log = console,
  fetchCatalogs: readCatalogs = fetchCatalogs,
  fetchDetails: readDetails = fetchDetails,
  prerender = prerenderPages,
}) {
  let config;
  let catalogs = null;
  let details = null;
  let skills = [];
  let prerendered = {};
  const links = {
    origin,
    // The URL readers and agents use, whichever base the build read from.
    api: `${origin}/api/public/v1`,
    apiPath: "/api/public/v1",
    page: (id, params = {}) => {
      const page = pages.find((entry) => entry.id === id);
      const query = new URLSearchParams(params).toString();
      return `${origin}${config.base}${page.file}${query ? `?${query}` : ""}`;
    },
    file: (fileName) => `${origin}${config.base}${fileName}`,
    skill: (name) => `${origin}${config.base}skills/${name}.md`,
    // The last lines of every generated page: where this text and the guide live.
    footer: (fileName) =>
      `\n---\n\nThis page as Markdown: ${origin}${config.base}${fileName}. A guide to the whole site for AI agents: [llms.txt](${origin}${config.base}llms.txt).\n`,
  };
  const catalogPage = (page) => CATALOG_PAGES.find((entry) => entry.id === page.id);
  const pageOfHtml = (htmlFile) =>
    pages.find((page) => (page.file || "index.html") === htmlFile);

  /* Each page's Markdown: the catalog pages from the API, the others from
     their prerendered App. */
  const pageMarkdown = (page) => {
    const catalog = catalogPage(page);
    if (catalog) return renderCatalogPage(catalog, catalogs, links);
    const html = prerendered[page.file];
    if (html == null)
      throw new Error(`No prerendered markup for ${page.file || "index.html"}`);
    return `${htmlToMarkdown(html, { origin: `${origin}${config.base}`.replace(/\/$/, "") })}\n${links.footer(markdownFile(page))}`;
  };

  const readSkillsOrSkip = () => {
    try {
      skills = readSkills(simDir);
      log.log(
        `[agent-content] skills from ${path.resolve(simDir)}: ${skills.map((skill) => skill.name).join(", ")}`,
      );
    } catch (error) {
      if (!allowMissingSkills)
        throw new Error(
          `No agent skills to publish: ${error.message} To build the site without them, set ALLOW_NO_SKILLS=1.`,
        );
      log.warn(`[agent-content] no skills: ${error.message}`);
      skills = [];
    }
  };

  const readApiOrSkip = async () => {
    const base = apiBase || DEFAULT_API_BASE;
    try {
      catalogs = await readCatalogs(base);
      details = await readDetails(base, catalogs);
      log.log(
        `[agent-content] catalogs and ${details.kernels.size} kernel and ${details.trees.size} deployment details from ${base} (Sim ${catalogs.health.sim_commit})`,
      );
    } catch (error) {
      if (!allowMissing)
        throw new Error(
          `Cannot read the public API's catalogs for the pages' text and llms.txt: ${error.message}. Set PUBLIC_API_CATALOG_BASE to a running API, or ALLOW_NO_API_CATALOG=1 to build without the lists.`,
        );
      log.warn(
        `[agent-content] no catalogs, building without the lists: ${error.message}`,
      );
      catalogs = null;
      details = null;
    }
  };

  return {
    name: "agent-content",
    apply: "build",
    configResolved(value) {
      config = value;
    },
    async buildStart() {
      readSkillsOrSkip();
      const [rendered] = await Promise.all([
        prerender({
          root: config.root,
          origin: `${origin}${config.base}`.replace(/\/$/, ""),
          files: pages
            .filter((page) => !catalogPage(page))
            .map((page) => page.file),
        }),
        readApiOrSkip(),
      ]);
      prerendered = rendered;
    },
    transformIndexHtml: {
      order: "post",
      handler(html, context) {
        const page = pageOfHtml(path.basename(context.filename));
        if (!page) return html;
        return injectStaticContent(html, {
          markdown: pageMarkdown(page),
          markdownUrl: `${config.base}${markdownFile(page)}`,
        });
      },
    },
    /* Runs before sitemap.mjs's generateBundle (vite.config.js lists this
       plugin first), so the sitemap sees every Markdown file emitted here. */
    generateBundle() {
      const emit = (fileName, source) =>
        this.emitFile({ type: "asset", fileName, source });
      for (const page of pages) emit(markdownFile(page), pageMarkdown(page));

      const kernelPages = [];
      const deploymentPages = [];
      if (catalogs && details) {
        const archNames = new Map(
          catalogs.models.checkpoints.flatMap((checkpoint) =>
            checkpoint.presets.map((preset) => [
              preset.id,
              `${checkpoint.name}: ${preset.arch_name ?? preset.arch}`,
            ]),
          ),
        );
        for (const kernel of catalogs.kernels.kernels) {
          const detail = details.kernels.get(kernel.kind);
          if (!detail) continue;
          emit(
            kernelPagePath(kernel.kind),
            renderKernelPage({ kernel, detail, links, archNames }),
          );
          kernelPages.push({
            name: `${kernel.title} (${kernel.kind})`,
            url: links.file(kernelPagePath(kernel.kind)),
            summary: kernel.summary,
          });
        }
        for (const checkpoint of catalogs.models.checkpoints)
          for (const preset of checkpoint.presets) {
            emit(
              deploymentPagePath(preset.id),
              renderDeploymentPage({
                checkpoint,
                preset,
                tree: details.trees.get(preset.id),
                caseFields: catalogs.models.case_fields,
                simulationPresets: catalogs.simulationPresets.presets,
                links,
              }),
            );
            deploymentPages.push({
              name: archNames.get(preset.id),
              url: links.file(deploymentPagePath(preset.id)),
            });
          }
      }

      for (const skill of skills) emit(`skills/${skill.name}.md`, skill.bytes);

      // Each published post's Markdown source, beside its page.
      const posts = readBlogPosts(config.root, config.base).map((post) => {
        const fileName = `blog/${post.slug}/index.md`;
        emit(
          fileName,
          readFileSync(
            path.join(config.root, "content/blog", post.slug, "index.md"),
          ),
        );
        return { ...post, markdown: links.file(fileName) };
      });

      emit(
        "llms.txt",
        renderLlmsText({
          links,
          pages: pages.map((page) => ({
            name: page.name,
            markdown: links.file(markdownFile(page)),
            description: metaDescription(config.root, page.file),
          })),
          posts,
          skills: skills.map((skill) => ({
            ...skill,
            url: links.skill(skill.name),
          })),
          catalogs,
          kernelPages,
          deploymentPages,
        }),
      );
      emit(
        "robots.txt",
        renderRobotsText(`${origin}${config.base}`.replace(/\/$/, "")),
      );
    },
  };
}
