import { ArrowLeft, Check, Copy, Download, ExternalLink } from "lucide-react";
import { useEffect, useState, useSyncExternalStore } from "react";
import {
  apiUrl,
  expandRows,
  formatDate,
  loadConfigs,
  loadKernel,
  loadRows,
  shortGpu,
} from "./kernelData";
import { openInPage, readQuery, setQuery, subscribeUrl } from "../../url";
import { kernelHref } from "./Kernels";
import { GridExplorer } from "./GridExplorer";
import { SeriesExplorer, hasSeriesView } from "./SeriesExplorer";
import { Measurements, apiFilters } from "./Measurements";
import { Tag } from "./Tag";
import s from "./KernelDetail.module.css";

const TABS = [
  ["performance", "Performance"],
  ["about", "About"],
  ["implementations", "Implementations"],
  ["data", "Data and API"],
];
const search = () => window.location.search;

export function KernelDetail({ catalog, entry }) {
  useSyncExternalStore(subscribeUrl, search);
  const query = readQuery();
  const tab = TABS.some(([id]) => id === query.tab) ? query.tab : "performance";
  const [loaded, setLoaded] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    Promise.all([loadKernel(entry.kind), loadRows(entry.kind)]).then(
      ([kernel, rows]) => setLoaded({ kernel, records: expandRows(rows) }),
      setError,
    );
  }, [entry.kind]);
  const { kernel, records } = loaded ?? {};
  useEffect(() => {
    document.title = `${entry.title} | Kernels | ServingStudio`;
  }, [entry.title]);

  return (
    <div className={s.page}>
      <header className={`wrap ${s.header}`}>
        <a
          className={s.back}
          href={kernelHref({})}
          onClick={(event) => openInPage(event, {})}
        >
          <ArrowLeft size={18} aria-hidden="true" />
          All kernels
        </a>
        <h1>{entry.title}</h1>
        <p className={s.identity}>
          <code>{entry.kind}</code>
          <Tag type="category" value={entry.category} />
          {entry.subcategory && <Tag type="category" value={entry.subcategory} />}
        </p>
        {entry.summary && <p className={s.summary}>{entry.summary}</p>}
      </header>

      <div className={`wrap ${s.tabs}`} role="tablist" aria-label="Kernel sections">
        {TABS.map(([id, label]) => (
          <button
            key={id}
            type="button"
            role="tab"
            id={`tab-${id}`}
            aria-selected={tab === id}
            aria-controls={`panel-${id}`}
            onClick={() =>
              setQuery(
                { ...query, tab: id === "performance" ? "" : id },
                { push: true },
              )
            }
          >
            {label}
          </button>
        ))}
      </div>

      <section
        className={`wrap ${s.panel}`}
        role="tabpanel"
        id={`panel-${tab}`}
        aria-labelledby={`tab-${tab}`}
      >
        {error ? (
          <p role="alert">
            The measurements did not load ({error.message}). The kernel data service
            may be down; reload the page to try again.
          </p>
        ) : !kernel ? (
          <p role="status" className={s.loading}>
            Loading {entry.rows.toLocaleString("en-US")} measurements…
          </p>
        ) : tab === "performance" ? (
          <Performance
            kernel={kernel}
            records={records}
            catalog={catalog}
            query={query}
          />
        ) : tab === "about" ? (
          <About kernel={kernel} />
        ) : tab === "implementations" ? (
          <Implementations kernel={kernel} records={records} />
        ) : (
          <DataAndApi
            kernel={kernel}
            records={records}
            catalog={catalog}
            query={query}
          />
        )}
      </section>
    </div>
  );
}

/* ---------- Performance ---------- */

/* Views of one kernel's numbers. A kind whose Sim doc declares a chart over
   several configs (kernel.view) opens on it: one line per config of one
   configuration's leaf. "Simulator grid" (the default otherwise, when a public
   deployment reads this kind) plots each config on the cache axes it
   interpolates over, per GPU and model. "Measurements" is a table of every
   profile.db row, whoever asked for it, to sort and filter. */
function Performance({ kernel, records, catalog, query }) {
  const [list, setList] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    loadConfigs(kernel.kind).then(setList, setError);
  }, [kernel.kind]);
  const update = (patch) => setQuery({ ...query, ...patch });
  const hasGrid = list?.configs.length > 0;
  const hasSeries = Boolean(list) && hasSeriesView(kernel, list);
  const views = [
    ...(hasSeries ? [["series", kernel.view.title]] : []),
    ["grid", "Simulator grid"],
    ["rows", "Measurements"],
  ];
  const available = (id) =>
    id === "series" ? hasSeries : id === "grid" ? hasGrid : id === "rows";
  // The first available view is the default and keeps the URL clean.
  const fallback = views.find(([id]) => available(id))[0];
  const view = available(query.view) ? query.view : fallback;
  if (!list && !error)
    return (
      <p role="status" className={s.loading}>
        Loading kernel configs…
      </p>
    );
  return (
    <>
      <div className={s.viewBar}>
        <div className={s.viewSwitch} role="radiogroup" aria-label="View">
          {views.map(([id, label]) => (
            <button
              key={id}
              type="button"
              role="radio"
              aria-checked={view === id}
              disabled={!available(id)}
              onClick={() => update({ view: id === fallback ? "" : id })}
            >
              {label}
              <span>
                {id === "series"
                  ? `by ${kernel.view.series.label.toLowerCase()}`
                  : id === "grid"
                    ? `${list?.configs.length ?? 0} configs`
                    : `${records.length.toLocaleString("en-US")} rows`}
              </span>
            </button>
          ))}
        </div>
        <p className={s.viewNote}>
          {error
            ? `The kernel configs did not load (${error.message}).`
            : !hasGrid
              ? "No public deployment reads this kernel's rows yet."
              : view === "series"
                ? `One chart per configuration and layer, one line per ${kernel.view.series.label.toLowerCase()}.`
                : view === "grid"
                  ? "The cells the simulator reads for each kernel config, on its own cache axes."
                  : "Every measured row for this kernel, including shapes no public deployment reads."}
        </p>
      </div>
      {view === "series" ? (
        <SeriesExplorer
          kernel={kernel}
          catalog={catalog}
          list={list}
          query={query}
          update={update}
        />
      ) : view === "grid" ? (
        <GridExplorer
          kernel={kernel}
          catalog={catalog}
          list={list}
          query={query}
          update={update}
        />
      ) : (
        <Measurements
          kernel={kernel}
          records={records}
          query={query}
          update={update}
        />
      )}
    </>
  );
}

const sortValues = (values) =>
  [...values].sort((a, b) =>
    typeof a === "number" && typeof b === "number"
      ? a - b
      : String(a).localeCompare(String(b)),
  );

/* ---------- About ---------- */

function About({ kernel }) {
  return (
    <div className={s.prose}>
      <h2>Overview</h2>
      <p className={s.description}>{kernel.description}</p>

      <h2>What is computed</h2>
      <pre className={s.formula}>{kernel.formula.join("\n")}</pre>

      <h2>Arguments</h2>
      <div
        className={s.tableScroll}
        tabIndex={0}
        role="region"
        aria-label="Arguments"
      >
        <table className={s.table}>
          <thead>
            <tr>
              <th scope="col">Name</th>
              <th scope="col">Unit</th>
              <th scope="col">Meaning</th>
            </tr>
          </thead>
          <tbody>
            {kernel.args.map((arg) => (
              <tr key={arg.name}>
                <th scope="row" className={s.code}>
                  {arg.name}
                </th>
                <td>{arg.unit || "–"}</td>
                <td>{arg.doc}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h2>How it was measured</h2>
      <p>{kernel.method}</p>

      <h2>Before you use these numbers</h2>
      <ul>
        {kernel.caveats.map((c) => (
          <li key={c}>{c}</li>
        ))}
      </ul>
    </div>
  );
}

/* ---------- Implementations ---------- */

/* Rows record the backend library version only for some runs, so each version
   says how many rows it covers: "2.11.0+cu130 (204 rows), not recorded (136 rows)".
   One version recorded on every row is just the version. */
function libraryVersions(rows) {
  const counts = new Map();
  rows.forEach((r) => {
    const version = r.provenance.backend_version ?? null;
    counts.set(version, (counts.get(version) || 0) + 1);
  });
  if (!rows.length || (counts.size === 1 && counts.has(null))) return null;
  if (counts.size === 1) return [...counts.keys()][0];
  return [...counts]
    .map(
      ([version, n]) =>
        `${version ?? "not recorded"} (${n.toLocaleString("en-US")} rows)`,
    )
    .join(", ");
}

function Implementations({ kernel, records }) {
  const measured = sortValues(new Set(records.map((r) => r.backend)));
  const names = [...new Set([...measured, ...Object.keys(kernel.backends)])];
  return (
    <div className={s.prose}>
      <h2>Backends</h2>
      <ul className={s.backendList}>
        {names.map((name) => {
          const doc = kernel.backends[name];
          const rows = records.filter((r) => r.backend === name);
          const versions = libraryVersions(rows);
          const gpus = [...new Set(rows.map((r) => shortGpu(r.gpu)))];
          return (
            <li key={name}>
              <h3>
                <code>{name}</code>
              </h3>
              {doc?.summary && <p>{doc.summary}</p>}
              <dl>
                <div>
                  <dt>Rows</dt>
                  <dd>
                    {rows.length
                      ? `${rows.length.toLocaleString("en-US")} on ${gpus.join(" and ")}`
                      : "Registered, not measured in this snapshot"}
                  </dd>
                </div>
                {versions && (
                  <div>
                    <dt>Library version</dt>
                    <dd>{versions}</dd>
                  </div>
                )}
                {doc?.url && (
                  <div>
                    <dt>Upstream</dt>
                    <dd>
                      <a
                        href={doc.url}
                        target="_blank"
                        rel="noreferrer"
                        className={s.external}
                      >
                        {linkText(doc.url)}
                        <ExternalLink size={14} aria-label="opens in a new tab" />
                      </a>
                    </dd>
                  </div>
                )}
              </dl>
            </li>
          );
        })}
      </ul>

      <h2>Torch reference</h2>
      {kernel.reference ? (
        <>
          <p>
            The semantic reference the backends are checked against. Source:{" "}
            <code>{kernel.reference.path}</code> in ServingStudio Sim, Apache 2.0.
          </p>
          <CodeBlock
            source={kernel.reference.source}
            name={kernel.reference.path}
          />
        </>
      ) : (
        <p>
          This kernel has no standalone torch reference yet. The backends are
          checked against each other and against the production framework they come
          from.
        </p>
      )}
    </div>
  );
}

function CodeBlock({ source, name }) {
  const [copied, setCopied] = useState(false);
  return (
    <div className={s.codeBlock}>
      <div className={s.codeHead}>
        <span>{name.split("/").at(-1)}</span>
        <button
          type="button"
          onClick={() => {
            navigator.clipboard.writeText(source);
            setCopied(true);
            setTimeout(() => setCopied(false), 1600);
          }}
        >
          {copied ? (
            <Check size={16} aria-hidden="true" />
          ) : (
            <Copy size={16} aria-hidden="true" />
          )}
          {copied ? "Copied" : "Copy code"}
        </button>
      </div>
      <pre tabIndex={0}>
        <code>{source}</code>
      </pre>
    </div>
  );
}

/* ---------- Data and API ---------- */

function DataAndApi({ kernel, records, catalog, query }) {
  // The request carries the Measurements table's filters the API can express:
  // each column picked down to one value (it filters by equality only).
  const params = new URLSearchParams(apiFilters(kernel, query));
  const rowsUrl = apiUrl(`kernels/${kernel.kind}/rows`);
  const csvParams = new URLSearchParams([...params, ["format", "csv"]]);
  const csvApi = `${rowsUrl}?${csvParams}`;
  const snippets = [
    ["curl", `curl "${csvApi}" -o ${kernel.kind}.csv`],
    [
      "Python",
      [
        "import requests",
        "",
        "params = {",
        ...[...params].map(
          ([k, v]) => `    ${JSON.stringify(k)}: ${JSON.stringify(v)},`,
        ),
        "}",
        `url = "${rowsUrl}"`,
        'rows = requests.get(url, params=params).json()["rows"]',
      ].join("\n"),
    ],
    ["DuckDB", `SELECT *\nFROM read_csv_auto('${csvApi}');`],
  ];
  return (
    <div className={s.prose}>
      <h2>Download</h2>
      <ul className={s.downloads}>
        <li>
          <a href={`${rowsUrl}?format=csv`} download>
            <Download size={18} aria-hidden="true" />
            Whole <code>{kernel.kind}</code> table, CSV
          </a>
          <span>{records.length.toLocaleString("en-US")} rows with provenance</span>
        </li>
        <li>
          <a href={apiUrl("kernels")} download="kernels.json">
            <Download size={18} aria-hidden="true" />
            Catalog of all {catalog.kernels.length} kernels, JSON
          </a>
          <span>Coverage per GPU, backend and precision</span>
        </li>
      </ul>

      <h2>Query</h2>
      <p>
        Every query parameter filters one column by equality: <code>gpu</code>,{" "}
        <code>backend</code> or an argument. Add <code>format=csv</code> for CSV.
      </p>
      {snippets.map(([label, code]) => (
        <div key={label} className={s.snippet}>
          <CodeBlock source={code} name={label} />
        </div>
      ))}

      <h2>Cite this data</h2>
      <p>
        ServingStudio Sim commit{" "}
        <code>{catalog.snapshot.sim_commit?.slice(0, 12) ?? "unknown"}</code>, last
        measured {formatDate(catalog.snapshot.last_measured_at)}. Each row also
        records the profiler commit, CUDA and driver versions, and, where the run
        captured it, the backend library version. The data and the torch reference
        code are licensed under Apache 2.0.
      </p>
    </div>
  );
}

/* "github.com/vllm-project/vllm/blob/main/csrc/layernorm_kernels.cu" reads as
   "github.com › layernorm_kernels.cu": the site and the file are what matter. */
function linkText(link) {
  const url = new URL(link);
  const last = url.pathname.split("/").filter(Boolean).at(-1);
  return last ? `${url.hostname} › ${last}` : url.hostname;
}
