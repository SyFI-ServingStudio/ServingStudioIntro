import { ArrowLeft, Check, Copy, Download, ExternalLink } from "lucide-react";
import { useEffect, useMemo, useState, useSyncExternalStore } from "react";
import {
  apiUrl,
  argNames,
  argRole,
  argUnit,
  downloadText,
  expandRows,
  formatValue,
  isDtype,
  isList,
  isNumeric,
  formatDate,
  formatNumber,
  loadKernel,
  loadRows,
  metricDoc,
  metricNames,
  precisionArg,
  readQuery,
  setQuery,
  shortGpu,
  subscribeUrl,
  toCsv,
} from "./kernelData";
import { kernelHref, openKernel } from "./Kernels";
import { ConfigPicker, buildLevels, shownDims } from "./ConfigPicker";
import { PerfChart, SERIES_COLORS, seriesColor } from "./PerfChart";
import { Tag, ToggleTag } from "./Tag";
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
          onClick={(event) => openKernel(event, {})}
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
          <Explorer
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

const ROLE_LABELS = {
  sweep: "Varies with the batch",
  config: "Fixed by the model",
};

const sortValues = (values) =>
  [...values].sort((a, b) =>
    typeof a === "number" && typeof b === "number"
      ? a - b
      : String(a).localeCompare(String(b)),
  );

function useExplorerState(kernel, records, query) {
  return useMemo(() => {
    const args = argNames(kernel);
    const dims = ["gpu", "backend", ...args];
    const values = Object.fromEntries(
      dims.map((d) => [d, sortValues(new Set(records.map((r) => r[d])))]),
    );
    const role = (a) => argRole(kernel, a);
    const numeric = (a) => isNumeric(kernel, a);
    // The x axis is what a run sweeps (tokens, batch, message size). A shape
    // dimension such as a GEMM's n or k is fixed by the model, so it is never
    // one. A kernel no supported model runs has no roles yet and falls back
    // to any numeric argument. In argument order, the default axis first.
    const sweeps = args.filter(
      (a) => values[a] && numeric(a) && role(a) === "sweep" && values[a].length > 1,
    );
    const numericSweeps = sweeps.length
      ? sweeps
      : args.filter((a) => numeric(a) && values[a].length > 1);
    const x = numericSweeps.includes(query.x) ? query.x : numericSweeps[0];
    // Two or more shape dimensions move together (n with k, heads with head
    // size), so lines of one of them would mix shapes. They are picked as a
    // whole in the configuration instead. A lone one, like num_gpus, compares.
    const shapeDims = args.filter(
      (a) => numeric(a) && role(a) !== "sweep" && values[a].length > 1,
    );
    const comparable = ["backend", "gpu", ...args].filter(
      (d) =>
        d !== x &&
        values[d].length > 1 &&
        !(shapeDims.length > 1 && shapeDims.includes(d)),
    );
    const color = comparable.includes(query.color) ? query.color : "backend";
    // With no numeric sweep, what varies is a list argument such as the
    // per-request (queries, context) pairs. There is no axis to plot it on,
    // so its rows are all listed rather than pinned to one value.
    const listDims = x
      ? []
      : args.filter((a) => isList(kernel, a) && values[a].length > 1);
    const filterDims = dims.filter(
      (d) => d !== x && d !== color && !listDims.includes(d),
    );

    // Explicit choices come from the URL. Every other filter defaults to the
    // configuration with the most measured rows among the rows that match.
    const selection = {};
    const matchesFixed = (r) =>
      Object.entries(selection).every(([d, v]) => String(r[d]) === v);
    filterDims.forEach((d) => {
      if (query[d] != null && values[d].some((v) => String(v) === query[d]))
        selection[d] = query[d];
    });
    const open = filterDims.filter((d) => !(d in selection));
    if (open.length) {
      const counts = new Map();
      records.filter(matchesFixed).forEach((r) => {
        const key = JSON.stringify(open.map((d) => r[d]));
        counts.set(key, (counts.get(key) || 0) + 1);
      });
      let best = null;
      counts.forEach((n, key) => {
        if (!best || n > best[1]) best = [key, n];
      });
      const tuple = best ? JSON.parse(best[0]) : open.map((d) => values[d][0]);
      open.forEach((d, i) => (selection[d] = String(tuple[i])));
    }
    return {
      dims,
      values,
      numericSweeps,
      comparable,
      x,
      color,
      listDims,
      filterDims,
      selection,
    };
  }, [kernel, records, query]);
}

function Explorer({ kernel, records, catalog, query }) {
  const state = useExplorerState(kernel, records, query);
  const {
    values,
    numericSweeps,
    comparable,
    x,
    color,
    listDims,
    filterDims,
    selection,
  } = state;
  // A metric no row records (energy for collectives) is not offered.
  const metrics = metricNames(kernel).filter((m) =>
    records.some((r) => r[m] != null),
  );
  const y = metrics.includes(query.y)
    ? query.y
    : metrics.includes(kernel.default_metric)
      ? kernel.default_metric
      : metrics[0];
  // A log axis needs every x above zero; a sweep that starts at 0 stays linear.
  const xValues = x ? values[x] : [];
  const positiveX = xValues.length > 0 && xValues[0] > 0;
  const spanX = positiveX ? xValues.at(-1) / xValues[0] : 0;
  const scale = query.scale || (spanX > 16 ? "logx" : "linear");
  const logX = positiveX && scale.includes("logx");
  const logY = scale.includes("logy");
  const showPeak = query.peak !== "off";

  const update = (patch, options) => setQuery({ ...query, ...patch }, options);

  const matching = records.filter((r) =>
    filterDims.every((d) => String(r[d]) === selection[d]),
  );
  const colorValues = values[color];
  const present = sortValues(new Set(matching.map((r) => r[color])));
  const series = present
    .slice(0, SERIES_COLORS.length)
    .map((value) => ({
      key: String(value),
      label:
        color === "gpu"
          ? shortGpu(value)
          : formatValue(value, argUnit(kernel, color)),
      color: seriesColor(colorValues, present, value),
      points: matching
        .filter((r) => r[color] === value && r[y] != null && (!logY || r[y] > 0))
        .map((r) => ({ x: r[x], y: r[y], record: r }))
        .sort((a, b) => a.x - b.x),
    }))
    .filter((serie) => serie.points.length);
  const dropped = present.length - Math.min(present.length, SERIES_COLORS.length);

  const peak = showPeak ? peakFor(kernel, y, color, selection, catalog) : null;
  // What the lines are is the first choice; the configuration then pins
  // everything else. Only dimensions with more than one value can be compared.
  const varyingDtypes = state.dims.filter(
    (d) => isDtype(kernel, d) && values[d].length > 1,
  );
  // In the order the configuration below reads: backend, GPU, precision,
  // shape, then sweeps.
  const rank = (d) =>
    d === "backend"
      ? 0
      : d === "gpu"
        ? 1
        : isDtype(kernel, d)
          ? 2
          : argRole(kernel, d) === "sweep"
            ? 4
            : 3;
  const compareDims = [...comparable].sort((a, b) => rank(a) - rank(b));
  const compareLabel = (d) =>
    d === "gpu"
      ? "GPU"
      : d === "backend"
        ? "Backend"
        : isDtype(kernel, d) && varyingDtypes.length === 1
          ? "Precision"
          : d;
  // The same, as a noun in a sentence: "one GPU", "one precision".
  const compareNoun = (d) =>
    d === "gpu" ? "GPU" : compareLabel(d) === d ? d : compareLabel(d).toLowerCase();
  const levels = buildLevels(kernel, values, filterDims);
  const shown = shownDims(levels);
  const fixedDims = filterDims.filter(
    (d) => values[d].length === 1 && !shown.includes(d),
  );

  const metric = metricDoc(kernel, y);
  const describe = `${metric.label} of ${kernel.kind} against ${x}, one line per ${color}. ${series.length} series, ${matching.length} measured rows.`;
  const listNames = listDims.join(" and ");

  return (
    <div className={s.explorer}>
      <section className={s.controls} aria-label="Chart settings">
        <div className={s.controlsHead}>
          <div className={s.group}>
            <span className={s.groupName}>Compare</span>
            <div className={s.chips} role="group" aria-label="Compare">
              {compareDims.map((d) => (
                <ToggleTag
                  key={d}
                  type="choice"
                  value={d}
                  pressed={d === color}
                  onClick={() => update({ color: d, [d]: "" })}
                >
                  {compareLabel(d)}
                </ToggleTag>
              ))}
            </div>
          </div>
          <div className={s.group}>
            <span className={s.groupName}>X axis</span>
            <div className={s.chips} role="group" aria-label="X axis">
              {!x ? (
                <span className={s.fixed}>None: only {listNames} varies</span>
              ) : numericSweeps.length === 1 ? (
                <Tag type="choice" value={x} title="The only dimension swept">
                  {x}
                </Tag>
              ) : (
                numericSweeps.map((a) => (
                  <ToggleTag
                    key={a}
                    type="choice"
                    value={a}
                    pressed={a === x}
                    onClick={() => update({ x: a, [a]: "" })}
                  >
                    {a}
                  </ToggleTag>
                ))
              )}
            </div>
          </div>
        </div>
        <div className={s.controlsBody}>
          <ConfigPicker
            levels={levels}
            records={records}
            deployments={kernel.used_by}
            models={catalog.models}
            selection={selection}
            compare={
              color === "backend" ? null : { dim: color, label: compareNoun(color) }
            }
            onPick={(patch) => update(patch)}
          />
          {fixedDims.length > 0 && (
            <p className={s.fixed}>
              Every row has{" "}
              {fixedDims.map((dim, i) => (
                <span key={dim}>
                  {i > 0 && ", "}
                  <code>{dim}</code>{" "}
                  {dim === "gpu"
                    ? shortGpu(values[dim][0])
                    : formatValue(values[dim][0], argUnit(kernel, dim))}
                </span>
              ))}
              .
            </p>
          )}
        </div>
      </section>

      <div className={s.chartColumn}>
        <div className={s.chartBar}>
          <div className={s.metricSwitch} role="radiogroup" aria-label="Metric">
            {metrics.map((m) => (
              <button
                key={m}
                type="button"
                role="radio"
                aria-checked={m === y}
                onClick={() => update({ y: m })}
              >
                {metricDoc(kernel, m).label}
                <span>{metricDoc(kernel, m).unit}</span>
              </button>
            ))}
          </div>
          <div className={s.chartOptions}>
            <label className={s.toggle}>
              <input
                type="checkbox"
                checked={logX}
                disabled={!positiveX}
                onChange={(event) =>
                  update({
                    scale:
                      `${event.target.checked ? "logx" : ""}${logY ? "logy" : ""}` ||
                      "linear",
                  })
                }
              />
              Log x
            </label>
            <label className={s.toggle}>
              <input
                type="checkbox"
                checked={logY}
                onChange={(event) =>
                  update({
                    scale:
                      `${logX ? "logx" : ""}${event.target.checked ? "logy" : ""}` ||
                      "linear",
                  })
                }
              />
              Log y
            </label>
            <label className={s.toggle}>
              <input
                type="checkbox"
                checked={showPeak}
                onChange={(event) =>
                  update({ peak: event.target.checked ? "" : "off" })
                }
              />
              Spec-sheet peak
            </label>
          </div>
        </div>

        {!x ? (
          <p className={s.note}>
            These rows differ only in {listNames}, which has no numeric axis to
            plot against. Each row is listed below.
          </p>
        ) : series.length ? (
          <>
            <PerfChart
              series={series}
              xName={x}
              xUnit={argUnit(kernel, x)}
              colorName={color === "gpu" ? "GPU" : color}
              yLabel={metric.label}
              yUnit={metric.unit}
              logX={logX}
              logY={logY}
              peak={peak}
              describe={describe}
            />
            {dropped > 0 && (
              <p className={s.note}>
                {dropped} more values of {color} have rows here. The chart shows the
                first {SERIES_COLORS.length}; compare something else to see them.
              </p>
            )}
            {y === "energy_j" && (
              <p className={s.note}>
                Rows without an energy reading are left out of this chart.
              </p>
            )}
          </>
        ) : (
          <EmptyState
            records={records}
            filterDims={filterDims}
            selection={selection}
            onApply={(tuple) => update(tuple)}
          />
        )}
      </div>

      <div className={s.rowsColumn}>
        <RowsTable
          kernel={kernel}
          rows={matching}
          columnsBy={x ? [x] : listDims}
          color={color}
        />
      </div>
    </div>
  );
}

/* The spec-sheet ceiling Sim gives this metric on the picked GPU. A throughput
   ceiling depends on the compute dtype, so it shows only with one picked. */
function peakFor(kernel, y, color, selection, catalog) {
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

function EmptyState({ records, filterDims, selection, onApply }) {
  const nearby = useMemo(() => {
    const seen = new Map();
    records.forEach((r) => {
      const tuple = Object.fromEntries(filterDims.map((d) => [d, String(r[d])]));
      const key = JSON.stringify(tuple);
      if (!seen.has(key)) {
        const score = filterDims.filter((d) => tuple[d] === selection[d]).length;
        seen.set(key, { tuple, score, rows: 0 });
      }
      seen.get(key).rows += 1;
    });
    return [...seen.values()]
      .sort((a, b) => b.score - a.score || b.rows - a.rows)
      .slice(0, 5);
  }, [records, filterDims, selection]);
  return (
    <div className={s.empty}>
      <p>
        Nothing was measured with this combination. These measured configurations
        are the closest:
      </p>
      <ul>
        {nearby.map(({ tuple, rows }) => (
          <li key={JSON.stringify(tuple)}>
            <button type="button" onClick={() => onApply(tuple)}>
              {filterDims
                .filter((d) => tuple[d] !== selection[d])
                .map(
                  (d) =>
                    `${d === "gpu" ? "GPU" : d} ${d === "gpu" ? shortGpu(tuple[d]) : tuple[d]}`,
                )
                .join(", ")}
              <span>{rows} rows</span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

function RowsTable({ kernel, rows, columnsBy, color }) {
  // Rows sort by the x axis when there is one; list columns keep row order.
  const x =
    columnsBy.length === 1 && typeof rows[0]?.[columnsBy[0]] === "number"
      ? columnsBy[0]
      : null;
  const [all, setAll] = useState(false);
  const sorted = useMemo(
    () =>
      [...rows].sort(
        (a, b) =>
          String(a[color]).localeCompare(String(b[color]), undefined, {
            numeric: true,
          }) || (x ? a[x] - b[x] : 0),
      ),
    [rows, x, color],
  );
  const shown = all ? sorted : sorted.slice(0, 12);
  const metrics = metricNames(kernel);
  const columns = [color, ...columnsBy, ...metrics];
  const csvColumns = ["gpu", "backend", ...argNames(kernel), ...metrics];
  if (!rows.length) return null;
  return (
    <div className={s.rows}>
      <div className={s.rowsHead}>
        <h2>Rows behind the chart</h2>
        <button
          type="button"
          className={s.action}
          onClick={() =>
            downloadText(
              `${kernel.kind}-selection.csv`,
              toCsv(sorted, csvColumns),
              "text/csv",
            )
          }
        >
          <Download size={18} aria-hidden="true" />
          Download these {rows.length} rows as CSV
        </button>
      </div>
      <div
        className={s.tableScroll}
        tabIndex={0}
        role="region"
        aria-label="Measured rows"
      >
        <table className={s.table}>
          <thead>
            <tr>
              {columns.map((c) => (
                <th
                  key={c}
                  scope="col"
                  className={metrics.includes(c) ? s.num : undefined}
                >
                  {metrics.includes(c)
                    ? `${metricDoc(kernel, c).label} ${metricDoc(kernel, c).unit}`
                    : c === "gpu"
                      ? "GPU"
                      : c}
                </th>
              ))}
              <th scope="col">Measured</th>
            </tr>
          </thead>
          <tbody>
            {shown.map((r, i) => (
              <tr key={i}>
                {columns.map((c) => (
                  <td key={c} className={metrics.includes(c) ? s.num : s.code}>
                    {metrics.includes(c)
                      ? formatNumber(r[c])
                      : c === "gpu"
                        ? shortGpu(r[c])
                        : formatValue(r[c], argUnit(kernel, c))}
                  </td>
                ))}
                <td className={s.date}>
                  {formatDate(r.provenance.profiler_run_at)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {sorted.length > 12 && (
        <button type="button" className={s.more} onClick={() => setAll(!all)}>
          {all ? "Show the first 12 rows" : `Show all ${sorted.length} rows`}
        </button>
      )}
    </div>
  );
}

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
              <th scope="col">Role</th>
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
                <td>{ROLE_LABELS[arg.role] ?? "Not run by a listed model yet"}</td>
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
  // The request carries the configuration the chart is showing, defaults included.
  const { selection, filterDims } = useExplorerState(kernel, records, query);
  const params = new URLSearchParams(filterDims.map((d) => [d, selection[d]]));
  const rowsUrl = apiUrl(`kernels/${kernel.kind}/rows`);
  const api = `${rowsUrl}?${params}`;
  const snippets = [
    ["curl", `curl "${api}&format=csv" -o ${kernel.kind}.csv`],
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
    ["DuckDB", `SELECT *\nFROM read_csv_auto('${api}&format=csv');`],
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
          <span>
            Coverage per GPU and backend, and the models that use each kernel
          </span>
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
