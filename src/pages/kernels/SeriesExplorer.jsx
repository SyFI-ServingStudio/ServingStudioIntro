import { useEffect, useState } from "react";
import { ChartBar } from "./ChartBar";
import { Level, deploymentLabel } from "./GridExplorer";
import {
  argUnit,
  formatNumber,
  formatValue,
  loadConfig,
  metricDoc,
  metricNames,
  modelFamily,
  modelName,
  peakFor,
  shortGpu,
} from "./kernelData";
import { PerfChart, SERIES_COLORS } from "./PerfChart";
import { Tag, ToggleTag } from "./Tag";
import picker from "./ConfigPicker.module.css";
import k from "./KernelDetail.module.css";
import g from "./GridExplorer.module.css";
import s from "./SeriesExplorer.module.css";

/* A chart the kind declares over several of its configs (kernel.view, a
   ConfigView in the kind's Sim doc): the configs one deployment builds for one
   leaf that differ only in view.series.field are one chart, a line per config.
   view.workload.field is picked with a selector, by the name /configs gives
   each config's value (config_labels). Series values are positions in an
   order, 0 first: line n reads "<label> n+1", and position 0, which leads the
   order, is drawn strongest. Nothing here knows the kind.

   URL keys: cgpu, cmodel (shared with the grid view), dep, work, leaf and
   backend. */

const byNumber = (a, b) =>
  String(a).localeCompare(String(b), undefined, { numeric: true });

// Position 0 in the base colour, later positions stepped toward the page, so
// the order reads as one hue fading; SERIES_COLORS[0] is tuned for the page.
const SURFACE = [12, 13, 15];
function rampColor(position, count) {
  const hex = SERIES_COLORS[0];
  const base = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
  const keep = count > 1 ? 1 - (0.62 * position) / (count - 1) : 1;
  const mixed = base.map((c, i) => Math.round(c * keep + SURFACE[i] * (1 - keep)));
  return `#${mixed.map((c) => c.toString(16).padStart(2, "0")).join("")}`;
}

/* GPU → model → deployment → the configs it uses that carry both view
   fields, each with the roles that ask for it. A config no deployment claims
   sits under model null. */
function buildIndex(list, view) {
  const deployments = new Map(list.deployments.map((d) => [d.id, d]));
  const gpus = new Map();
  for (const config of list.configs) {
    if (!(view.series.field in config.config_args)) continue;
    if (!config.config_labels?.[view.workload.field]) continue;
    if (!gpus.has(config.gpu)) gpus.set(config.gpu, new Map());
    const models = gpus.get(config.gpu);
    const claims = config.uses.flatMap((use) =>
      use.deployments.length
        ? use.deployments.map(({ id }) => [deployments.get(id), use])
        : [[null, use]],
    );
    const named = claims.some(([d]) => d);
    for (const [d, use] of claims) {
      if (!d && named) continue;
      const stem = d?.model_config ?? null;
      if (!models.has(stem)) models.set(stem, new Map());
      const byDeployment = models.get(stem);
      const key = d ? String(d.id) : "none";
      if (!byDeployment.has(key))
        byDeployment.set(key, { deployment: d, configs: new Map() });
      const configs = byDeployment.get(key).configs;
      if (!configs.has(config.config_hash))
        configs.set(config.config_hash, { config, roles: new Set() });
      configs.get(config.config_hash).roles.add(use.role);
    }
  }
  return gpus;
}

/* A deployment's configs by workload value, then by leaf: configs that agree
   on everything but the series field (the roles that ask, every other config
   value, how the workload is bound) are one leaf, a line each. */
function workloadsOf(entries, view) {
  const byLabel = new Map();
  for (const entry of entries) {
    const name = entry.config.config_labels[view.workload.field];
    if (!byLabel.has(name.label)) byLabel.set(name.label, { name, entries: [] });
    byLabel.get(name.label).entries.push(entry);
  }
  return [...byLabel.values()]
    .map((w) => ({ ...w, leaves: leavesOf(w.entries, view) }))
    .sort(
      (a, b) =>
        a.name.preference - b.name.preference ||
        byNumber(a.name.label, b.name.label),
    );
}

function leavesOf(entries, view) {
  const field = view.series.field;
  const leaves = new Map();
  for (const entry of entries) {
    const { config } = entry;
    const others = Object.entries(config.config_args).filter(
      ([key]) => key !== field,
    );
    const key = JSON.stringify([
      [...entry.roles].sort(),
      others.sort(),
      config.config_labels[view.workload.field].binding,
    ]);
    if (!leaves.has(key))
      leaves.set(key, { roles: [...entry.roles].sort(), lines: new Map() });
    const lines = leaves.get(key).lines;
    const position = config.config_args[field];
    if (!lines.has(position)) lines.set(position, config);
  }
  return [...leaves.values()].map((leaf) => {
    const lines = [...leaf.lines]
      .sort(([a], [b]) => a - b)
      .map(([position, config]) => ({ position, config }));
    return { ...leaf, lines, id: lines[0].config.config_hash.slice(0, 12) };
  });
}

/* What tells leaves apart: the steps of their role paths that differ
   ("mtp.step_0" against "mtp.recurrent"), else the binding values that do. */
function leafNames(leaves, view) {
  const paths = leaves.map((leaf) => leaf.roles[0].split("."));
  const shortest = Math.min(...paths.map((p) => p.length));
  let head = 0;
  while (head < shortest - 1 && paths.every((p) => p[head] === paths[0][head]))
    head += 1;
  let tail = 0;
  while (
    tail < shortest - head - 1 &&
    paths.every((p) => p.at(-1 - tail) === paths[0].at(-1 - tail))
  )
    tail += 1;
  const texts = paths.map((p) => p.slice(head, p.length - tail).join("."));
  if (new Set(texts).size === leaves.length) return texts;
  const bindingOf = (leaf) =>
    leaf.lines[0].config.config_labels[view.workload.field].binding;
  const differs = Object.keys(bindingOf(leaves[0])).filter(
    (key) => new Set(leaves.map((l) => JSON.stringify(bindingOf(l)[key]))).size > 1,
  );
  return leaves.map((leaf) =>
    differs.length
      ? differs
          .map((key) => `${key} ${formatValue(bindingOf(leaf)[key])}`)
          .join(" · ")
      : leaf.id,
  );
}

export function SeriesExplorer({ kernel, catalog, list, query, update }) {
  const view = kernel.view;
  const index = buildIndex(list, view);
  const models = catalog.models;
  const rank = (stem) => {
    const i = models.findIndex((m) => m.model_config === stem);
    return i < 0 ? models.length : i;
  };
  const gpuNames = [...index.keys()].sort((a, b) => {
    const order = catalog.gpus.map((x) => x.name);
    return (
      (order.indexOf(a) + 1 || order.length + 1) -
      (order.indexOf(b) + 1 || order.length + 1)
    );
  });
  if (!gpuNames.length)
    return (
      <p className={k.note}>
        No registered config carries both of this view&apos;s fields.
      </p>
    );
  const gpu = index.has(query.cgpu) ? query.cgpu : gpuNames[0];
  const byModel = index.get(gpu);
  const stems = [...byModel.keys()].sort((a, b) => rank(a) - rank(b));
  const modelKey = (stem) => stem ?? "";
  const stem = stems.find((m) => modelKey(m) === query.cmodel) ?? stems[0];
  const byDeployment = byModel.get(stem);
  const labelOf = (d) => (d ? deploymentLabel(d) : "Built at the deployment level");
  const deploymentKeys = [...byDeployment.keys()].sort((a, b) =>
    byNumber(
      labelOf(byDeployment.get(a).deployment),
      labelOf(byDeployment.get(b).deployment),
    ),
  );
  // The default deployment draws the most lines: the view is about comparing them.
  const widest = (key) =>
    Math.max(
      ...workloadsOf([...byDeployment.get(key).configs.values()], view).flatMap(
        (w) => w.leaves.map((l) => l.lines.length),
      ),
    );
  const depKey = byDeployment.has(query.dep)
    ? query.dep
    : [...deploymentKeys].sort((a, b) => widest(b) - widest(a))[0];
  const { deployment, configs } = byDeployment.get(depKey);
  const workloads = workloadsOf([...configs.values()], view);
  // The API orders measured routings first; the default is the first of them.
  const workload =
    workloads.find((w) => w.name.label === query.work) ?? workloads[0];
  const leaves = workload.leaves;
  const names = leafNames(leaves, view);
  const leafIndex = Math.max(
    0,
    leaves.findIndex((l) => l.id === query.leaf),
  );
  const leaf = leaves[leafIndex];

  const here = { cgpu: gpu, cmodel: modelKey(stem), dep: depKey };

  return (
    <div className={k.explorer}>
      <section className={k.controls} aria-label={view.title}>
        <div className={g.levels}>
          <Level label="GPU">
            {gpuNames.length === 1 ? (
              <Tag type="gpu" value={shortGpu(gpu)} />
            ) : (
              gpuNames.map((name) => (
                <ToggleTag
                  key={name}
                  type="gpu"
                  value={shortGpu(name)}
                  pressed={name === gpu}
                  onClick={() => update({ cgpu: name })}
                />
              ))
            )}
          </Level>
          <Level label="Model">
            {stems.map((m) => (
              <ToggleTag
                key={modelKey(m)}
                type="family"
                value={m ? modelFamily(models, m) : null}
                pressed={m === stem}
                title={m ?? "Configs no model deployment claims"}
                onClick={() => update({ cgpu: gpu, cmodel: modelKey(m) })}
              >
                {m ? modelName(models, m) : "No model"}
              </ToggleTag>
            ))}
          </Level>
        </div>
        <div className={s.levels}>
          <Level label="Deployment">
            {deploymentKeys.map((key) => (
              <ToggleTag
                key={key}
                type="choice"
                value={key}
                pressed={key === depKey}
                onClick={() => update({ ...here, dep: key, work: "", leaf: "" })}
              >
                <span className={s.code}>
                  {labelOf(byDeployment.get(key).deployment)}
                </span>
              </ToggleTag>
            ))}
          </Level>
          <Level label={view.workload.label}>
            {workloads.map((w) => (
              <ToggleTag
                key={w.name.label}
                type="choice"
                value={w.name.label}
                pressed={w === workload}
                title={[w.name.label, w.name.fingerprint].join("\n")}
                onClick={() => update({ ...here, work: w.name.label, leaf: "" })}
              >
                {w.name.label !== w.name.routing && (
                  <span className={picker.op}>{w.name.routing}</span>
                )}
                <span className={s.code}>{w.name.label}</span>
              </ToggleTag>
            ))}
          </Level>
          {leaves.length > 1 && (
            <Level label="Layer">
              {leaves.map((l, i) => (
                <ToggleTag
                  key={l.id}
                  type="choice"
                  value={l.id}
                  pressed={i === leafIndex}
                  title={l.roles.join("\n")}
                  onClick={() =>
                    update({ ...here, work: workload.name.label, leaf: l.id })
                  }
                >
                  <span className={s.code}>{names[i]}</span>
                </ToggleTag>
              ))}
            </Level>
          )}
        </div>
        <p className={s.summary}>{view.summary}</p>
      </section>

      <LinesChart
        key={leaf.lines.map((l) => l.config.config_hash).join()}
        kernel={kernel}
        catalog={catalog}
        view={view}
        leaf={leaf}
        workload={workload}
        deployment={deployment}
        query={query}
        update={update}
      />
    </div>
  );
}

function LinesChart({
  kernel,
  catalog,
  view,
  leaf,
  workload,
  deployment,
  query,
  update,
}) {
  const [details, setDetails] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    Promise.all(
      leaf.lines.map(({ config }) =>
        loadConfig(kernel.kind, config.config_hash, config.gpu),
      ),
    ).then(setDetails, setError);
  }, [kernel.kind, leaf]);
  if (error)
    return (
      <p role="alert" className={k.note}>
        These configs&apos; grids did not load ({error.message}).
      </p>
    );
  if (!details)
    return (
      <p role="status" className={k.loading}>
        Loading {leaf.lines.length} configs…
      </p>
    );
  return (
    <LinesView
      kernel={kernel}
      catalog={catalog}
      view={view}
      leaf={leaf}
      details={details}
      workload={workload}
      deployment={deployment}
      query={query}
      update={update}
    />
  );
}

function LinesView({
  kernel,
  catalog,
  view,
  leaf,
  details,
  workload,
  deployment,
  query,
  update,
}) {
  const [first] = details;
  const coords = first.cache_coords;
  // The sweep axis on x: the first cache axis that moves. Any other is held
  // at its most measured value, per config.
  const xi = Math.max(
    0,
    first.axes.findIndex((axis) => axis.length > 1),
  );
  const xName = coords[xi];
  const hasMeasured = (p) => Object.keys(p.measured).length > 0;
  const sliceOf = (detail) => {
    const at = detail.axes.map((axis, i) => {
      if (i === xi) return null;
      const n = (v) =>
        detail.points.filter((p) => p.coords[i] === v && hasMeasured(p)).length;
      return [...axis].sort((a, b) => n(b) - n(a) || a - b)[0];
    });
    return detail.points
      .filter((p) => p.coords.every((v, i) => i === xi || v === at[i]))
      .sort((a, b) => a.coords[xi] - b.coords[xi]);
  };
  const slices = details.map(sliceOf);

  // One backend at a time: the URL's, else the one that measured most cells.
  const cellsBy = new Map();
  slices
    .flat()
    .forEach((p) =>
      Object.keys(p.measured).forEach((b) =>
        cellsBy.set(b, (cellsBy.get(b) ?? 0) + 1),
      ),
    );
  const backends = [...cellsBy.keys()].sort(
    (a, b) => cellsBy.get(b) - cellsBy.get(a) || byNumber(a, b),
  );
  const backend = backends.includes(query.backend) ? query.backend : backends[0];

  const metrics = metricNames(kernel).filter((m) =>
    slices.flat().some((p) => p.measured[backend]?.[m] != null),
  );
  const y = metrics.includes(query.y)
    ? query.y
    : metrics.includes(kernel.default_metric)
      ? kernel.default_metric
      : metrics[0];
  const xs = [...new Set(slices.flat().map((p) => p.coords[xi]))].sort(
    (a, b) => a - b,
  );
  const positiveX = xs[0] > 0;
  const scale =
    query.scale || (positiveX && xs.at(-1) / xs[0] > 16 ? "logx" : "linear");
  const logX = positiveX && scale.includes("logx");
  const logY = scale.includes("logy");
  const showPeak = query.peak !== "off";

  const count = leaf.lines.length;
  const lineLabel = (position) => `${view.series.label} ${position + 1}`;
  const value = (p) => {
    const v = p.measured[backend]?.[y];
    return v == null || (logY && v <= 0) ? null : v;
  };
  const series = leaf.lines.map(({ position, config }, i) => ({
    key: config.config_hash,
    label: lineLabel(position),
    color: rampColor(position, count),
    width: position === 0 ? 4.5 : 2,
    points: slices[i].map((p) => ({ x: p.coords[xi], y: value(p), record: p })),
  }));
  const lead = leaf.lines.findIndex((l) => l.position === 0);
  const cells =
    lead < 0
      ? undefined
      : slices[lead].map((p) => ({
          x: p.coords[xi],
          status: !p.feasible
            ? "infeasible"
            : p.measured[backend]
              ? "measured"
              : "missing",
        }));

  const metric = y ? metricDoc(kernel, y) : null;
  const peak =
    showPeak && y
      ? peakFor(kernel, y, "series", { gpu: first.gpu, ...first.fixed }, catalog)
      : null;
  const unit = argUnit(kernel, xName);
  const xLabel = unit && unit !== "bytes" ? `${xName} (${unit})` : xName;
  const name = workload.name;

  return (
    <>
      <div className={k.chartColumn}>
        <ChartBar
          kernel={kernel}
          metrics={metrics}
          y={y}
          logX={logX}
          logY={logY}
          positiveX={positiveX}
          showPeak={showPeak}
          update={update}
        />
        {backends.length > 1 && (
          <div className={g.axes}>
            <Level label="Backend">
              {backends.map((b) => (
                <ToggleTag
                  key={b}
                  type="choice"
                  value={b}
                  pressed={b === backend}
                  onClick={() => update({ backend: b })}
                >
                  {b}
                </ToggleTag>
              ))}
            </Level>
          </div>
        )}
        {metric && series.some((serie) => serie.points.some((p) => p.y != null)) ? (
          <PerfChart
            series={series}
            xName={xName}
            xLabel={xLabel}
            xUnit={unit}
            // Each line's label already names the series.
            colorName=""
            yLabel={metric.label}
            yUnit={metric.unit}
            logX={logX}
            logY={logY}
            peak={peak}
            cells={cells}
            describe={`${metric.label} of ${kernel.kind} over ${xName}, one line per ${view.series.label.toLowerCase()} of ${count}, ${backend}, ${view.workload.label.toLowerCase()} ${name.label}.`}
            note={`Each line is one config's grid for ${backend}; ${lineLabel(0)}, drawn thickest, leads the order. The ticks along the bottom are its cells.`}
            foot={() => null}
          />
        ) : (
          <p className={k.note}>
            No backend has measured a cell of these configs yet.
          </p>
        )}
      </div>

      <div className={k.rowsColumn}>
        <div className={g.facts}>
          <h2>These configs</h2>
          <dl>
            <div>
              <dt>{view.series.label}</dt>
              <dd>{view.series.doc}</dd>
            </div>
            <div>
              <dt>{view.workload.label}</dt>
              <dd className={g.stack}>
                <code>{name.label}</code>
                <span className={s.muted}>
                  {name.routing}, {name.fingerprint}
                  {Object.entries(name.binding).map(
                    ([key, v]) => `, ${key} ${formatValue(v)}`,
                  )}
                </span>
                <span className={s.muted}>{view.workload.doc}</span>
              </dd>
            </div>
            <div>
              <dt>Deployment</dt>
              <dd>
                <code>
                  {deployment
                    ? deploymentLabel(deployment)
                    : "Built at the deployment level"}
                </code>
              </dd>
            </div>
            <div>
              <dt>Fixed profile.db args</dt>
              <dd>
                {Object.entries(first.fixed).map(([key, v]) => (
                  <span key={key} className={g.fact}>
                    <code>{key}</code> {formatValue(v, argUnit(kernel, key))}
                  </span>
                ))}
              </dd>
            </div>
            <div>
              <dt>Asked for by</dt>
              <dd className={g.stack}>
                {leaf.roles.map((role) => (
                  <code key={role}>{role}</code>
                ))}
              </dd>
            </div>
          </dl>
        </div>
        {metric && (
          <LinesTable
            kernel={kernel}
            xName={xName}
            xs={xs}
            series={series}
            metric={metric}
          />
        )}
      </div>
    </>
  );
}

function LinesTable({ kernel, xName, xs, series, metric }) {
  const [all, setAll] = useState(false);
  const at = (serie, x) => serie.points.find((p) => p.x === x)?.y;
  // Only the cells some line measured; the rest are in the grid view.
  const rows = xs.filter((x) => series.some((serie) => at(serie, x) != null));
  const shown = all ? rows : rows.slice(0, 12);
  return (
    <div className={k.rows}>
      <div className={k.rowsHead}>
        <h2>Values behind the chart</h2>
      </div>
      <div
        className={k.tableScroll}
        tabIndex={0}
        role="region"
        aria-label="Values behind the chart"
      >
        <table className={k.table}>
          <thead>
            <tr>
              <th scope="col">{xName}</th>
              {series.map((serie) => (
                <th key={serie.key} scope="col" className={k.num}>
                  {serie.label}
                  <span className={g.unit}>
                    {metric.label} {metric.unit}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {shown.map((x) => (
              <tr key={x}>
                <td className={k.code}>{formatValue(x, argUnit(kernel, xName))}</td>
                {series.map((serie) => {
                  const v = at(serie, x);
                  return (
                    <td key={serie.key} className={k.num}>
                      {v == null ? "–" : formatNumber(v)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className={k.note}>
        {rows.length} of {xs.length} {xName} cells measured by some line.
      </p>
      {rows.length > 12 && (
        <button type="button" className={k.more} onClick={() => setAll(!all)}>
          {all ? "Show the first 12 rows" : `Show all ${rows.length} rows`}
        </button>
      )}
    </div>
  );
}

export const hasSeriesView = (kernel, list) =>
  Boolean(kernel.view) &&
  list.configs.some(
    (c) =>
      kernel.view.series.field in c.config_args &&
      c.config_labels?.[kernel.view.workload.field],
  );
