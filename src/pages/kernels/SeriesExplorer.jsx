import { useEffect, useState } from "react";
import { ChartBar } from "./ChartBar";
import { Level, presetAxes } from "./GridExplorer";
import {
  archName,
  argUnit,
  formatNumber,
  formatValue,
  loadConfig,
  metricDoc,
  metricNames,
  modelFamily,
  modelName,
  peakFor,
  presetArch,
  presetCheckpoint,
  shortGpu,
} from "./kernelData";
import { memberHref, memberMatches, paramsText } from "../models/modelData";
import { MemberPicker } from "../models/MemberPicker";
import { PerfChart, SERIES_COLORS } from "./PerfChart";
import { Tag, ToggleTag } from "./Tag";
import k from "./KernelDetail.module.css";
import g from "./GridExplorer.module.css";
import s from "./SeriesExplorer.module.css";

/* A chart the kind declares over several of its configs (kernel.view, a
   ConfigView in the kind's Sim doc): the configs one parameter set of a
   public preset builds for one leaf that differ only in view.series.field
   are one chart, a line per config. view.workload.field is the routing the
   parameter set reads, so picking the parameter set picks it. Series values
   are positions in an order, 0 first: line n reads "<label> n+1", and
   position 0, which leads the order, is drawn strongest. The chart shows
   each line relative to the first one by default, since the lines differ by
   far less than a line moves along x; absolute values are a switch away.
   Nothing here knows the kind.

   URL keys: cgpu, cmodel (shared with the grid view), dep (the preset's id),
   m.<axis> (the member's value of each axis), leaf, backend and ymode
   ("absolute", else relative). */

const byNumber = (a, b) =>
  String(a).localeCompare(String(b), undefined, { numeric: true });
const memberKey = (params) => JSON.stringify(params);

/* A position's colour: its categorical slot, so neighbours in the order,
   which sit next to each other on the chart, stay apart (every adjacent pair
   of slots does, colour-blind included). Past the slots, which no view has
   needed yet, the order reads as one hue fading toward the page instead. */
const SURFACE = [12, 13, 15];
const positionColor = (position, positions) =>
  positions.every((p) => p < SERIES_COLORS.length)
    ? SERIES_COLORS[position]
    : rampColor(position, positions.length);
function rampColor(position, count) {
  const hex = SERIES_COLORS[0];
  const base = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
  const keep = count > 1 ? 1 - (0.62 * position) / (count - 1) : 1;
  const mixed = base.map((c, i) => Math.round(c * keep + SURFACE[i] * (1 - keep)));
  return `#${mixed.map((c) => c.toString(16).padStart(2, "0")).join("")}`;
}

/* GPU → model → preset → member → the configs it reads that carry both
   view fields, each with the roles that ask for it. */
const carries = (config, view) =>
  view.series.field in config.identity &&
  (config.structured.includes(view.workload.field) ||
    view.workload.field in config.identity);

function buildIndex(list, view) {
  const gpus = new Map();
  for (const config of list.configs) {
    if (!carries(config, view)) continue;
    if (!gpus.has(config.gpu)) gpus.set(config.gpu, new Map());
    const models = gpus.get(config.gpu);
    for (const use of config.uses) {
      const model = presetCheckpoint(use.preset);
      if (!models.has(model)) models.set(model, new Map());
      const presets = models.get(model);
      if (!presets.has(use.preset))
        presets.set(use.preset, {
          preset: use.preset,
          members: new Map(),
          configs: new Map(),
        });
      const d = presets.get(use.preset);
      const key = memberKey(use.params);
      d.members.set(key, use.params);
      if (!d.configs.has(key)) d.configs.set(key, new Map());
      const configs = d.configs.get(key);
      if (!configs.has(config.id))
        configs.set(config.id, { config, roles: new Set() });
      use.roles.forEach((role) => configs.get(config.id).roles.add(role));
    }
  }
  return gpus;
}

/* One member's configs by leaf: configs that agree on everything but the
   series field (the roles that ask, every other config value) are one leaf,
   a line each. */
function leavesOf(entries, view) {
  const field = view.series.field;
  const leaves = new Map();
  for (const entry of entries) {
    const { config } = entry;
    const others = Object.entries(config.identity).filter(([key]) => key !== field);
    const key = JSON.stringify([[...entry.roles].sort(), others.sort()]);
    if (!leaves.has(key))
      leaves.set(key, { roles: [...entry.roles].sort(), lines: new Map() });
    const lines = leaves.get(key).lines;
    const position = config.identity[field];
    if (!lines.has(position)) lines.set(position, config);
  }
  return [...leaves.values()].map((leaf) => {
    const lines = [...leaf.lines]
      .sort(([a], [b]) => a - b)
      .map(([position, config]) => ({ position, config }));
    return { ...leaf, lines, id: lines[0].config.id };
  });
}

/* What tells leaves apart: the steps of their role paths that differ
   ("mtp.step_0" against "mtp.recurrent"), else the binding values that do. */
function leafNames(leaves) {
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
  return new Set(texts).size === leaves.length
    ? texts
    : leaves.map((leaf) => leaf.id.slice(0, 6));
}

export function SeriesExplorer({ kernel, catalog, list, query, update }) {
  const view = kernel.view;
  const index = buildIndex(list, view);
  const models = catalog.models;
  const rank = (stem) => {
    const i = models.findIndex((m) => m.checkpoint.split("/").at(-1) === stem);
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
        No public deployment reads a config that carries both of this view&apos;s
        fields.
      </p>
    );
  const gpu = index.has(query.cgpu) ? query.cgpu : gpuNames[0];
  const byModel = index.get(gpu);
  const stems = [...byModel.keys()].sort((a, b) => rank(a) - rank(b));
  const stem = stems.find((m) => m === query.cmodel) ?? stems[0];
  const byPreset = byModel.get(stem);
  const presetIds = [...byPreset.keys()].sort(byNumber);
  // The default draws the most lines: the view is about comparing them.
  const widest = (d, key) =>
    Math.max(
      ...leavesOf([...d.configs.get(key).values()], view).map(
        (l) => l.lines.length,
      ),
    );
  const widestMember = (d) =>
    [...d.members.keys()].reduce((best, key) =>
      widest(d, key) > widest(d, best) ? key : best,
    );
  const depId = byPreset.has(query.dep)
    ? query.dep
    : [...presetIds].sort(
        (a, b) =>
          widest(byPreset.get(b), widestMember(byPreset.get(b))) -
          widest(byPreset.get(a), widestMember(byPreset.get(a))),
      )[0];
  const d = byPreset.get(depId);
  const axes = presetAxes(d.members);
  const asked = Object.fromEntries(
    axes.map((axis) => [axis.name, query[`m.${axis.name}`]]),
  );
  const memberId =
    [...d.members.keys()].find((key) =>
      memberMatches({ params: d.members.get(key) }, asked, Object.keys(asked)),
    ) ?? widestMember(d);
  const params = d.members.get(memberId);
  const leaves = leavesOf([...d.configs.get(memberId).values()], view);
  const names = leafNames(leaves);
  const leafIndex = Math.max(
    0,
    leaves.findIndex((l) => l.id === query.leaf),
  );
  const leaf = leaves[leafIndex];

  const memberQuery = (p) =>
    Object.fromEntries(Object.entries(p).map(([name, v]) => [`m.${name}`, v]));
  const cleared = Object.fromEntries(
    Object.keys(query)
      .filter((key) => key.startsWith("m."))
      .map((key) => [key, ""]),
  );
  const here = { cgpu: gpu, cmodel: stem, dep: depId, ...memberQuery(params) };

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
                  onClick={() =>
                    update({ ...cleared, cgpu: name, dep: "", leaf: "" })
                  }
                />
              ))
            )}
          </Level>
          <Level label="Model">
            {stems.map((m) => (
              <ToggleTag
                key={m}
                type="family"
                value={modelFamily(models, m)}
                pressed={m === stem}
                title={m}
                onClick={() =>
                  update({ ...cleared, cgpu: gpu, cmodel: m, dep: "", leaf: "" })
                }
              >
                {modelName(models, m)}
              </ToggleTag>
            ))}
          </Level>
        </div>
        <div className={s.levels}>
          <Level label="Deployment">
            {presetIds.map((id) => (
              <ToggleTag
                key={id}
                type="choice"
                value={id}
                pressed={id === depId}
                title={presetArch(id)}
                onClick={() =>
                  update({ ...cleared, cgpu: gpu, cmodel: stem, dep: id, leaf: "" })
                }
              >
                {archName(catalog, id)}
              </ToggleTag>
            ))}
          </Level>
          <MemberPicker
            axes={axes}
            members={[...d.members.values()].map((p) => ({ params: p }))}
            current={params}
            onPick={(p) => update({ ...here, ...memberQuery(p), leaf: "" })}
          />
          {leaves.length > 1 && (
            <Level label="Layer">
              {leaves.map((l, i) => (
                <ToggleTag
                  key={l.id}
                  type="choice"
                  value={l.id}
                  pressed={i === leafIndex}
                  title={l.roles.join("\n")}
                  onClick={() => update({ ...here, leaf: l.id })}
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
        key={leaf.lines.map((l) => l.config.id).join()}
        kernel={kernel}
        catalog={catalog}
        view={view}
        leaf={leaf}
        preset={depId}
        params={params}
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
  preset,
  params,
  query,
  update,
}) {
  const [details, setDetails] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    Promise.all(
      leaf.lines.map(({ config }) => loadConfig(kernel.kind, config.id)),
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
      preset={preset}
      params={params}
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
  preset,
  params,
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
  const count = leaf.lines.length;
  const lineLabel = (position) => `${view.series.label} ${position + 1}`;

  // Relative: each line divided, cell by cell, by the first in the order at
  // the same x. A cell the reference did not measure has no ratio, and none
  // is made up. Log y and the peak stay as set, for the absolute view.
  const reference = lineLabel(leaf.lines[0].position);
  const relative = count > 1 && query.ymode !== "absolute";
  const wantLogY = scale.includes("logy");
  const logY = wantLogY && !relative;
  const wantPeak = query.peak !== "off";
  const showPeak = wantPeak && !relative;
  const measured = (p) => p.measured[backend]?.[y] ?? null;
  const base = new Map(slices[0].map((p) => [p.coords[xi], measured(p)]));
  const pointOf = (p) => {
    const x = p.coords[xi];
    const v = measured(p);
    if (!relative)
      return { x, y: v == null || (logY && v <= 0) ? null : v, record: p };
    const by = base.get(x);
    return by > 0 && v != null
      ? { x, y: v / by, abs: v, record: p }
      : {
          x,
          y: null,
          record: p,
          missing: v == null ? null : `no ${reference} value`,
        };
  };
  const positions = leaf.lines.map((l) => l.position);
  const series = leaf.lines.map(({ position, config }, i) => ({
    key: config.id,
    label: lineLabel(position),
    color: positionColor(position, positions),
    width: position === 0 ? 4.5 : 2,
    points: slices[i].map(pointOf),
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
  const yLabel = relative
    ? `${metric?.label} relative to ${reference}`
    : metric?.label;
  const peak =
    showPeak && y
      ? peakFor(kernel, y, "series", { gpu: first.gpu, ...first.fixed }, catalog)
      : null;
  const unit = argUnit(kernel, xName);
  const xLabel = unit && unit !== "bytes" ? `${xName} (${unit})` : xName;

  return (
    <>
      <div className={k.chartColumn}>
        <ChartBar
          kernel={kernel}
          metrics={metrics}
          y={y}
          logX={logX}
          logY={wantLogY}
          positiveX={positiveX}
          showPeak={wantPeak}
          relative={
            count > 1 && {
              on: relative,
              reference,
              set: (on) => update({ ymode: on ? "" : "absolute" }),
            }
          }
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
            yLabel={yLabel}
            yUnit={relative ? "" : metric.unit}
            logX={logX}
            logY={logY}
            yFit={relative}
            peak={peak}
            cells={cells}
            formatY={
              relative
                ? (p) =>
                    `${formatNumber(p.y)}× · ${formatNumber(p.abs)} ${metric.unit}`
                : undefined
            }
            describe={`${yLabel} of ${kernel.kind} over ${xName}, one line per ${view.series.label.toLowerCase()} of ${count}, ${backend}, ${archName(catalog, preset)} ${paramsText(params)}.`}
            note={
              relative
                ? `Each line is one config's grid for ${backend}, divided cell by cell by ${reference}'s, which leads the order and so lies flat at 1; a cell ${reference} did not measure is left out. The ticks along the bottom are its cells.`
                : `Each line is one config's grid for ${backend}; ${reference}, drawn thickest, leads the order. The ticks along the bottom are its cells.`
            }
            foot={() => null}
          />
        ) : (
          <p className={k.note}>
            {relative && series.some((serie) => serie.points.some((p) => p.missing))
              ? `${reference} has no ${backend} cell to divide the others by.`
              : "No backend has measured a cell of these configs yet."}
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
                <span className={s.muted}>{view.workload.doc}</span>
                <span className={s.muted}>
                  The parameter set below reads one; its preset names it.
                </span>
              </dd>
            </div>
            <div>
              <dt>Parameter set</dt>
              <dd>
                <a
                  className={g.use}
                  href={memberHref(preset, params)}
                  title={preset}
                >
                  <strong>
                    {modelName(catalog.models, presetCheckpoint(preset))} ·{" "}
                    {archName(catalog, preset)}
                  </strong>
                  {Object.keys(params).length > 0 && (
                    <span>{paramsText(params)}</span>
                  )}
                </a>
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
            relative={relative && reference}
          />
        )}
      </div>
    </>
  );
}

/* Relative, the reference keeps its absolute column, so every value can be
   read back, and the other lines show their ratio to it. */
function LinesTable({ kernel, xName, xs, series, metric, relative }) {
  const [all, setAll] = useState(false);
  const at = (serie, x) => {
    const p = serie.points.find((point) => point.x === x);
    return relative && serie === series[0] ? p?.abs : p?.y;
  };
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
                    {relative && serie !== series[0]
                      ? `relative to ${relative}`
                      : `${metric.label} ${metric.unit}`}
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
  Boolean(kernel.view) && list.configs.some((c) => carries(c, kernel.view));
