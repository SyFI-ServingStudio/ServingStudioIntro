import { Download, ExternalLink } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { ChartBar } from "./ChartBar";
import { Level } from "./Level";
import {
  apiUrl,
  archName,
  argUnit,
  byNumber,
  downloadText,
  formatNumber,
  formatValue,
  loadConfig,
  metricDoc,
  metricNames,
  modelFamily,
  modelName,
  peakFor,
  presetArch,
  memberKey,
  presetCheckpoint,
  shortGpu,
  sortGpus,
  sortModels,
} from "./kernelData";
import { memberHref, paramsText } from "../models/modelData";
import { memberSubset } from "./memberSubset";
import { PerfChart, seriesColor } from "./PerfChart";
import { Tag, ToggleTag, tagColor } from "./Tag";
import picker from "./ConfigPicker.module.css";
import k from "./KernelDetail.module.css";
import s from "./GridExplorer.module.css";

/* The kernel configs public deployments read this kind's rows with, one
   chart per config on the grid it reads its cost from: GPU, then the model,
   then the config, then its cells along one cache axis with a line per
   backend.

   Everything here comes from /configs (which configs exist, which preset
   members use them) and /configs/{id} (the cells). URL keys: cgpu, cmodel
   (a checkpoint's repository name), config (its id), cx (the cache axis on
   x) and at.<axis> (the value of each other axis). */

// A kernel is built under a dotted role, its scope path outermost first
// ("unified.body.attention.qkv_proj"); the last step names the op.
const opOf = (role) => role.split(".").at(-1);
const measuredCells = (config) => Math.max(0, ...Object.values(config.measured));
// Entries a config's role list shows before the rest fold away.
const LIST_SHOWN = 4;

/* The axes a preset's members tell apart, with the values they take. */
export function presetAxes(members) {
  const all = [...members.values()];
  const names = Object.keys(all[0] ?? {});
  return names.map((name) => ({
    name,
    values: [...new Set(all.map((params) => params[name]))].sort(byNumber),
  }));
}

/* Which members of a preset something covers, by the axes that decide it:
   "ep_size 4", or its few combinations of those axes, else a count. */
const MAX_COMBINATIONS = 4;
function membersText(d, members) {
  const all = [...d.members.entries()];
  const subset = memberSubset(
    all.map(([, params]) => params),
    all.map(([key]) => members.includes(key)),
  );
  if (subset?.values)
    return subset.axes
      .map(
        (name, i) =>
          `${name} ${subset.values[i].map((v) => formatValue(v)).join(" · ")}`,
      )
      .join(", ");
  if (subset && subset.tuples.length <= MAX_COMBINATIONS)
    return subset.tuples
      .map((tuple) =>
        subset.axes.map((name, i) => `${name} ${formatValue(tuple[i])}`).join(", "),
      )
      .join("; ");
  return `${members.length} of ${all.length} configurations`;
}

/* A preset as its chips read it: the arch, then each axis its members move,
   with the values they take. */
const presetLabel = (d) => {
  const axes = presetAxes(d.members)
    .filter((axis) => axis.values.length > 1)
    .map(
      (axis) =>
        `${axis.name} ${axis.values.map((v) => formatValue(v)).join(" · ")}`,
    );
  return axes.length ? `${d.name} (${axes.join("; ")})` : d.name;
};

/* configs → GPU → model → preset → the configs its members read, each with
   the roles that read it and which members ask. A preset knows its members
   by the ones that read this kind. */
function buildIndex(list, catalog) {
  const gpus = new Map();
  for (const config of list.configs) {
    if (!gpus.has(config.gpu)) gpus.set(config.gpu, new Map());
    const models = gpus.get(config.gpu);
    for (const use of config.uses) {
      const model = presetCheckpoint(use.preset);
      if (!models.has(model)) models.set(model, new Map());
      const presets = models.get(model);
      if (!presets.has(use.preset))
        presets.set(use.preset, {
          preset: use.preset,
          name: archName(catalog, use.preset),
          members: new Map(),
          entries: new Map(),
        });
      const d = presets.get(use.preset);
      const member = memberKey(use.params);
      d.members.set(member, use.params);
      if (!d.entries.has(config.id))
        d.entries.set(config.id, { config, roles: new Set(), members: new Set() });
      const entry = d.entries.get(config.id);
      use.roles.forEach((role) => entry.roles.add(role));
      entry.members.add(member);
    }
  }
  return gpus;
}

/* A model's presets that read the same configs, each with every member,
   read as one group. A preset some of whose configs only some members read
   stays on its own, so its captions can name those members. */
function groupsOf(byPreset) {
  const groups = new Map();
  for (const d of byPreset.values()) {
    const whole = [...d.entries.values()].every(
      (e) => e.members.size === d.members.size,
    );
    const key = whole ? [...d.entries.keys()].sort().join() : `preset ${d.preset}`;
    if (!groups.has(key))
      groups.set(key, {
        presets: [],
        entries: [...d.entries.values()].map((e) => ({
          ...e,
          roles: new Set(e.roles),
        })),
      });
    const group = groups.get(key);
    group.presets.push(d);
    // Roles of every merged preset.
    group.entries.forEach((entry) =>
      d.entries.get(entry.config.id).roles.forEach((r) => entry.roles.add(r)),
    );
  }
  return [...groups.values()]
    .map((g) => ({
      ...g,
      presets: g.presets.sort((a, b) => byNumber(presetLabel(a), presetLabel(b))),
      labels: g.presets.map(presetLabel).sort(byNumber),
    }))
    .sort((a, b) => byNumber(a.labels[0], b.labels[0]));
}

const valueOf = (config, key) =>
  key in config.fixed ? config.fixed[key] : config.identity[key];
const tupleOf = (config, keys) =>
  JSON.stringify(keys.map((key) => valueOf(config, key)));

/* What tells a set of configs apart on their chips: the profile.db args that
   differ between them, then, while two chips still read alike, each Rust
   config value no DB column carries (a folded rank position, say) that
   splits some of them. */
function distinguishing(configs) {
  const differs = (key) =>
    new Set(configs.map((c) => JSON.stringify(valueOf(c, key)))).size > 1;
  const fixedKeys = [...new Set(configs.flatMap((c) => Object.keys(c.fixed)))];
  const keys = fixedKeys.filter(differs);
  const extra = [
    ...new Set(configs.flatMap((c) => Object.keys(c.identity))),
  ].filter((key) => !fixedKeys.includes(key) && differs(key));
  const distinct = (list) => new Set(configs.map((c) => tupleOf(c, list))).size;
  for (const key of extra) {
    if (distinct(keys) === configs.length) break;
    if (distinct([...keys, key]) > distinct(keys)) keys.push(key);
  }
  return keys;
}

export function GridExplorer({ kernel, catalog, list, query, update }) {
  const index = useMemo(() => buildIndex(list, catalog), [list, catalog]);
  const models = catalog.models;
  // A link may name a config alone: it opens on that config's GPU and model.
  const named = list.configs.find((c) => c.id === query.config);

  const gpuNames = sortGpus(index.keys(), catalog);
  const gpu = index.has(query.cgpu) ? query.cgpu : (named?.gpu ?? gpuNames[0]);
  const byModel = index.get(gpu);
  const stems = sortModels(byModel.keys(), catalog);
  const namedModel =
    named?.gpu === gpu ? presetCheckpoint(named.uses[0].preset) : null;
  const stem =
    stems.find((m) => m === query.cmodel) ??
    stems.find((m) => m === namedModel) ??
    stems[0];
  const groups = groupsOf(byModel.get(stem));
  const entries = [
    ...new Map(
      groups.flatMap((g) => g.entries).map((e) => [e.config.id, e]),
    ).values(),
  ];
  const configs = entries.map((e) => e.config);
  // Keys read in the kernel's argument order; Rust-only config values last.
  const argIndex = (key) => {
    const i = kernel.args.findIndex((a) => a.name === key);
    return i < 0 ? kernel.args.length : i;
  };
  const ordered = (list) => [...list].sort((a, b) => argIndex(a) - argIndex(b));
  const allOps = new Set(entries.flatMap((e) => [...e.roles].map(opOf)));
  /* A chip names what tells its config apart from every config on this GPU
     for the same op, whichever model or deployment uses it: two deployment
     groups can each hold one q_absorb config that differ only in
     num_batches (EP4 against EP8), and the chips must say so. A field every
     such config shares is left out. */
  const opsOf = (entry) => [...new Set([...entry.roles].map(opOf))].sort().join();
  const keysByOps = useMemo(() => {
    const byOps = new Map();
    for (const presets of index.get(gpu).values())
      for (const { entries: claimed } of presets.values())
        for (const e of claimed.values()) {
          const ops = opsOf(e);
          if (!byOps.has(ops)) byOps.set(ops, new Map());
          byOps.get(ops).set(e.config.id, e.config);
        }
    return new Map(
      [...byOps].map(([ops, set]) => [
        ops,
        ordered(distinguishing([...set.values()])),
      ]),
    );
    // `ordered` only reads the kernel's argument order.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [index, gpu, kernel]);
  const keysOf = (entry) => keysByOps.get(opsOf(entry)) ?? [];
  const keys = ordered([...new Set(entries.flatMap(keysOf))]);
  const best = [...configs].sort((a, b) => measuredCells(b) - measuredCells(a))[0];
  const config = configs.find((c) => c.id === query.config) ?? best;

  const pick = (patch) =>
    update({ cgpu: gpu, cmodel: stem, config: config.id, ...patch });

  const chipText = (entry) => {
    const ops = [...new Set([...entry.roles].map(opOf))];
    const parts = keysOf(entry).map(
      (key) =>
        // A config without this value (another variant's field) reads "–".
        `${key} ${formatValue(valueOf(entry.config, key) ?? "–", argUnit(kernel, key))}`,
    );
    return {
      ops: allOps.size > 1 ? ops.join(", ") : null,
      text: parts.join(" · "),
    };
  };

  /* A config only some of a preset's members use (one max_model_len of
     several, say) goes under a caption naming them, after the configs every
     member uses. Configs that still read alike differ in a structured config
     value no chip can show (a recorded expert-demand table, say); their chips
     add the config's short id. */
  const chipsOf = (group) => {
    const [d] = group.presets;
    const order = [...d.members.keys()];
    const parts = new Map();
    for (const e of group.entries) {
      const members = [...e.members].sort(
        (a, b) => order.indexOf(a) - order.indexOf(b),
      );
      const whole = members.length === d.members.size;
      const id = whole ? "all" : members.join("|");
      if (!parts.has(id))
        parts.set(id, {
          caption: whole ? null : `${membersText(d, members)} only`,
          order: whole ? [-1] : members.map((m) => order.indexOf(m)),
          entries: [],
        });
      parts.get(id).entries.push(e);
    }
    const label = (e) => {
      const { ops, text } = chipText(e);
      return `${ops}|${text}`;
    };
    return [...parts.values()]
      .sort((a, b) => a.order[0] - b.order[0] || a.order.length - b.order.length)
      .map((part) => {
        const labels = part.entries.map(label);
        const alike = new Set(labels.filter((l, i) => labels.indexOf(l) !== i));
        return { ...part, alike: (e) => alike.has(label(e)) };
      });
  };

  return (
    <div className={k.explorer}>
      <section className={k.controls} aria-label="Kernel config">
        <div className={s.levels}>
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
                  count={list.configs.filter((c) => c.gpu === name).length}
                  onClick={() => update({ cgpu: name })}
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
                onClick={() => update({ cgpu: gpu, cmodel: m })}
              >
                {modelName(models, m)}
              </ToggleTag>
            ))}
          </Level>
        </div>

        <div className={`${picker.level} ${picker.shapeLevel} ${s.configs}`}>
          <span className={picker.levelName}>
            Kernel config
            <span className={picker.levelCount}>
              {configs.length} on {shortGpu(gpu)}
              {allOps.size === 1 && `, all for ${[...allOps][0]}`}
              {keys.length > 0 && `, told apart by ${keys.join(", ")}`}
            </span>
          </span>
          {groups.map((group) => (
            <section
              key={group.labels.join()}
              className={picker.model}
              style={{
                "--family":
                  tagColor("family", modelFamily(models, stem)) ??
                  "var(--c-line-600)",
              }}
              aria-label={group.labels.join("; ")}
            >
              <ul className={s.deployments}>
                {group.presets.map((d) => (
                  <li key={d.preset}>
                    <span className={s.deployment} title={presetArch(d.preset)}>
                      {presetLabel(d)}
                    </span>
                  </li>
                ))}
              </ul>
              {chipsOf(group).map((part) => (
                <div key={part.caption ?? "all"} className={s.part}>
                  {part.caption && <p className={s.caption}>{part.caption}</p>}
                  <div className={picker.choices}>
                    {part.entries.map((entry) => {
                      const { ops, text } = chipText(entry);
                      const c = entry.config;
                      const measured = measuredCells(c);
                      return (
                        <ToggleTag
                          key={c.id}
                          type="choice"
                          value={c.id}
                          pressed={c.id === config.id}
                          faint={measured === 0}
                          title={[
                            ...[...entry.roles].sort(),
                            `config ${c.id}`,
                          ].join("\n")}
                          onClick={() => pick({ config: c.id })}
                        >
                          {ops && <span className={picker.op}>{ops}</span>}
                          {text || (!ops && [...allOps].join(", "))}
                          {part.alike(entry) && (
                            <span className={s.hash}>{c.id.slice(0, 6)}</span>
                          )}
                          <span className={s.coverage}>
                            {measured}/{c.cells}
                          </span>
                        </ToggleTag>
                      );
                    })}
                  </div>
                </div>
              ))}
            </section>
          ))}
        </div>
      </section>

      <GridChart
        key={config.id}
        kernel={kernel}
        catalog={catalog}
        config={config}
        entry={entries.find((e) => e.config === config)}
        query={query}
        update={update}
      />
    </div>
  );
}

function GridChart({ kernel, catalog, config, entry, query, update }) {
  const [detail, setDetail] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    loadConfig(kernel.kind, config.id).then(setDetail, setError);
  }, [kernel.kind, config.id]);

  if (error)
    return (
      <p role="alert" className={k.note}>
        This config&apos;s grid did not load ({error.message}).
      </p>
    );
  if (!detail)
    return (
      <p role="status" className={k.loading}>
        Loading {config.cells.toLocaleString("en-US")} grid cells…
      </p>
    );
  return (
    <GridView
      kernel={kernel}
      catalog={catalog}
      detail={detail}
      entry={entry}
      query={query}
      update={update}
    />
  );
}

function GridView({ kernel, catalog, detail, entry, query, update }) {
  const coords = detail.cache_coords;
  const axes = detail.axes;
  const points = detail.points;
  const moving = coords.filter((_, i) => axes[i].length > 1);
  const xName = moving.includes(query.cx) ? query.cx : (moving[0] ?? coords[0]);
  const xi = coords.indexOf(xName);
  const hasMeasured = (p) => Object.keys(p.measured).length > 0;

  // Every other axis is held at one value: the URL's, else the one with the
  // most measured cells.
  const at = {};
  coords.forEach((name, i) => {
    if (i === xi) return;
    const wanted = Number(query[`at.${name}`]);
    if (axes[i].includes(wanted)) at[name] = wanted;
    else {
      const count = (v) =>
        points.filter((p) => p.coords[i] === v && hasMeasured(p)).length;
      at[name] = [...axes[i]].sort((a, b) => count(b) - count(a) || a - b)[0];
    }
  });
  const slice = points
    .filter((p) => coords.every((name, i) => i === xi || p.coords[i] === at[name]))
    .sort((a, b) => a.coords[xi] - b.coords[xi]);

  const metrics = metricNames(kernel).filter((m) =>
    points.some((p) => Object.values(p.measured).some((row) => row[m] != null)),
  );
  const y = metrics.includes(query.y)
    ? query.y
    : metrics.includes(kernel.default_metric)
      ? kernel.default_metric
      : metrics[0];
  const xs = axes[xi];
  const positiveX = xs[0] > 0;
  const scale =
    query.scale || (positiveX && xs.at(-1) / xs[0] > 16 ? "logx" : "linear");
  const logX = positiveX && scale.includes("logx");
  const logY = scale.includes("logy");
  const showPeak = query.peak !== "off";

  // Backends keep their colour across configs: the kind's registered backends
  // in order, then any that measured without being registered now.
  const measuredBackends = [
    ...new Set(points.flatMap((p) => Object.keys(p.measured))),
  ];
  const allBackends = [
    ...new Set([...Object.keys(kernel.backends), ...measuredBackends]),
  ];
  const present = allBackends.filter((b) => slice.some((p) => p.measured[b]));
  const value = (p, b) => {
    const v = p.measured[b]?.[y];
    return v == null || (logY && v <= 0) ? null : v;
  };
  const series = present.map((b) => ({
    key: b,
    label: b,
    color: seriesColor(allBackends, present, b),
    points: slice.map((p) => ({ x: p.coords[xi], y: value(p, b), record: p })),
  }));
  const cells = slice.map((p) => ({
    x: p.coords[xi],
    status: !p.feasible ? "infeasible" : hasMeasured(p) ? "measured" : "missing",
  }));

  const metric = metricDoc(kernel, y);
  const selection = { gpu: detail.gpu, ...detail.fixed };
  const peak =
    showPeak && y ? peakFor(kernel, y, "backend", selection, catalog) : null;
  const unit = argUnit(kernel, xName);
  const xLabel = unit && unit !== "bytes" ? `${xName} (${unit})` : xName;
  const measured = slice.filter(hasMeasured).length;
  const infeasible = slice.filter((p) => !p.feasible).length;
  const pinned = coords.filter((_, i) => i !== xi);

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
        {(moving.length > 1 || pinned.length > 0) && (
          <div className={s.axes}>
            {moving.length > 1 && (
              <Level label="Grid axis on x">
                {moving.map((name) => (
                  <ToggleTag
                    key={name}
                    type="choice"
                    value={name}
                    pressed={name === xName}
                    onClick={() => update({ cx: name })}
                  >
                    {name}
                  </ToggleTag>
                ))}
              </Level>
            )}
            {pinned.map((name) => {
              const i = coords.indexOf(name);
              return (
                <Level key={name} label={name}>
                  {axes[i].length === 1 ? (
                    <Tag type="choice" value={name}>
                      {formatValue(axes[i][0], argUnit(kernel, name))}
                    </Tag>
                  ) : (
                    axes[i].map((v) => (
                      <ToggleTag
                        key={v}
                        type="choice"
                        value={String(v)}
                        small
                        pressed={v === at[name]}
                        faint={
                          !points.some((p) => p.coords[i] === v && hasMeasured(p))
                        }
                        onClick={() => update({ [`at.${name}`]: String(v) })}
                      >
                        {formatValue(v, argUnit(kernel, name))}
                      </ToggleTag>
                    ))
                  )}
                </Level>
              );
            })}
          </div>
        )}

        {series.length ? (
          <PerfChart
            series={series}
            xName={xName}
            xLabel={xLabel}
            xUnit={unit}
            colorName="Backend"
            yLabel={metric.label}
            yUnit={metric.unit}
            logX={logX}
            logY={logY}
            peak={peak}
            cells={cells}
            describe={`${metric.label} of ${kernel.kind} over the simulator's ${xName} grid, one line per backend. ${measured} of ${slice.length} cells measured.`}
            note="Each point is one cell of the grid the simulator reads for this config; between cells it interpolates. The ticks along the bottom are the cells: solid where a backend measured it, dashed where none did, a cross where the kernel cannot run."
            foot={(hovered) => (
              <CellFoot point={hovered[0]?.record} detail={detail} />
            )}
          />
        ) : (
          <p className={k.note}>
            No backend has measured a cell of this config
            {pinned.length ? " at the values picked above" : ""}. The simulator
            profiles missing cells the first time a run reads them.
          </p>
        )}
        <p className={k.note}>
          {measured} of {slice.length} cells on this line measured
          {infeasible ? `, ${infeasible} the kernel cannot run` : ""}.
        </p>
      </div>

      <div className={k.rowsColumn}>
        <ConfigFacts
          kernel={kernel}
          catalog={catalog}
          detail={detail}
          entry={entry}
        />
        <CellTable
          kernel={kernel}
          detail={detail}
          slice={slice}
          coords={coords}
          backends={present}
          y={y}
        />
      </div>
    </>
  );
}

function CellFoot({ point, detail }) {
  if (!point) return null;
  const outliers = Object.entries(point.measured)
    .filter(([, row]) => row.outlier)
    .map(([b]) => b);
  // On a one-axis grid the tooltip head already names the cell.
  if (detail.cache_coords.length === 1 && point.feasible && !outliers.length)
    return null;
  const others = detail.cache_coords
    .map((name, i) => `${name} ${formatNumber(point.coords[i])}`)
    .join(", ");
  return (
    <p className={s.foot}>
      Grid cell {others}
      {!point.feasible && ". The kernel cannot run this cell"}
      {outliers.length > 0 && `. Flagged as an outlier: ${outliers.join(", ")}`}
    </p>
  );
}

/* What the picked config holds fixed, and who asked for it. */
/** A list that shows its first LIST_SHOWN entries and folds the rest behind a toggle. */
function FoldedList({ items, render }) {
  const [all, setAll] = useState(false);
  return (
    <dd className={s.stack}>
      {(all ? items : items.slice(0, LIST_SHOWN)).map(render)}
      {items.length > LIST_SHOWN && (
        <button
          type="button"
          className={s.moreSources}
          onClick={() => setAll(!all)}
        >
          {all ? "Show fewer" : `and ${items.length - LIST_SHOWN} more`}
        </button>
      )}
    </dd>
  );
}

function ConfigFacts({ kernel, catalog, detail, entry }) {
  const fixed = Object.entries(detail.fixed);
  const own = Object.entries(detail.identity).filter(
    ([key]) => !(key in detail.fixed) && !detail.structured.includes(key),
  );
  const url = apiUrl(`kernels/${kernel.kind}/configs/${detail.id}`);
  return (
    <div className={s.facts}>
      <h2>This config</h2>
      <dl>
        <div>
          <dt>Fixed arguments</dt>
          <dd>
            {fixed.map(([key, v]) => (
              <span key={key} className={s.fact}>
                <code>{key}</code> {formatValue(v, argUnit(kernel, key))}
              </span>
            ))}
          </dd>
        </div>
        <div>
          <dt>Swept with the grid</dt>
          <dd>
            {detail.swept.map((key) => (
              <code key={key} className={s.fact}>
                {key}
              </code>
            ))}
          </dd>
        </div>
        {(own.length > 0 || detail.structured.length > 0) && (
          <div>
            <dt>Other config values</dt>
            <dd>
              {own.map(([key, v]) => (
                <span key={key} className={s.fact}>
                  <code>{key}</code> {formatValue(v)}
                </span>
              ))}
              {detail.structured.map((key) => (
                <span key={key} className={s.fact}>
                  <code>{key}</code> in the API response
                </span>
              ))}
            </dd>
          </div>
        )}
        <div>
          <dt>Used by</dt>
          <FoldedList
            items={detail.uses}
            render={(use) => (
              <a
                key={memberKey([use.preset, use.params])}
                className={s.use}
                href={memberHref(use.preset, use.params)}
                title={[use.preset, ...use.roles].join("\n")}
              >
                <strong>
                  {modelName(catalog.models, presetCheckpoint(use.preset))} ·{" "}
                  {archName(catalog, use.preset)}
                </strong>
                {Object.keys(use.params).length > 0 && (
                  <span>{paramsText(use.params)}</span>
                )}
              </a>
            )}
          />
        </div>
        {entry && (
          <div>
            <dt>Asked for by</dt>
            <FoldedList
              items={[...entry.roles].sort()}
              render={(role) => <code key={role}>{role}</code>}
            />
          </div>
        )}
        <div>
          <dt>Grid</dt>
          <dd>
            {detail.cache_coords
              .map((name, i) => `${name} ${detail.axes[i].length}`)
              .join(" × ")}{" "}
            cells, config <code>{detail.id}</code>{" "}
            <a href={url} target="_blank" rel="noreferrer" className={k.external}>
              JSON
              <ExternalLink size={14} aria-label="opens in a new tab" />
            </a>
          </dd>
        </div>
      </dl>
    </div>
  );
}

function CellTable({ kernel, detail, slice, coords, backends, y }) {
  const [all, setAll] = useState(false);
  const shown = all ? slice : slice.slice(0, 12);
  const metric = metricDoc(kernel, y);
  const metrics = metricNames(kernel);
  const csv = () => {
    const header = [...coords, "feasible", "backend", ...metrics, "outlier"];
    const lines = detail.points.flatMap((p) => {
      const rows = Object.entries(p.measured);
      return (rows.length ? rows : [[null, {}]]).map(([backend, row]) =>
        [
          ...p.coords,
          p.feasible,
          backend ?? "",
          ...metrics.map((m) => row[m] ?? ""),
          row.outlier ?? "",
        ].join(","),
      );
    });
    downloadText(
      `${kernel.kind}-${detail.id}.csv`,
      [header.join(","), ...lines].join("\n"),
      "text/csv",
    );
  };
  return (
    <div className={k.rows}>
      <div className={k.rowsHead}>
        <h2>Cells behind the chart</h2>
        <button type="button" className={k.action} onClick={csv}>
          <Download size={18} aria-hidden="true" />
          Download all {detail.points.length} cells as CSV
        </button>
      </div>
      <div
        className={k.tableScroll}
        tabIndex={0}
        role="region"
        aria-label="Grid cells"
      >
        <table className={k.table}>
          <thead>
            <tr>
              {coords.map((c) => (
                <th key={c} scope="col">
                  {c}
                </th>
              ))}
              {backends.map((b) => (
                <th key={b} scope="col" className={k.num}>
                  <code>{b}</code>
                  <span className={s.unit}>
                    {metric.label} {metric.unit}
                  </span>
                </th>
              ))}
              <th scope="col">Cell</th>
            </tr>
          </thead>
          <tbody>
            {shown.map((p) => (
              <tr key={p.coords.join()}>
                {p.coords.map((v, i) => (
                  <td key={coords[i]} className={k.code}>
                    {formatValue(v, argUnit(kernel, coords[i]))}
                  </td>
                ))}
                {backends.map((b) => (
                  <td key={b} className={k.num}>
                    {p.measured[b] ? formatNumber(p.measured[b][y]) : "–"}
                  </td>
                ))}
                <td className={k.date}>
                  {!p.feasible
                    ? "cannot run"
                    : Object.keys(p.measured).length
                      ? "measured"
                      : "not measured"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {slice.length > 12 && (
        <button type="button" className={k.more} onClick={() => setAll(!all)}>
          {all ? "Show the first 12 cells" : `Show all ${slice.length} cells`}
        </button>
      )}
    </div>
  );
}
