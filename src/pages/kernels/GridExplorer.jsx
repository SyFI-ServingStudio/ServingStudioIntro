import { BadgeCheck, Download, ExternalLink } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { ChartBar } from "./ChartBar";
import {
  apiUrl,
  argUnit,
  downloadText,
  engineName,
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
import { PerfChart, seriesColor } from "./PerfChart";
import { Tag, ToggleTag, tagColor } from "./Tag";
import picker from "./ConfigPicker.module.css";
import k from "./KernelDetail.module.css";
import s from "./GridExplorer.module.css";

/* The kernel configs the simulator registered for this kind, one chart per
   config on the grid it reads its cost from: GPU, then the model, then the
   config, then its cells along one cache axis with a line per backend.

   Everything here comes from /configs (which configs exist, who uses them)
   and /configs/{hash} (the cells). URL keys: cgpu, cmodel, config, cx (the
   cache axis on x) and at.<axis> (the value of each other axis). */

// A kernel is built under a dotted role, its scope path outermost first
// ("unified.body.attention.qkv_proj"); the last step names the op.
const opOf = (role) => role.split(".").at(-1);
const byNumber = (a, b) => a.localeCompare(b, undefined, { numeric: true });
const measuredCells = (config) => Math.max(0, ...Object.values(config.measured));
// Entries a config's role list shows before the rest fold away.
const LIST_SHOWN = 4;
// Where the site shows predictions checked against real serving runs.
const ALIGNMENT_HREF = `${import.meta.env.BASE_URL}features.html#accuracy`;

/* A deployment entry is one #[supported] row: the params the row lists
   several values for (d.varies) hold a list, one value per member. These say
   which members something covers: "max_model_len 524288", or each member
   spelled out when the row varies more than one param. */
function membersText(d, members) {
  if (d.varies.length === 1) {
    const [name] = d.varies;
    return `${name} ${members.map((i) => formatValue(d.members[i].params[name])).join(" · ")}`;
  }
  return members
    .map((i) =>
      d.varies
        .map((name) => `${name} ${formatValue(d.members[i].params[name])}`)
        .join(", "),
    )
    .join("; ");
}
const allMembers = (d, members) => members.length === d.members.length;

/* The label the API gives, with numbers read as the chips read them: a
   param the row lists several values for reads "max_model_len 8,192 · 65,536". */
const deploymentLabel = (d) =>
  [
    d.arch,
    ...Object.entries(d.params).map(
      ([name, v]) =>
        `${name} ${Array.isArray(v) ? v.map((x) => formatValue(x)).join(" · ") : formatValue(v)}`,
    ),
  ].join(", ");

/* "Validated against SGLang serving", from the alignment runs the API found
   for this deployment; "at max_model_len 131072" when only some members ran. */
function validatedText(d) {
  const engines = d.validated_against.map(engineName);
  const against = engines.length
    ? `${engines.join(" and ")} serving`
    : "real serving";
  const members = d.members.flatMap((m, i) => (m.validated ? [i] : []));
  return allMembers(d, members)
    ? `Validated against ${against}`
    : `Validated against ${against} at ${membersText(d, members)}`;
}

/* configs → GPU → model → deployment → the configs it uses, each with the
   roles that use it and which of the deployment's members ask. A config no
   deployment claims (built only at the deployment level) sits under model
   null. */
function buildIndex(list) {
  const deployments = new Map(list.deployments.map((d) => [d.id, d]));
  const gpus = new Map();
  for (const config of list.configs) {
    if (!gpus.has(config.gpu)) gpus.set(config.gpu, new Map());
    const models = gpus.get(config.gpu);
    const claims = config.uses.flatMap((use) =>
      use.deployments.length
        ? use.deployments.map(({ id, members }) => [
            deployments.get(id),
            use,
            members,
          ])
        : [[null, use, []]],
    );
    const named = claims.some(([d]) => d);
    for (const [d, use, members] of claims) {
      if (!d && named) continue;
      const stem = d?.model_config ?? null;
      if (!models.has(stem)) models.set(stem, new Map());
      const byDeployment = models.get(stem);
      const key = d ? d.id : "none";
      if (!byDeployment.has(key))
        byDeployment.set(key, { deployment: d, entries: new Map() });
      const entries = byDeployment.get(key).entries;
      if (!entries.has(config.config_hash))
        entries.set(config.config_hash, {
          config,
          roles: new Set(),
          members: new Set(),
        });
      const entry = entries.get(config.config_hash);
      entry.roles.add(use.role);
      members.forEach((m) => entry.members.add(m));
    }
  }
  return gpus;
}

/* A model's deployments that use the same configs, each by every member,
   read as one group. A deployment some of whose configs only some members
   use stays on its own, so its captions can name those members. */
function groupsOf(byDeployment) {
  const groups = new Map();
  for (const { deployment, entries } of byDeployment.values()) {
    const whole = [...entries.values()].every(
      (e) => !deployment || allMembers(deployment, [...e.members]),
    );
    const key = whole
      ? [...entries.keys()].sort().join()
      : `deployment ${deployment.id}`;
    if (!groups.has(key))
      groups.set(key, { deployments: [], entries: [...entries.values()] });
    const group = groups.get(key);
    group.deployments.push(deployment);
    // Roles of every merged deployment.
    group.entries.forEach((entry) => {
      entries
        .get(entry.config.config_hash)
        .roles.forEach((r) => entry.roles.add(r));
    });
  }
  const labelOf = (d) => d?.label ?? "Built at the deployment level";
  return [...groups.values()]
    .map((g) => ({
      ...g,
      deployments: g.deployments.sort((a, b) => byNumber(labelOf(a), labelOf(b))),
      labels: g.deployments.map(labelOf).sort(byNumber),
    }))
    .sort((a, b) => byNumber(a.labels[0], b.labels[0]));
}

const valueOf = (config, key) =>
  key in config.fixed ? config.fixed[key] : config.config_args[key];
const tupleOf = (config, keys) =>
  JSON.stringify(keys.map((key) => valueOf(config, key)));

/* What tells a model's configs apart on its chips: the profile.db args that
   differ between them, then, while two chips still read alike, the Rust
   config values that no DB column carries (a folded rank position, say). */
function distinguishing(configs) {
  const differs = (key) =>
    new Set(configs.map((c) => JSON.stringify(valueOf(c, key)))).size > 1;
  const fixedKeys = [...new Set(configs.flatMap((c) => Object.keys(c.fixed)))];
  const keys = fixedKeys.filter(differs);
  const extra = [
    ...new Set(configs.flatMap((c) => Object.keys(c.config_args))),
  ].filter((key) => !fixedKeys.includes(key) && differs(key));
  for (const key of extra) {
    const seen = new Set(configs.map((c) => tupleOf(c, keys)));
    if (seen.size === configs.length) break;
    keys.push(key);
  }
  return keys;
}

export function GridExplorer({ kernel, catalog, list, query, update }) {
  const index = useMemo(() => buildIndex(list), [list]);
  const models = catalog.models;
  const rank = (stem) => {
    const i = models.findIndex((m) => m.model_config === stem);
    return i < 0 ? models.length : i;
  };

  // GPUs in the catalog's order (most rows first), then any it does not list.
  const gpuNames = [...index.keys()].sort((a, b) => {
    const order = catalog.gpus.map((g) => g.name);
    return (
      (order.indexOf(a) + 1 || order.length + 1) -
      (order.indexOf(b) + 1 || order.length + 1)
    );
  });
  const gpu = index.has(query.cgpu) ? query.cgpu : gpuNames[0];
  const byModel = index.get(gpu);
  const stems = [...byModel.keys()].sort((a, b) => rank(a) - rank(b));
  const modelKey = (stem) => stem ?? "";
  const stem = stems.find((m) => modelKey(m) === query.cmodel) ?? stems[0];
  const groups = groupsOf(byModel.get(stem));
  const entries = [
    ...new Map(
      groups.flatMap((g) => g.entries).map((e) => [e.config.config_hash, e]),
    ).values(),
  ];
  const configs = entries.map((e) => e.config);
  // Keys read in the kernel's argument order; Rust-only config values last.
  const argIndex = (key) => {
    const i = kernel.args.findIndex((a) => a.name === key);
    return i < 0 ? kernel.args.length : i;
  };
  const ordered = (list) => [...list].sort((a, b) => argIndex(a) - argIndex(b));
  const keys = ordered(distinguishing(configs));
  const allOps = new Set(entries.flatMap((e) => [...e.roles].map(opOf)));
  const best = [...configs].sort((a, b) => measuredCells(b) - measuredCells(a))[0];
  const config = configs.find((c) => c.config_hash === query.config) ?? best;

  const pick = (patch) =>
    update({
      cgpu: gpu,
      cmodel: modelKey(stem),
      config: config.config_hash,
      ...patch,
    });

  const chipText = (entry, chipKeys) => {
    const ops = [...new Set([...entry.roles].map(opOf))];
    const parts = chipKeys.map(
      (key) =>
        // A config without this value (another variant's field) reads "–".
        `${key} ${formatValue(valueOf(entry.config, key) ?? "–", argUnit(kernel, key))}`,
    );
    return {
      ops: allOps.size > 1 ? ops.join(", ") : null,
      text: parts.join(" · "),
    };
  };

  /* A group's chips name only what differs inside the group (its deployment
     label already says the rest). A config only some of a deployment's
     members use (one max_model_len of several, say) goes under a caption
     naming them, after the configs every member uses. Configs that still read
     alike differ in a structured config value no chip can show (a recorded
     expert-demand table, say); their chips add the config's short hash. */
  const chipsOf = (group) => {
    const groupKeys =
      group.entries.length > 1
        ? ordered(distinguishing(group.entries.map((e) => e.config)))
        : keys;
    const [deployment] = group.deployments;
    const parts = new Map();
    for (const e of group.entries) {
      const members = [...e.members].sort((a, b) => a - b);
      const whole = !deployment || allMembers(deployment, members);
      const id = whole ? "all" : members.join();
      if (!parts.has(id))
        parts.set(id, {
          caption: whole ? null : `${membersText(deployment, members)} only`,
          order: whole ? [-1] : members,
          keys: groupKeys,
          entries: [],
        });
      parts.get(id).entries.push(e);
    }
    const label = (e) => {
      const { ops, text } = chipText(e, groupKeys);
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
                  count={[...index.get(name).values()].reduce(
                    (n, d) =>
                      n +
                      new Set([...d.values()].flatMap((x) => [...x.entries.keys()]))
                        .size,
                    0,
                  )}
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
                {group.deployments.map((d) => (
                  <li key={d?.id ?? "none"}>
                    <span className={s.deployment}>
                      {d ? deploymentLabel(d) : "Built at the deployment level"}
                    </span>
                    {d?.validated && (
                      <a className={s.validated} href={ALIGNMENT_HREF}>
                        <BadgeCheck size={16} aria-hidden="true" />
                        {validatedText(d)}
                      </a>
                    )}
                  </li>
                ))}
              </ul>
              {chipsOf(group).map((part) => (
                <div key={part.caption ?? "all"} className={s.part}>
                  {part.caption && <p className={s.caption}>{part.caption}</p>}
                  <div className={picker.choices}>
                    {part.entries.map((entry) => {
                      const { ops, text } = chipText(entry, part.keys);
                      const c = entry.config;
                      const measured = measuredCells(c);
                      return (
                        <ToggleTag
                          key={c.config_hash}
                          type="choice"
                          value={c.config_hash}
                          pressed={c.config_hash === config.config_hash}
                          faint={measured === 0}
                          title={[
                            ...[...entry.roles].sort(),
                            `config ${c.config_hash.slice(0, 12)}`,
                          ].join("\n")}
                          onClick={() => pick({ config: c.config_hash })}
                        >
                          {ops && <span className={picker.op}>{ops}</span>}
                          {text || (!ops && [...allOps].join(", "))}
                          {part.alike(entry) && (
                            <span className={s.hash}>
                              {c.config_hash.slice(0, 6)}
                            </span>
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
        key={`${config.config_hash}|${gpu}`}
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

function Level({ label, children }) {
  return (
    <div className={picker.level} role="group" aria-label={label}>
      <span className={picker.levelName}>{label}</span>
      <div className={picker.choices}>{children}</div>
    </div>
  );
}

function GridChart({ kernel, catalog, config, entry, query, update }) {
  const [detail, setDetail] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    loadConfig(kernel.kind, config.config_hash, config.gpu).then(
      setDetail,
      setError,
    );
  }, [kernel.kind, config.config_hash, config.gpu]);

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
        <ConfigFacts kernel={kernel} detail={detail} entry={entry} />
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

function ConfigFacts({ kernel, detail, entry }) {
  const fixed = Object.entries(detail.fixed);
  const own = Object.entries(detail.config_args).filter(
    ([key]) => !(key in detail.fixed),
  );
  const url = apiUrl(
    `kernels/${kernel.kind}/configs/${detail.config_hash}?gpu=${encodeURIComponent(detail.gpu)}`,
  );
  return (
    <div className={s.facts}>
      <h2>This config</h2>
      <dl>
        <div>
          <dt>Fixed profile.db args</dt>
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
        {(own.length > 0 || detail.config_args_omitted.length > 0) && (
          <div>
            <dt>Other config values</dt>
            <dd>
              {own.map(([key, v]) => (
                <span key={key} className={s.fact}>
                  <code>{key}</code> {formatValue(v)}
                </span>
              ))}
              {detail.config_args_omitted.map((key) => (
                <span key={key} className={s.fact}>
                  <code>{key}</code> in the API response
                </span>
              ))}
            </dd>
          </div>
        )}
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
            cells, config <code>{detail.config_hash.slice(0, 12)}</code>{" "}
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
      `${kernel.kind}-${detail.config_hash.slice(0, 12)}.csv`,
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
