import {
  argRole,
  argUnit,
  formatValue,
  isDtype,
  isList,
  shortGpu,
} from "./kernelData";
import { useState } from "react";
import { Tag, ToggleTag, tagColor } from "./Tag";
import s from "./ConfigPicker.module.css";

/* Picking a configuration, one level at a time: GPU, then precision, then the
   shapes measured under those two, then any sweep not on the x axis. Each level
   offers only what exists under the levels above it, and picking one resets
   the levels below to their best-measured choice, so no pick leads to an empty
   chart. */

const isNumber = (values, d) => typeof values[d][0] === "number";
// Past this many values a list argument is picked from a menu.
const LIST_MENU_AFTER = 8;

const PRECISIONS = [
  ["nvfp4", "NVFP4"],
  ["mxfp4", "MXFP4"],
  ["fp8", "FP8"],
  ["fp16", "FP16"],
  ["bf16", "BF16"],
];
const precisionOf = (value) =>
  PRECISIONS.find(([prefix]) =>
    String(value).toLowerCase().startsWith(prefix),
  )?.[1];
// "fp8_e4m3" reads as "FP8 e4m3": the family the main page filters by, then
// the variant.
const formatDtype = (value) => {
  const family = precisionOf(value);
  if (!family) return String(value);
  const rest = String(value).slice(family.length).replace(/^_/, "");
  return rest ? `${family} ${rest}` : family;
};
// The column header for a shape dimension: num_qo_heads is "qo_heads".
const shortName = (d) => d.replace(/^num_/, "");

/* GPU and precision always show, as a lone tag when only one was measured, so
   the ladder reads the same on every kernel. Other dimensions with one value
   are left to the note under the picker. */
export function buildLevels(kernel, values, allDims) {
  const levels = [];
  const role = (d) => argRole(kernel, d);
  const dtype = (d) => isDtype(kernel, d);
  const filterDims = allDims.filter(
    (d) => d === "gpu" || dtype(d) || values[d].length > 1,
  );
  if (filterDims.includes("gpu"))
    levels.push({ id: "gpu", label: "GPU", dims: ["gpu"], kind: "gpu" });
  const dtypes = filterDims.filter(dtype);
  if (dtypes.length) {
    const varying = dtypes.filter((d) => values[d].length > 1);
    levels.push({
      id: "precision",
      label:
        varying.length === 1 && varying[0] !== "dtype"
          ? `Precision (${varying[0]})`
          : "Precision",
      dims: dtypes,
      varying,
      kind: "precision",
    });
  }
  filterDims
    .filter((d) => d !== "gpu" && !dtype(d) && !isNumber(values, d))
    .forEach((d) =>
      levels.push({
        id: d,
        label: d,
        dims: [d],
        kind: "choice",
        list: isList(kernel, d),
      }),
    );
  const shape = filterDims.filter(
    (d) => isNumber(values, d) && role(d) !== "sweep",
  );
  if (shape.length)
    levels.push({
      id: "shape",
      label: shape.length > 1 ? "Shape" : shape[0],
      dims: shape,
      kind: shape.length > 1 ? "list" : "choice",
    });
  filterDims
    .filter((d) => isNumber(values, d) && role(d) === "sweep")
    .forEach((d) =>
      levels.push({ id: d, label: d, dims: [d], kind: "choice", small: true }),
    );
  // Each level formats its values in their units.
  return levels.map((level) => ({
    ...level,
    units: level.dims.map((d) => argUnit(kernel, d)),
  }));
}

// The dimensions a level names on screen, so the note need not repeat them.
export const shownDims = (levels) =>
  levels.flatMap((l) =>
    l.kind === "precision"
      ? l.varying.length
        ? l.varying
        : l.dims.slice(0, 1)
      : l.dims,
  );

const tupleKey = (tuple) => JSON.stringify(tuple.map(String));
const compareTuples = (a, b) => {
  for (let i = 0; i < a.length; i++) {
    const d =
      typeof a[i] === "number"
        ? a[i] - b[i]
        : String(a[i]).localeCompare(String(b[i]));
    if (d) return d;
  }
  return 0;
};

/* The most lines a choice can draw: over every full configuration under it
   (the levels above as picked, this choice, any pick on the levels below),
   the most compared values measured. One line has nothing to compare, so the
   choice recedes. Backends are the exception (no compare is passed): they are
   the default view, and one backend's line is a fine thing to look at. */
function linesUnder(rows, dims, tuple, below, compare) {
  const seen = new Map();
  rows
    .filter((r) => dims.every((d, k) => String(r[d]) === String(tuple[k])))
    .forEach((r) => {
      const key = JSON.stringify(below.map((d) => r[d]));
      if (!seen.has(key)) seen.set(key, new Set());
      seen.get(key).add(String(r[compare.dim]));
    });
  return Math.max(0, ...[...seen.values()].map((values) => values.size));
}

export function ConfigPicker({
  levels,
  records,
  deployments,
  models,
  selection,
  compare,
  onPick,
}) {
  return (
    <div className={s.picker}>
      {levels.map((level, index) => {
        const upstream = levels.slice(0, index).flatMap((l) => l.dims);
        const pool = records.filter((r) =>
          upstream.every((d) => String(r[d]) === selection[d]),
        );
        const pick = (tuple) => {
          const patch = {};
          levels.forEach((l, j) =>
            l.dims.forEach((d, k) => {
              if (j < index) patch[d] = selection[d];
              else if (j === index) patch[d] = String(tuple[k]);
              else patch[d] = "";
            }),
          );
          onPick(patch);
        };
        return (
          <Level
            key={level.id}
            level={level}
            pool={pool}
            records={records}
            selection={selection}
            deployments={deployments}
            models={models}
            below={levels.slice(index + 1).flatMap((l) => l.dims)}
            compare={compare}
            pick={pick}
          />
        );
      })}
    </div>
  );
}

function Level({
  level,
  pool,
  records,
  selection,
  deployments,
  models,
  below,
  compare,
  pick,
}) {
  const tupleOf = (r) => level.dims.map((d) => r[d]);
  const unique = (rows) => {
    const seen = new Map();
    rows.forEach((r) => {
      const tuple = tupleOf(r);
      seen.set(tupleKey(tuple), tuple);
    });
    return [...seen.values()].sort(compareTuples);
  };
  const all = unique(records);
  const available = new Set(unique(pool).map(tupleKey));
  const current = tupleKey(level.dims.map((d) => selection[d]));
  const labelId = `level-${level.id}`;
  const single = (tuple) =>
    compare != null && linesUnder(pool, level.dims, tuple, below, compare) < 2;

  if (level.kind === "list")
    return (
      <ShapeList
        level={level}
        tuples={unique(pool)}
        current={current}
        groups={modelGroups(deployments, models, level, unique(pool), selection)}
        single={single}
        compare={compare}
        pick={pick}
      />
    );

  const label = (tuple) => {
    if (level.kind === "gpu") return shortGpu(tuple[0]);
    if (level.kind === "precision") {
      const shown = level.varying.length ? level.varying : level.dims.slice(0, 1);
      return shown
        .map((d) => {
          const text = formatDtype(tuple[level.dims.indexOf(d)]);
          return shown.length > 1 ? `${d.replace(/_dtype$/, "")} ${text}` : text;
        })
        .join(", ");
    }
    return formatValue(tuple[0], level.units[0]);
  };
  // A combination takes the colour of its most quantized operand.
  const tagValue = (tuple) =>
    level.kind === "gpu"
      ? shortGpu(tuple[0])
      : level.kind === "precision"
        ? PRECISIONS.map(([prefix]) =>
            tuple.find((v) => String(v).toLowerCase().startsWith(prefix)),
          ).find(Boolean)
        : String(tuple[0]);
  const type = level.kind === "choice" ? "choice" : level.kind;

  // Many list values (one per request or expert, hundreds of entries each)
  // would wall the picker with chips; they are offered as a menu instead.
  if (level.list && all.length > LIST_MENU_AFTER)
    return (
      <div className={`${s.level} ${s.menuLevel}`}>
        <label className={s.levelName} htmlFor={labelId}>
          {level.label}
          <span className={s.levelCount}>
            {available.size} of {all.length} under the choices above
          </span>
        </label>
        <select
          id={labelId}
          className={s.menu}
          value={current}
          onChange={(event) => pick(JSON.parse(event.target.value))}
        >
          {all.map((tuple) => {
            const key = tupleKey(tuple);
            return (
              <option key={key} value={key} disabled={!available.has(key)}>
                {label(tuple)}
              </option>
            );
          })}
        </select>
      </div>
    );

  return (
    <div className={s.level} role="group" aria-labelledby={labelId}>
      <span id={labelId} className={s.levelName}>
        {level.label}
      </span>
      <div className={s.choices}>
        {all.length === 1 ? (
          <Tag type={type} value={tagValue(all[0])} title="The only one measured">
            {label(all[0])}
          </Tag>
        ) : (
          all.map((tuple) => {
            const key = tupleKey(tuple);
            return (
              <ToggleTag
                key={key}
                type={type}
                value={tagValue(tuple)}
                small={level.small}
                pressed={key === current}
                disabled={!available.has(key) && key !== current}
                faint={available.has(key) && single(tuple)}
                title={
                  !available.has(key)
                    ? "Not measured under the choices above"
                    : single(tuple)
                      ? `Measured for one ${compare.label} only: one line, nothing to compare`
                      : undefined
                }
                onClick={() => pick(tuple)}
              >
                {label(tuple)}
              </ToggleTag>
            );
          })
        )}
      </div>
    </div>
  );
}

/* The model deployments that ask for the shapes on offer (their supported
   cost trees and the kernel configs the simulator registered), each with the
   shapes it asks for and the layer that asks. A shape counts only if the
   deployment runs on the GPU picked above and in the precision picked above,
   so a label never names a model for a measurement it would not make. Models
   keep the catalog's order. */
function modelGroups(deployments, models, level, tuples, selection) {
  const offered = new Set(tuples.map(tupleKey));
  const order = (d) => models.findIndex((m) => m.model_config === d.model_config);
  return (
    deployments
      .filter((d) => !selection.gpu || d.gpu === selection.gpu)
      .map((d) => {
        const items = new Map();
        d.shapes.forEach((shape) => {
          const agrees = Object.entries(shape.db).every(
            ([dim, v]) =>
              !(dim in selection) ||
              level.dims.includes(dim) ||
              String(v) === selection[dim],
          );
          const tuple = level.dims.map((dim) => shape.db[dim]);
          const key = tupleKey(tuple);
          if (!agrees || !offered.has(key)) return;
          const item = items.get(key) ?? { key, tuple, ops: [], paths: [] };
          const op = shape.layer.split(".").at(-1);
          if (!item.ops.includes(op)) item.ops.push(op);
          if (!item.paths.includes(shape.layer)) item.paths.push(shape.layer);
          items.set(key, item);
        });
        return {
          model: d.model?.name ?? d.model_config,
          family: d.model?.family,
          order: order(d),
          deployment: d.label,
          items: [...items.values()],
        };
      })
      .filter((g) => g.items.length)
      // Deployments of one model that ask for the same shapes read as one group.
      .reduce((merged, g) => {
        const same = merged.find(
          (m) =>
            m.model === g.model &&
            m.items.map((i) => i.key).join() === g.items.map((i) => i.key).join(),
        );
        if (same) same.deployments.push(g.deployment);
        else merged.push({ ...g, deployments: [g.deployment] });
        return merged;
      }, [])
      .sort(
        (a, b) =>
          a.order - b.order ||
          a.deployments[0].localeCompare(b.deployments[0], undefined, {
            numeric: true,
          }),
      )
  );
}

/* Shapes are several numbers at once, so a chip reads "1,024 × 6,144" under
   a name that says which numbers they are: "Shape (n × k)". They are grouped
   by the model deployment that asks for them, each chip naming its layer, and
   the shapes no listed model asks for fold away under "Other". */
function ShapeList({ level, tuples, current, groups, single, compare, pick }) {
  const labelId = "level-shape";
  const comparable = tuples.filter((tuple) => !single(tuple)).length;
  const claimed = new Set(groups.flatMap((g) => g.items.map((i) => i.key)));
  const others = tuples.filter((tuple) => !claimed.has(tupleKey(tuple)));
  // "Other" opens by itself while it holds the pick, and stays open once a
  // shape in it has been picked, so the list does not fold under the pointer.
  const pickInOthers = others.some((t) => tupleKey(t) === current);
  const [open, setOpen] = useState(pickInOthers);
  const showOthers = open || !groups.length || pickInOthers;
  const dims = (tuple) =>
    tuple.map((v, i) => formatValue(v, level.units[i])).join(" × ");
  const chip = (tuple, ops, paths, inOthers) => {
    const key = tupleKey(tuple);
    return (
      <ToggleTag
        key={key}
        type="choice"
        value={key}
        pressed={key === current}
        faint={single(tuple)}
        title={[
          level.dims
            .map((d, i) => `${d} ${formatValue(tuple[i], level.units[i])}`)
            .join(", "),
          paths?.join("\n"),
          single(tuple) && `measured for one ${compare.label} only`,
        ]
          .filter(Boolean)
          .join("\n")}
        onClick={() => {
          if (inOthers) setOpen(true);
          pick(tuple);
        }}
      >
        {ops && <span className={s.op}>{ops.join(", ")}</span>}
        {dims(tuple)}
      </ToggleTag>
    );
  };
  return (
    <div
      className={`${s.level} ${s.shapeLevel}`}
      role="group"
      aria-labelledby={labelId}
    >
      <span id={labelId} className={s.levelName}>
        Shape ({level.dims.map(shortName).join(" × ")})
        <span className={s.levelCount}>
          {tuples.length} measured
          {groups.length > 0 && `, ${claimed.size} used by the models below`}
          {comparable < tuples.length &&
            `, ${comparable} with more than one ${compare.label}`}
        </span>
      </span>
      {groups.map((g) => (
        <section
          key={`${g.model}|${g.deployments.join()}`}
          className={s.model}
          style={{ "--family": tagColor("family", g.family) }}
          aria-label={`${g.model}, ${g.deployments.join("; ")}`}
        >
          <p className={s.modelName}>
            {g.model}
            <span>{g.deployments.join("; ")}</span>
          </p>
          <div className={s.choices}>
            {g.items.map((item) => chip(item.tuple, item.ops, item.paths))}
          </div>
        </section>
      ))}
      {others.length > 0 && groups.length > 0 && (
        <button
          type="button"
          className={s.othersToggle}
          aria-expanded={showOthers}
          onClick={() => setOpen(!showOthers)}
        >
          Other measured shapes
          <span className={s.levelCount}>{others.length}</span>
        </button>
      )}
      {others.length > 0 && showOthers && (
        <div className={s.choices}>
          {others.map((tuple) => chip(tuple, null, null, true))}
        </div>
      )}
    </div>
  );
}
