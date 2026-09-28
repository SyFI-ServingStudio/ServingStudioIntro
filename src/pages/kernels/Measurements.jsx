import { ChevronDown, ChevronUp, Download, X } from "lucide-react";
import { useMemo, useState } from "react";
import {
  argNames,
  argUnit,
  downloadText,
  formatBytes,
  formatDate,
  formatNumber,
  isDtype,
  isList,
  isNumeric,
  metricDoc,
  metricNames,
  shortGpu,
  toCsv,
} from "./kernelData";
import { ToggleTag } from "./Tag";
import s from "./Measurements.module.css";

/* Every profile.db row of one kind as a table to sort and filter, whether or
   not a supported deployment reads it.

   State lives in the query string like the rest of the page, under "m.":
   m.sort / m.dir (column and direction), m.f.<column> (picked values,
   comma-separated), m.min.<column> / m.max.<column> (a numeric range),
   m.q.<column> (text the value contains), m.page and m.prov (provenance
   columns shown). */

const PAGE_SIZE = 100;
// A column with at most this many values filters by picking them.
const CHOICES_MAX = 16;
const PROVENANCE = [
  ["profiler_run_at", "Measured"],
  ["profiler_git_hash", "Profiler commit"],
  ["cuda_version", "CUDA"],
  ["driver_version", "Driver"],
  ["backend_version", "Library version"],
];

/* A list argument arrives as JSON text, sometimes hundreds of entries long
   (one per expert, request or query row). A cell reads its shape instead:
   entries, how many are non-zero, the largest and the total. A list of
   tuples totals each position. */
const listCache = new Map();
function listStats(text) {
  if (listCache.has(text)) return listCache.get(text);
  let items = null;
  try {
    items = JSON.parse(text);
  } catch {
    items = null;
  }
  let stats = null;
  if (Array.isArray(items)) {
    if (items.every((v) => typeof v === "number")) {
      stats = {
        length: items.length,
        nonzero: items.filter((v) => v !== 0).length,
        max: items.length ? Math.max(...items) : 0,
        sum: items.reduce((a, b) => a + b, 0),
      };
    } else if (
      items.every((v) => Array.isArray(v) && v.every((x) => typeof x === "number"))
    ) {
      const width = Math.max(0, ...items.map((v) => v.length));
      const sums = Array.from({ length: width }, (_, i) =>
        items.reduce((a, v) => a + (v[i] ?? 0), 0),
      );
      stats = { length: items.length, sums, sum: sums[0] ?? 0 };
    } else stats = { length: items.length, sum: items.length };
  }
  listCache.set(text, stats);
  return stats;
}

const count = (n) => n.toLocaleString("en-US");
function listSummary(text, unit) {
  const stats = listStats(text);
  if (!stats) return String(text);
  const entries = `${count(stats.length)} ${stats.length === 1 ? "entry" : "entries"}`;
  if (stats.sums) return `${entries}, totals (${stats.sums.map(count).join(", ")})`;
  if (stats.nonzero == null) return entries;
  const of = unit ? ` ${unit}` : "";
  return `${entries}, ${count(stats.nonzero)} non-zero, max ${count(stats.max)}, sum ${count(stats.sum)}${of}`;
}

const byText = (a, b) =>
  String(a).localeCompare(String(b), undefined, { numeric: true });

/* The columns the table can show, in profile.db order: the row's key, the
   arguments, the metrics, then provenance. Each knows how it sorts, filters
   and reads. */
function buildColumns(kernel, records) {
  const argDoc = (name) => kernel.args.find((a) => a.name === name);
  const columns = [
    { key: "gpu", label: "GPU", kind: "label", format: shortGpu },
    { key: "backend", label: "Backend", kind: "label" },
    ...argNames(kernel).map((name) => {
      const unit = argUnit(kernel, name);
      const doc = argDoc(name)?.doc;
      const title = [doc, unit && `Unit: ${unit}`].filter(Boolean).join("\n");
      if (isList(kernel, name))
        return {
          key: name,
          label: name,
          title,
          kind: "list",
          sortValue: (v) => listStats(v)?.sum ?? 0,
          format: (v) => listSummary(v, unit),
        };
      if (isNumeric(kernel, name))
        return {
          key: name,
          label: name,
          title,
          kind: "number",
          format: (v) =>
            v == null
              ? "–"
              : unit === "bytes"
                ? formatBytes(v)
                : v.toLocaleString("en-US"),
        };
      return {
        key: name,
        label: name,
        title,
        kind: isDtype(kernel, name) ? "dtype" : "label",
      };
    }),
    ...metricNames(kernel)
      .filter((m) => records.some((r) => r[m] != null))
      .map((m) => {
        const { label, unit } = metricDoc(kernel, m);
        // Kernel times are mostly well under a millisecond: read them in µs
        // when the typical row is.
        if (unit === "ms") {
          const times = records
            .map((r) => r[m])
            .filter((v) => v != null)
            .sort((a, b) => a - b);
          const micro = times[Math.floor(times.length / 2)] < 1;
          return {
            key: m,
            label,
            unit: micro ? "µs" : "ms",
            kind: "metric",
            format: (v) => (v == null ? "–" : formatNumber(micro ? v * 1000 : v)),
          };
        }
        return {
          key: m,
          label,
          unit,
          kind: "metric",
          format: (v) => (v == null ? "–" : formatNumber(v)),
        };
      }),
    ...PROVENANCE.map(([key, label]) => ({
      key,
      label,
      kind: "provenance",
      get: (r) => r.provenance[key],
      format:
        key === "profiler_run_at"
          ? formatDate
          : key === "profiler_git_hash"
            ? (v) => (v ? v.slice(0, 12) : "–")
            : (v) => v ?? "not recorded",
    })),
  ];
  return columns.map((c) => ({
    get: (r) => r[c.key],
    format: (v) => (v == null ? "–" : String(v)),
    sortValue: (v) => v,
    ...c,
  }));
}

/* How each column filters: by picking values when it has few (and always for
   the GPU, backend and dtypes), by a range when it is a number with many,
   else by text the value contains. */
function filterKind(column, values) {
  if (column.kind === "list") return "text";
  if (["dtype"].includes(column.kind) || ["gpu", "backend"].includes(column.key))
    return "choice";
  if (values.length <= CHOICES_MAX && values.every((v) => !String(v).includes(",")))
    return "choice";
  if (column.kind === "number" || column.kind === "metric") return "range";
  return "text";
}

const sortValues = (values) =>
  [...values].sort((a, b) =>
    typeof a === "number" && typeof b === "number" ? a - b : byText(a, b),
  );

function readFilters(columns, query) {
  const filters = {};
  for (const c of columns) {
    const picked = query[`m.f.${c.key}`];
    const min = query[`m.min.${c.key}`];
    const max = query[`m.max.${c.key}`];
    const text = query[`m.q.${c.key}`];
    if (picked != null) filters[c.key] = { picked: new Set(picked.split(",")) };
    else if (min != null || max != null)
      filters[c.key] = {
        min: min != null && min !== "" ? Number(min) : null,
        max: max != null && max !== "" ? Number(max) : null,
      };
    else if (text) filters[c.key] = { text };
  }
  return filters;
}

function matches(column, filter, record) {
  const v = column.get(record);
  if (filter.picked) return filter.picked.has(String(v));
  if (filter.text != null)
    return String(v ?? "")
      .toLowerCase()
      .includes(filter.text.toLowerCase());
  // A metric's range is in the unit the column shows.
  const scale = column.unit === "µs" ? 1000 : 1;
  const x = v == null ? null : v * scale;
  if (x == null) return filter.min == null && filter.max == null;
  return (
    (filter.min == null || x >= filter.min) &&
    (filter.max == null || x <= filter.max)
  );
}

/* The API's rows route filters by equality only: the picked single values
   are the part of this table's filters a copied request can carry. */
export function apiFilters(kernel, query) {
  const keys = ["gpu", "backend", ...argNames(kernel)];
  return keys.flatMap((key) => {
    const picked = query[`m.f.${key}`];
    return picked != null && !picked.includes(",") ? [[key, picked]] : [];
  });
}

export function Measurements({ kernel, records, query, update }) {
  const columns = useMemo(() => buildColumns(kernel, records), [kernel, records]);
  const values = useMemo(
    () =>
      Object.fromEntries(
        columns.map((c) => [
          c.key,
          sortValues(
            new Set(records.map((r) => c.get(r)).filter((v) => v != null)),
          ),
        ]),
      ),
    [columns, records],
  );
  const filters = readFilters(columns, query);
  const showProvenance = query["m.prov"] === "1";
  const sortKey = columns.some((c) => c.key === query["m.sort"])
    ? query["m.sort"]
    : null;
  const descending = query["m.dir"] === "desc";

  const filterKey = JSON.stringify(
    Object.entries(query).filter(([k]) => /^m\.(f|min|max|q)\./.test(k)),
  );
  // Rows kept by every filter, and per value of each picked-from column, how
  // many rows the other filters keep: what picking it would show.
  const { kept, facets } = useMemo(() => {
    const active = columns.filter((c) => filters[c.key]);
    const facets = {};
    const failing = (r) => {
      let miss = null;
      for (const c of active) {
        if (!matches(c, filters[c.key], r)) {
          if (miss) return false;
          miss = c.key;
        }
      }
      return miss ?? true;
    };
    const kept = [];
    const choiceColumns = columns.filter(
      (c) => filterKind(c, values[c.key]) === "choice" && values[c.key].length > 1,
    );
    for (const c of choiceColumns) facets[c.key] = new Map();
    for (const r of records) {
      const miss = failing(r);
      if (miss === false) continue;
      if (miss === true) kept.push(r);
      for (const c of choiceColumns) {
        if (miss !== true && miss !== c.key) continue;
        const v = String(c.get(r));
        facets[c.key].set(v, (facets[c.key].get(v) ?? 0) + 1);
      }
    }
    return { kept, facets };
    // filterKey stands for `filters`, which is rebuilt on every render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [columns, records, values, filterKey]);
  const shown = useMemo(() => {
    if (!sortKey) return kept;
    const column = columns.find((c) => c.key === sortKey);
    const value = (r) => column.sortValue(column.get(r));
    const sign = descending ? -1 : 1;
    return [...kept].sort((a, b) => {
      const x = value(a);
      const y = value(b);
      if (x == null || y == null) return x == null ? (y == null ? 0 : 1) : -1;
      return (
        sign *
        (typeof x === "number" && typeof y === "number" ? x - y : byText(x, y))
      );
    });
  }, [columns, kept, sortKey, descending]);

  // Columns with one value in every row of the kind get no filter; columns
  // with one value in every row shown fold into a line above the table.
  const filterable = columns.filter(
    (c) => c.kind !== "provenance" && values[c.key].length > 1,
  );
  const constant = columns.filter((c) => {
    if (c.kind === "provenance" || c.kind === "metric") return false;
    const first = shown.length ? String(c.get(shown[0])) : null;
    return shown.every((r) => String(c.get(r)) === first);
  });
  const visible = columns.filter(
    (c) =>
      !constant.includes(c) &&
      (c.kind !== "provenance" || showProvenance || c.key === "profiler_run_at"),
  );

  const pages = Math.max(1, Math.ceil(shown.length / PAGE_SIZE));
  const page = Math.min(pages, Math.max(1, Number(query["m.page"]) || 1));
  const rows = shown.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);

  const set = (patch) => update({ "m.page": "", ...patch });
  const clearAll = () =>
    update(
      Object.fromEntries(
        Object.keys(query)
          .filter((k) => /^m\.(f|min|max|q)\./.test(k) || k === "m.page")
          .map((k) => [k, ""]),
      ),
    );
  const sortBy = (key) =>
    set(
      sortKey !== key
        ? { "m.sort": key, "m.dir": "" }
        : descending
          ? { "m.sort": "", "m.dir": "" }
          : { "m.sort": key, "m.dir": "desc" },
    );
  const active = Object.keys(filters).length;

  const csv = () => {
    const keys = ["gpu", "backend", ...argNames(kernel), ...metricNames(kernel)];
    downloadText(`${kernel.kind}-rows.csv`, toCsv(shown, keys), "text/csv");
  };

  return (
    <div className={s.measurements}>
      <section className={s.filters} aria-label="Filter rows">
        <div className={s.filtersHead}>
          <h2>Filter</h2>
          <p className={s.count} role="status">
            {count(shown.length)} of {count(records.length)} rows
          </p>
          {active > 0 && (
            <button type="button" className={s.clear} onClick={clearAll}>
              <X size={16} aria-hidden="true" />
              Clear {active === 1 ? "the filter" : `${active} filters`}
            </button>
          )}
        </div>
        <div className={s.filterGrid}>
          {filterable.map((c) => (
            <ColumnFilter
              key={c.key}
              column={c}
              values={values[c.key]}
              filter={filters[c.key]}
              kind={filterKind(c, values[c.key])}
              facet={facets[c.key]}
              set={set}
            />
          ))}
        </div>
      </section>

      <div className={s.tableHead}>
        {constant.length > 0 && shown.length > 0 ? (
          <p className={s.constant}>
            Every row shown has{" "}
            {constant.map((c, i) => (
              <span key={c.key}>
                {i > 0 && ", "}
                <code>{c.key}</code> {c.format(c.get(shown[0]))}
              </span>
            ))}
            .
          </p>
        ) : (
          <span />
        )}
        <div className={s.actions}>
          <label className={s.toggle}>
            <input
              type="checkbox"
              checked={showProvenance}
              onChange={(e) => update({ "m.prov": e.target.checked ? "1" : "" })}
            />
            Provenance columns
          </label>
          <button
            type="button"
            className={s.action}
            onClick={csv}
            disabled={!shown.length}
          >
            <Download size={18} aria-hidden="true" />
            Download {count(shown.length)} rows as CSV
          </button>
        </div>
      </div>

      {shown.length ? (
        <div
          className={s.scroll}
          tabIndex={0}
          role="region"
          aria-label={`Measured rows of ${kernel.kind}`}
        >
          <table className={s.table}>
            <thead>
              <tr>
                {visible.map((c) => (
                  <th
                    key={c.key}
                    scope="col"
                    className={
                      c.kind === "metric" || c.kind === "number" ? s.num : undefined
                    }
                    aria-sort={
                      sortKey === c.key
                        ? descending
                          ? "descending"
                          : "ascending"
                        : "none"
                    }
                  >
                    <button
                      type="button"
                      title={c.title || undefined}
                      onClick={() => sortBy(c.key)}
                    >
                      <span
                        className={
                          c.kind === "metric" ||
                          c.kind === "provenance" ||
                          ["gpu", "backend"].includes(c.key)
                            ? undefined
                            : s.argName
                        }
                      >
                        {c.label}
                      </span>
                      {c.unit && <span className={s.unit}>{c.unit}</span>}
                      {sortKey === c.key &&
                        (descending ? (
                          <ChevronDown size={16} aria-hidden="true" />
                        ) : (
                          <ChevronUp size={16} aria-hidden="true" />
                        ))}
                    </button>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={(page - 1) * PAGE_SIZE + i}>
                  {visible.map((c) => {
                    const v = c.get(r);
                    const text = c.format(v);
                    return (
                      <td
                        key={c.key}
                        className={
                          c.kind === "metric" || c.kind === "number"
                            ? s.num
                            : c.kind === "list"
                              ? s.list
                              : c.kind === "provenance"
                                ? s.provenance
                                : undefined
                        }
                        title={
                          c.kind === "list"
                            ? String(v)
                            : c.key === "profiler_run_at"
                              ? PROVENANCE.slice(1)
                                  .map(
                                    ([k, label]) =>
                                      `${label}: ${r.provenance[k] ?? "not recorded"}`,
                                  )
                                  .join("\n")
                              : undefined
                        }
                      >
                        {c.kind === "list" ? (
                          <ListCell text={v} summary={text} />
                        ) : (
                          text
                        )}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className={s.empty}>
          No row matches these filters.{" "}
          <button type="button" onClick={clearAll}>
            Clear them
          </button>
        </p>
      )}

      {pages > 1 && (
        <nav className={s.pager} aria-label="Pages of rows">
          <button
            type="button"
            disabled={page === 1}
            onClick={() =>
              update({ "m.page": page - 1 > 1 ? String(page - 1) : "" })
            }
          >
            Previous
          </button>
          <span>
            Rows {count((page - 1) * PAGE_SIZE + 1)}–
            {count(Math.min(page * PAGE_SIZE, shown.length))} of{" "}
            {count(shown.length)}
          </span>
          <button
            type="button"
            disabled={page === pages}
            onClick={() => update({ "m.page": String(page + 1) })}
          >
            Next
          </button>
        </nav>
      )}
    </div>
  );
}

/* A long list reads as its summary; the whole value opens under it. */
function ListCell({ text, summary }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button
        type="button"
        className={s.listToggle}
        aria-expanded={open}
        onClick={() => setOpen(!open)}
      >
        {summary}
      </button>
      {open && <code className={s.listValue}>{text}</code>}
    </>
  );
}

function ColumnFilter({ column, values, filter, kind, facet, set }) {
  const key = column.key;
  const name = (
    <span className={s.filterName} title={column.title || undefined}>
      {column.kind === "metric" ? column.label : <code>{column.label}</code>}
      {column.unit && <span className={s.unit}>{column.unit}</span>}
    </span>
  );
  if (kind === "choice") {
    const picked = filter?.picked;
    const toggle = (v) => {
      const next = new Set(picked ?? []);
      if (next.has(v)) next.delete(v);
      else next.add(v);
      set({ [`m.f.${key}`]: [...next].join(",") });
    };
    return (
      <div className={s.filter} role="group" aria-label={column.label}>
        {name}
        <div className={s.choices}>
          {values.map((v) => (
            <ToggleTag
              key={String(v)}
              type={
                key === "gpu"
                  ? "gpu"
                  : column.kind === "dtype"
                    ? "precision"
                    : "choice"
              }
              value={key === "gpu" ? shortGpu(v) : String(v)}
              pressed={picked ? picked.has(String(v)) : false}
              count={facet?.get(String(v)) ?? 0}
              disabled={false}
              faint={!facet?.get(String(v))}
              onClick={() => toggle(String(v))}
            >
              {column.format(v)}
            </ToggleTag>
          ))}
        </div>
      </div>
    );
  }
  if (kind === "range") {
    const scale = column.unit === "µs" ? 1000 : 1;
    const low = values[0] * scale;
    const high = values.at(-1) * scale;
    const input = (bound, placeholder) => (
      <input
        type="number"
        inputMode="decimal"
        className={s.number}
        aria-label={`${column.label} ${bound === "min" ? "at least" : "at most"}`}
        placeholder={formatNumber(placeholder)}
        value={filter?.[bound] ?? ""}
        onChange={(e) =>
          set({
            [`m.${bound}.${key}`]: e.target.value,
            [`m.${bound === "min" ? "max" : "min"}.${key}`]:
              filter?.[bound === "min" ? "max" : "min"] ?? "",
          })
        }
      />
    );
    return (
      <div className={s.filter} role="group" aria-label={column.label}>
        {name}
        <div className={s.range}>
          {input("min", low)}
          <span aria-hidden="true">to</span>
          {input("max", high)}
        </div>
      </div>
    );
  }
  return (
    <div className={s.filter}>
      <label htmlFor={`filter-${key}`}>{name}</label>
      <input
        id={`filter-${key}`}
        type="search"
        className={s.text}
        placeholder="contains…"
        value={filter?.text ?? ""}
        onChange={(e) => set({ [`m.q.${key}`]: e.target.value })}
      />
    </div>
  );
}
