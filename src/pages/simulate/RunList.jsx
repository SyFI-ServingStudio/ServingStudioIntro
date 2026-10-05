import { Pin, X } from "lucide-react";
import { useState } from "react";
import { Pending } from "../models/TreeParts";
import models from "../models/Models.module.css";
import { requestsText, runName, runParams } from "./runs";
import { count, latencyText } from "./simulateData";
import s from "./Simulate.module.css";

const STATUS = {
  queued: "Waiting",
  running: "Running",
  failed: "Failed",
  timed_out: "Timed out",
};

/* The numbers a reader compares runs by, each with how to print it. */
const COLUMNS = [
  ["TTFT p50", (m) => m.ttft_ms?.p50, latencyText],
  ["TTFT p99", (m) => m.ttft_ms?.p99, latencyText],
  ["TPOT p50", (m) => m.tpot_ms?.p50, latencyText],
  ["TPOT p99", (m) => m.tpot_ms?.p99, latencyText],
  [
    "Throughput per GPU",
    (m) => m.throughput.total_tok_s_per_gpu,
    (v) => `${count(v)} tok/s`,
  ],
];

/* The runs this browser started, side by side. Pinning one makes it the
   baseline: every other finished run shows how far each number is from it.
   Picking a run shows it in the panel. */
export function RunList({ ids, records, selected, presets, onSelect, onRemove }) {
  const [pinned, setPinned] = useState(null);
  if (!ids.length) return null;
  const done = (id) =>
    records[id]?.status === "done" ? records[id].summary : null;
  const base = ids.includes(pinned) ? done(pinned) : null;
  return (
    <section className={`wrap ${s.runs}`} aria-labelledby="runs-title">
      <h2 id="runs-title">
        Your runs <span>{ids.length}</span>
      </h2>
      <p className={s.hint}>
        {base
          ? "Each run's difference from the pinned one is under its number."
          : "Pin a finished run to see how far the others are from it."}
      </p>
      <div className={s.tableScroll}>
        <table className={s.table}>
          <thead>
            <tr>
              <th scope="col">Run</th>
              <th scope="col">Finished</th>
              {COLUMNS.map(([label]) => (
                <th key={label} scope="col">
                  {label}
                </th>
              ))}
              <th scope="col">
                <span className={s.visuallyHidden}>Pin or remove</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {ids.map((id) => {
              const record = records[id];
              const summary = done(id);
              const isBase = id === pinned && base;
              return (
                <tr
                  key={id}
                  aria-current={id === selected || undefined}
                  data-pinned={isBase || undefined}
                >
                  <th scope="row">
                    <button type="button" onClick={() => onSelect(id)}>
                      {record?.preset ? (
                        <>
                          <span>{runName(record, presets)}</span>
                          <small>{runParams(record)}</small>
                          <small>{requestsText(record)}</small>
                        </>
                      ) : (
                        <Pending className={models.pendingStat} />
                      )}
                    </button>
                  </th>
                  {summary ? (
                    <>
                      <td>
                        {count(summary.requests.finished)} /{" "}
                        {count(summary.requests.total)}
                      </td>
                      {COLUMNS.map(([label, read, show]) => (
                        <td key={label}>
                          {show(read(summary))}
                          {base && !isBase && (
                            <small className={s.delta}>
                              {difference(read(summary), read(base))}
                            </small>
                          )}
                        </td>
                      ))}
                    </>
                  ) : (
                    <td colSpan={COLUMNS.length + 1} className={s.status}>
                      {record ? (STATUS[record.status] ?? record.status) : ""}
                    </td>
                  )}
                  <td className={s.rowActions}>
                    {summary && (
                      <button
                        type="button"
                        className={models.removeButton}
                        aria-pressed={id === pinned}
                        aria-label={
                          id === pinned
                            ? "Unpin this run"
                            : "Compare the others to this run"
                        }
                        title={
                          id === pinned
                            ? "Unpin this run"
                            : "Compare the others to this run"
                        }
                        onClick={() => setPinned(id === pinned ? null : id)}
                      >
                        <Pin size={16} aria-hidden="true" />
                      </button>
                    )}
                    <button
                      type="button"
                      className={models.removeButton}
                      aria-label="Remove this run"
                      title="Remove this run"
                      onClick={() => {
                        if (id === pinned) setPinned(null);
                        onRemove(id);
                      }}
                    >
                      <X size={16} aria-hidden="true" />
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

/* How far a number is from the baseline's, in percent. */
function difference(value, base) {
  if (value == null || base == null || base === 0) return "";
  const change = ((value - base) / base) * 100;
  if (Math.abs(change) < 0.5) return "same";
  const text = Math.abs(change) >= 10 ? Math.round(change) : change.toFixed(1);
  return `${change > 0 ? "+" : "−"}${String(text).replace("-", "")}%`;
}
