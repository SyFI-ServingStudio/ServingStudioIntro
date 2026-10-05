import { useCallback, useEffect, useRef, useState } from "react";
import { paramsText } from "../models/modelData";
import {
  deleteSimulation,
  readSimulation,
  saveRuns,
  savedRuns,
  unfinished,
  count,
  rate,
} from "./simulateData";

const POLL_MS = 1500;

/* The runs this browser started and each one's record, as GET
   /simulations/{id} gives it. A run still queued or running is read again
   every POLL_MS until it finishes; one the service no longer has (kept a
   day) is dropped. `extra` is a run a link names, shown but not kept. */
export function useRuns(extra) {
  const [ids, setIds] = useState(savedRuns);
  const [records, setRecords] = useState({});
  const recordsRef = useRef(records);
  recordsRef.current = records;

  const shown = extra && !ids.includes(extra) ? [extra, ...ids] : ids;
  const key = shown.join(",");

  useEffect(() => {
    let live = true;
    let timer = null;
    const controller = new AbortController();
    const read = async () => {
      const due = key
        .split(",")
        .filter(Boolean)
        .filter((id) => {
          const record = recordsRef.current[id];
          return !record || (!record.gone && unfinished(record.status));
        });
      await Promise.all(
        due.map((id) =>
          readSimulation(id, controller.signal).then(
            (record) => live && setRecords((all) => ({ ...all, [id]: record })),
            (error) => {
              if (!live || error.name === "AbortError") return;
              // 404: expired or never existed. Anything else is read again.
              if (error.status === 404)
                setRecords((all) => ({ ...all, [id]: { gone: true } }));
            },
          ),
        ),
      );
      if (live) timer = setTimeout(read, POLL_MS);
    };
    read();
    return () => {
      live = false;
      controller.abort();
      clearTimeout(timer);
    };
  }, [key]);

  // A saved run the service no longer has leaves the list.
  useEffect(() => {
    const kept = ids.filter((id) => !records[id]?.gone);
    if (kept.length !== ids.length) {
      setIds(kept);
      saveRuns(kept);
    }
  }, [ids, records]);

  const add = useCallback((id, record) => {
    setRecords((all) => ({ ...all, [id]: record }));
    setIds((list) => {
      const next = [id, ...list.filter((other) => other !== id)];
      saveRuns(next);
      return next;
    });
  }, []);

  /* Forget a run here. One this browser started the service drops too (a
     run still on its way stops, so stopping one removes it); one a link
     named is someone's to keep, so it is only hidden. */
  const remove = useCallback((id) => {
    if (savedRuns().includes(id)) deleteSimulation(id).catch(() => {});
    setIds((list) => {
      const next = list.filter((other) => other !== id);
      saveRuns(next);
      return next;
    });
  }, []);

  return {
    ids: shown.filter((id) => !records[id]?.gone),
    // The runs this browser started, which it may stop.
    own: ids,
    records,
    add,
    remove,
  };
}

/* What a run ran, in a reader's words: the model and its deployment;
   `runParams` gives the axis values. */
export function runName(record, presets) {
  const preset = presets.get(record.preset);
  return preset ? `${preset.checkpointName} · ${preset.name}` : record.preset;
}

export const runParams = (record) => paramsText(record.params);

/* The traffic a run served and how hard it came, from its record. */
export function requestsText(record) {
  const w = record.workload;
  const source =
    w.source === "capture"
      ? w.capture
      : w.source === "generated"
        ? "your shape"
        : "your file";
  const n = w.trace?.requests;
  const own = w.source === "capture" ? "as recorded" : "at its own times";
  const per = w.trace?.sessions ? "conversations" : "requests";
  const load =
    w.load?.concurrency != null
      ? `${count(w.load.concurrency)} in flight`
      : w.load?.rate != null
        ? `${rate(w.load.rate)} ${per}/s`
        : w.trace?.rate != null
          ? `${rate(w.trace.rate)} ${per}/s, ${own}`
          : own;
  return `${n == null ? "" : `${count(n)} `}${per} from ${source}, ${load}`;
}
