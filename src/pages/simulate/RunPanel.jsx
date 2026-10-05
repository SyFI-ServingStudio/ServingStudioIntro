import { LoaderCircle, Play, Square, TriangleAlert } from "lucide-react";
import { useEffect, useState } from "react";
import { ReadMoreButton, ResultOverlay } from "../models/ReadMore";
import { Pending } from "../models/TreeParts";
import models from "../models/Models.module.css";
import { requestsText, runName, runParams } from "./runs";
import { count, duration, latency, rate } from "./simulateData";
import s from "./Simulate.module.css";

/* The run button and the run it shows: the one just started, or one picked
   from the list. Its record says what ran, so it never reads as the setup
   beside it once that changes. */
export function RunPanel({
  request,
  starting,
  startError,
  onRun,
  record,
  id,
  presets,
  onStop,
}) {
  return (
    <section className={s.panel} aria-labelledby="run-title">
      <h2 id="run-title" className={s.panelTitle}>
        Run
      </h2>
      <button
        type="button"
        className={`button button-primary ${s.run}`}
        onClick={onRun}
        disabled={!request.body || starting}
        aria-busy={starting || undefined}
      >
        {starting ? (
          <LoaderCircle size={18} className={models.spin} aria-hidden="true" />
        ) : (
          <Play size={18} aria-hidden="true" />
        )}
        Run simulation
      </button>
      {request.blocked ? (
        <p className={s.hint}>{request.blocked}</p>
      ) : startError ? (
        <p className={s.failure} role="alert">
          {startError}
        </p>
      ) : (
        <p className={s.hint}>
          Most runs take seconds; results are kept for a day.
        </p>
      )}
      <div className={s.result} aria-live="polite">
        {!id ? (
          <p className={s.empty}>
            Your first run's TTFT, TPOT, end-to-end latency and throughput appear
            here.
          </p>
        ) : !record ? (
          <p className={s.hint} role="status">
            <Pending className={models.pendingStat} /> Reading the run…
          </p>
        ) : (
          <RunView record={record} presets={presets} onStop={onStop} />
        )}
      </div>
    </section>
  );
}

const METRICS = [
  ["ttft_ms", "TTFT"],
  ["tpot_ms", "TPOT"],
  ["e2e_ms", "End to end"],
];

function RunView({ record, presets, onStop }) {
  const [reading, setReading] = useState(false);
  const name = runName(record, presets);
  const summary = record.summary;
  return (
    <div className={s.view}>
      <div className={s.what}>
        <p className={s.whatName}>{name}</p>
        <p>{runParams(record)}</p>
        <p>{requestsText(record)}</p>
      </div>

      {record.status === "queued" || record.status === "running" ? (
        <Progress record={record} onStop={() => onStop(record.simulation_id)} />
      ) : record.status !== "done" ? (
        <div className={models.predictStatus} data-tone="warn" role="alert">
          <TriangleAlert size={18} aria-hidden="true" />
          <div>
            <p className={models.predictStatusTitle}>
              {record.status === "timed_out"
                ? "The run passed the time limit and was stopped"
                : record.status === "cancelled"
                  ? "The run was stopped"
                  : "The run failed"}
            </p>
            {record.error && <p className={models.failureText}>{record.error}</p>}
          </div>
        </div>
      ) : !summary ? (
        <p className={s.failure} role="alert">
          The run finished but left no summary.
        </p>
      ) : (
        <>
          <p className={s.finished}>
            {count(summary.requests.finished)} of {count(summary.requests.total)}{" "}
            requests finished in {duration(summary.sim_ms)} of simulated time
            {summary.cause && summary.cause !== "DrainComplete" && (
              <> (stopped by {summary.cause})</>
            )}
            .
          </p>
          <dl className={s.metrics}>
            {METRICS.map(([key, label]) => (
              <Metric key={key} label={label} values={summary[key]} />
            ))}
          </dl>
          <dl className={s.throughput}>
            <div>
              <dt>Throughput per GPU</dt>
              <dd>
                {count(summary.throughput.total_tok_s_per_gpu)} <span>tok/s</span>
              </dd>
            </div>
            <div>
              <dt>Requests finished</dt>
              <dd>
                {rate(summary.throughput.completed_req_s)} <span>per s</span>
              </dd>
            </div>
          </dl>
          {record.workload.shortened > 0 && (
            <p className={s.hint}>
              {count(record.workload.shortened)}{" "}
              {record.workload.shortened === 1 ? "request was" : "requests were"}{" "}
              shortened by a few tokens to leave room for the last draft, as the
              server would draft fewer there.
            </p>
          )}
          <p className={s.hint}>
            {count(summary.throughput.prefill_tok_s)} prefill and{" "}
            {count(summary.throughput.decode_tok_s)} decode tok/s over{" "}
            {count(record.gpus)} {record.gpus === 1 ? "GPU" : "GPUs"}.{" "}
            {sentence(record.routing.label)}.
          </p>
          {record.run_id && (
            <div className={models.presets}>
              <ReadMoreButton onClick={() => setReading(true)} />
            </div>
          )}
          {reading && (
            <ResultOverlay
              kind="run"
              id={record.run_id}
              name={`${name}, ${runParams(record)}`}
              onClose={() => setReading(false)}
            />
          )}
        </>
      )}
    </div>
  );
}

const sentence = (text) => text.charAt(0).toUpperCase() + text.slice(1);

/* One latency: its median large, its tail beside it. */
function Metric({ label, values }) {
  const [p50, p90, p99] = ["p50", "p90", "p99"].map((q) => latency(values?.[q]));
  return (
    <div>
      <dt>{label}</dt>
      <dd>
        <strong>
          {p50.value} <span>{p50.unit}</span>
        </strong>
        <small>median</small>
      </dd>
      <dd className={s.tail}>
        <span>
          p90 {p90.value} {p90.unit}
        </span>
        <span>
          p99 {p99.value} {p99.unit}
        </span>
      </dd>
    </div>
  );
}

/* A run on its way: its place in the queue, or how long it has run. */
function Progress({ record, onStop }) {
  const [now, setNow] = useState(Date.now);
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, []);
  const started = record.started_at ? Date.parse(record.started_at) : null;
  const elapsed = started ? Math.max(0, Math.round((now - started) / 1000)) : 0;
  return (
    <div className={s.progress} role="status">
      <LoaderCircle size={18} className={models.spin} aria-hidden="true" />
      <p>
        {record.status === "queued"
          ? record.queue_position
            ? `Waiting, ${record.queue_position} ${record.queue_position === 1 ? "run" : "runs"} ahead`
            : "Waiting to start"
          : `Running, ${elapsed} s`}
      </p>
      <button type="button" className={models.addButton} onClick={onStop}>
        <Square size={14} aria-hidden="true" />
        Stop
      </button>
    </div>
  );
}
