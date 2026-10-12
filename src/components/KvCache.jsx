import { useLayoutEffect, useRef, useState } from "react";
import {
  bytesText,
  capacity,
  concurrencyText,
  engineText,
  layoutAt,
  layoutRows,
  layoutTotals,
  lengthStops,
  optionsText,
  spreadLabels,
  requestRows,
} from "./kvCache";
import s from "./KvCache.module.css";

const GROWTH_NOTE = {
  per_token: "grows with the context",
  sliding_window: "keeps a window",
  per_request: "one per request",
};

/* The arch's cache on its own, per GPU: what a token of context, the
   windows and a request's state cost, then one table of the parts the arch
   declares and what each holds at a context the reader slides to. The
   Models page shows it; there is no worker yet, so no budget or
   concurrency. */
export function KvLayout({ kv, partitions, maxModelLen }) {
  const totals = layoutTotals(kv);
  const shards = Math.max(1, kv.attention_shards);
  const stops = lengthStops(maxModelLen ?? DEFAULT_LONGEST);
  const [picked, setPicked] = useState(null);
  // A length picked on another member snaps to this one's nearest stop.
  const index =
    picked == null
      ? stops.length - 1
      : stops.reduce(
          (best, stop, at) =>
            Math.abs(stop - picked) < Math.abs(stops[best] - picked) ? at : best,
          0,
        );
  const tokens = stops[index];
  const held = layoutAt(kv, tokens);
  const engine = engineText(kv);
  return (
    <section className={s.card} aria-labelledby="kv-layout-title">
      <h2 id="kv-layout-title">KV cache</h2>
      <p className={s.lead}>
        What one request keeps on each GPU between steps.
        {shards > 1 && (
          <> A KV partition spans {shards} GPUs; the bytes are each GPU's share.</>
        )}
        {partitions > 1 && (
          <> A replica holds {partitions} partitions, each with its own cache.</>
        )}
      </p>
      <dl className={s.stats}>
        {totals.token != null && (
          <div>
            <dt>Per token of context</dt>
            <dd>{bytesText(totals.token)}</dd>
          </div>
        )}
        {totals.window != null && (
          <div>
            <dt>Sliding windows, when full</dt>
            <dd>{bytesText(totals.window)}</dd>
          </div>
        )}
        {totals.request != null && (
          <div>
            <dt>State per request</dt>
            <dd>{bytesText(totals.request)}</dd>
          </div>
        )}
      </dl>

      <div className={s.picker}>
        <label className={s.length}>
          <span className={s.pickerLabel}>Context length</span>
          <output className={s.pickerValue}>
            {tokens.toLocaleString("en-US")}
            <span>
              {" "}
              tokens
              {maxModelLen != null &&
                tokens === maxModelLen &&
                ", the longest it serves"}
            </span>
          </output>
          <input
            type="range"
            min={0}
            max={stops.length - 1}
            step={1}
            value={index}
            onChange={(event) => setPicked(stops[Number(event.target.value)])}
            aria-valuetext={`${tokens.toLocaleString("en-US")} tokens`}
          />
          <span className={s.ends} aria-hidden="true">
            <span>{stops[0].toLocaleString("en-US")}</span>
            <span>{stops[stops.length - 1].toLocaleString("en-US")}</span>
          </span>
        </label>
        <div className={s.total}>
          <span className={s.pickerLabel}>One decoding request holds</span>
          <span className={s.pickerValue}>
            {bytesText(held.total)}
            <span> per GPU</span>
          </span>
        </div>
        {held.rows.length > 1 && <ShareBar rows={held.rows} />}
      </div>

      <PartsAtLength kv={kv} held={held} tokens={tokens} />
      <p className={s.note}>
        {engine && <>{engine} </>}A prefill chunk adds its tokens to each window,
        and prefix caching keeps one more copy of a reusable state. Those, and how
        many requests fit, depend on the server's settings and the memory it leaves
        for this cache; the Simulate page shows them for each deployment.
      </p>
    </section>
  );
}

/* The layout's parts: what each holds at the picked length and its share,
   which the slider moves and a narrow screen keeps in view, then what the
   part is kept for and costs. */
function PartsAtLength({ kv, held, tokens }) {
  const rows = layoutRows(kv);
  const parts = rows.length > 1;
  return (
    <div className={s.scroll}>
      <table className={`${s.table} ${s.wide}`}>
        <thead>
          <tr>
            <th scope="col">Part</th>
            <th scope="col">At {tokens.toLocaleString("en-US")} tokens</th>
            <th scope="col" className={s.num}>
              Per GPU
            </th>
            {parts && (
              <th scope="col" className={s.num}>
                Share
              </th>
            )}
            <th scope="col" className={s.layout}>
              Kept for
            </th>
            <th scope="col">Size per GPU</th>
            <th scope="col">Prefix reuse</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={row.name}>
              <th scope="row">
                {parts && <i data-tone={index % TONES} aria-hidden="true" />}
                <code>{row.name}</code>
              </th>
              <td>{held.rows[index].holds}</td>
              <td className={`${s.num} ${s.strong}`}>
                {bytesText(held.rows[index].bytes)}
              </td>
              {parts && (
                <td className={s.num}>{sharePercent(held.rows[index].share)}</td>
              )}
              <td className={s.layout} title={GROWTH_NOTE[row.kind]}>
                {row.grows}
              </td>
              <td>
                {row.bytes}
                {row.overhead && <small>vLLM: {row.overhead}</small>}
              </td>
              <td>{row.reusable ? "Yes" : "No"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* The parts' shares as one bar, each named by its segment's middle, above
   and below by turns. Labels on one side that would still meet are spread
   apart once their widths are known; a leader ties each to its segment. */
function ShareBar({ rows }) {
  const widths = rows.map((row) => Math.max(row.share, 0.004));
  const sum = widths.reduce((total, width) => total + width, 0);
  let start = 0;
  const segments = rows.map((row, index) => {
    const width = widths[index] / sum;
    const segment = { row, index, width, middle: start + width / 2 };
    start += width;
    return segment;
  });
  const side = (index) => (index % 2 ? "below" : "above");

  const box = useRef(null);
  const labels = useRef([]);
  const [lefts, setLefts] = useState(null);
  const shares = segments.map(({ row }) => sharePercent(row.share)).join();
  useLayoutEffect(() => {
    const place = () => {
      const span = box.current.clientWidth;
      const placed = [];
      for (const which of ["above", "below"]) {
        const row = segments.filter(({ index }) => side(index) === which);
        const spread = spreadLabels(
          row.map(({ middle }) => middle * span),
          row.map(({ index }) => labels.current[index].offsetWidth),
          span,
        );
        row.forEach(({ index }, at) => (placed[index] = spread[at]));
      }
      setLefts(placed);
    };
    place();
    const observer = new ResizeObserver(place);
    observer.observe(box.current);
    return () => observer.disconnect();
    // Labels change width only with the shares they print.
  }, [shares, rows.length]);

  return (
    <div className={s.shares} ref={box} aria-hidden="true">
      {segments.map(({ row, index, middle }) => (
        <span
          key={row.name}
          ref={(node) => (labels.current[index] = node)}
          className={s.shareLabel}
          data-side={side(index)}
          style={
            lefts?.[index] != null
              ? { left: `${lefts[index]}px` }
              : {
                  left: `${middle * 100}%`,
                  transform: `translateX(-${middle * 100}%)`,
                }
          }
        >
          <code>{row.name}</code> {sharePercent(row.share)}
        </span>
      ))}
      {segments.map(({ row, index, middle }) => (
        <i
          key={row.name}
          className={s.leader}
          data-side={side(index)}
          style={{ left: `${middle * 100}%` }}
        />
      ))}
      <div className={s.bar}>
        {segments.map(({ row, index, width }) => (
          <span
            key={row.name}
            data-tone={index % TONES}
            style={{ width: `${width * 100}%` }}
            title={`${row.name}: ${bytesText(row.bytes)}`}
          />
        ))}
      </div>
    </div>
  );
}

/* A simulation member's pool: how many requests of the longest context fit,
   what one of them holds part by part, and, folded, the layout behind it. */
export function KvCapacity({ kv, role, gpu }) {
  const facts = capacity(kv);
  if (!facts) return null;
  const rows = requestRows(kv);
  const otherLabel =
    facts.otherMode === "none" ? "Without vLLM's overhead" : "With vLLM's overhead";
  return (
    <div className={s.pool}>
      {role && (
        <h3>
          {role === "stage"
            ? "Each pipeline stage"
            : `${role[0].toUpperCase()}${role.slice(1)} pool`}
        </h3>
      )}
      <dl className={s.stats}>
        <div>
          <dt>
            Requests of {facts.maxModelLen.toLocaleString("en-US")} tokens at once
          </dt>
          <dd data-short={facts.concurrency < 1 || undefined}>
            {concurrencyText(facts.concurrency)}
            {facts.partitions > 1 && (
              <span> per partition, ×{facts.partitions} per replica</span>
            )}
          </dd>
        </div>
        {facts.other != null && !alike(facts.other, facts.concurrency) && (
          <div>
            <dt>{otherLabel}</dt>
            <dd className={s.secondary}>{concurrencyText(facts.other)}</dd>
          </div>
        )}
        <div>
          <dt>KV memory per GPU</dt>
          <dd className={s.secondary}>{bytesText(facts.budget)}</dd>
        </div>
        {facts.tokens != null && (
          <div>
            <dt>Tokens it holds</dt>
            <dd className={s.secondary}>{facts.tokens.toLocaleString("en-US")}</dd>
          </div>
        )}
      </dl>
      {facts.concurrency < 1 && (
        <p className={s.warn}>
          One request of the full {facts.maxModelLen.toLocaleString("en-US")}-token
          context needs {bytesText(facts.perRequest)} per GPU, more than this cache
          holds; vLLM would refuse to start with it. Shorter requests still run.
        </p>
      )}

      <RequestParts rows={rows}>
        One request of {facts.maxModelLen.toLocaleString("en-US")} tokens holds{" "}
        <b>{bytesText(facts.perRequest)}</b> per GPU
        {gpu && <> on {gpu}</>}
      </RequestParts>

      <p className={s.note}>
        {optionsText(kv)} KV memory per GPU is the budget this deployment gives the
        cache, after the weights and activations.
      </p>
      <details className={s.details}>
        <summary>The model's cache layout</summary>
        <LayoutTable kv={kv} />
      </details>
    </div>
  );
}

const TONES = 5;

/* One request part by part: a bar of shares over the table that names them.
   A one-part cache needs no shares. */
function RequestParts({ rows, children }) {
  const parts = rows.length > 1;
  return (
    <figure className={s.request}>
      <figcaption>{children}</figcaption>
      {parts && <ShareBar rows={rows} />}
      <table className={s.table}>
        <thead>
          <tr>
            <th scope="col">Part</th>
            <th scope="col">Holds</th>
            <th scope="col" className={s.num}>
              Per GPU
            </th>
            {parts && (
              <th scope="col" className={s.num}>
                Share
              </th>
            )}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={row.name}>
              <th scope="row">
                {parts && <i data-tone={index % TONES} aria-hidden="true" />}
                <code>{row.name}</code>
              </th>
              <td>{row.holds}</td>
              <td className={s.num}>{bytesText(row.bytes)}</td>
              {parts && <td className={s.num}>{sharePercent(row.share)}</td>}
            </tr>
          ))}
        </tbody>
      </table>
    </figure>
  );
}

// A share to a whole percent; a sliver shows as under one.
const sharePercent = (share) =>
  share > 0 && share < 0.005 ? "<1%" : `${Math.round(share * 100)}%`;

// The slider's end when the member names no longest context.
const DEFAULT_LONGEST = 131072;

// Two concurrencies that print alike: vLLM's overhead changes nothing here.
const alike = (a, b) => Math.abs(a - b) <= 0.005 * Math.max(a, b);

function LayoutTable({ kv }) {
  const rows = layoutRows(kv);
  const engine = engineText(kv);
  const overhead = rows.some((row) => row.overhead);
  return (
    <>
      <table className={s.table}>
        <thead>
          <tr>
            <th scope="col">Part</th>
            <th scope="col">Kept for</th>
            <th scope="col" className={s.num}>
              Bytes per GPU
            </th>
            {overhead && <th scope="col">vLLM adds</th>}
            <th scope="col">Prefix reuse</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.name}>
              <th scope="row">
                <code>{row.name}</code>
              </th>
              <td title={GROWTH_NOTE[row.kind]}>{row.grows}</td>
              <td className={s.num}>{row.bytes}</td>
              {overhead && <td>{row.overhead || "-"}</td>}
              <td>{row.reusable ? "Yes" : "No"}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {engine && <p className={s.note}>{engine}</p>}
    </>
  );
}
