import { useLayoutEffect, useRef, useState } from "react";
import {
  amplificationText,
  bytesText,
  capacity,
  concurrencyText,
  engineText,
  hasSizes,
  layoutRows,
  layoutSizes,
  lengthStops,
  optionsText,
  placementWords,
  spreadLabels,
  requestRows,
  timesText,
  vllmAddsText,
} from "./kvCache";
import s from "./KvCache.module.css";

const GROWTH_NOTE = {
  per_token: "grows with the context",
  sliding_window: "keeps a window",
  per_request: "one per request",
};

/* The arch's cache on its own, in the model's own bytes and in the three
   sizes the API gives: stored once (the least the model needs), on each GPU
   and over the KV partition's GPUs. First what a token, a full window and a
   request's state cost; then one decoding request at a context the reader
   slides to, part by part, with where each part sits on the GPUs. The
   Models page shows it; there is no worker yet, so vLLM's overhead, the
   budget and concurrency are the Simulate page's. */
export function KvLayout({ kv, partitions, maxModelLen }) {
  const gpus = Math.max(1, kv.attention_shards);
  const stops = lengthStops(maxModelLen ?? DEFAULT_LONGEST);
  const [picked, setPicked] = useState(null);
  if (!hasSizes(kv)) return null;
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
  const sizes = layoutSizes(kv, tokens);
  const adds = vllmAddsText(kv);
  return (
    <section className={s.card} aria-labelledby="kv-layout-title">
      <h2 id="kv-layout-title">KV cache</h2>
      <p className={s.lead}>
        What one request keeps between decoding steps, in the model's own bytes.
        {gpus > 1 ? (
          <>
            {" "}
            One cache spans {gpus} GPUs. Each part is split across them or kept
            whole on every one, so each size is given three ways: stored once,
            which is the least the model needs; on each GPU; and over all {gpus}.
          </>
        ) : (
          <> One GPU holds the whole cache.</>
        )}
        {partitions > 1 && (
          <> A replica holds {partitions} such caches, one per attention partition.</>
        )}
      </p>

      <SizeTable sizes={sizes} gpus={gpus} />

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
            {bytesText(sizes.total.gpu)}
            {gpus > 1 && <span> per GPU</span>}
          </span>
          {gpus > 1 && (
            <span className={s.totalSizes}>
              {bytesText(sizes.total.partition)} over {gpus} GPUs ·{" "}
              {bytesText(sizes.total.once)} stored once
            </span>
          )}
        </div>
        {sizes.rows.length > 1 && (
          <ShareBar rows={sizes.rows.map((row) => ({ ...row, bytes: row.gpu }))} />
        )}
      </div>

      <PartsAtLength sizes={sizes} gpus={gpus} tokens={tokens} />
      <p className={s.note}>
        {adds && <>{adds} </>}
        {adds ? "That, a" : "A"} prefill chunk in each window and prefix caching's
        extra copy of a reusable state are what the server adds on top; how many
        requests fit depends on the memory it leaves for this cache. The Simulate
        page charges all of it for each deployment.
      </p>
    </section>
  );
}

/* The three sizes of a figure as table cells, or the one size a one-GPU
   cache has. `amplified` is the partition's bytes over one copy. */
function SizeCells({ size, gpus, strong }) {
  const text = (bytes) => (bytes == null ? "-" : bytesText(bytes));
  if (gpus <= 1)
    return <td className={`${s.num} ${strong ? s.strong : ""}`}>{text(size?.gpu)}</td>;
  return (
    <>
      <td className={s.num}>{text(size?.once)}</td>
      <td className={`${s.num} ${strong ? s.strong : ""}`}>{text(size?.gpu)}</td>
      <td className={s.num}>{text(size?.partition)}</td>
      <td className={s.num}>
        {size && size.once > 0 ? timesText(size.partition / size.once) : "-"}
      </td>
    </>
  );
}

function SizeHeads({ gpus }) {
  if (gpus <= 1)
    return (
      <th scope="col" className={s.num}>
        Bytes
      </th>
    );
  return (
    <>
      <th scope="col" className={s.num} title="The model's bytes, stored once">
        Stored once
      </th>
      <th scope="col" className={s.num}>
        Per GPU
      </th>
      <th scope="col" className={s.num}>
        All {gpus} GPUs
      </th>
      <th
        scope="col"
        className={s.num}
        title="All the GPUs hold, over one copy stored once"
      >
        Amplified
      </th>
    </>
  );
}

/* The cache's sizes that do not depend on the context picked: a token of
   context, the windows when full, a request's state. */
function SizeTable({ sizes, gpus }) {
  const lines = [
    ["Each token of context", sizes.token],
    ["Sliding windows, when full", sizes.window],
    ["State per request", sizes.request],
  ].filter(([, size]) => size != null);
  return (
    <table className={`${s.table} ${s.sizes}`}>
      <thead>
        <tr>
          <th scope="col">
            <span className={s.visuallyHidden}>Figure</span>
          </th>
          <SizeHeads gpus={gpus} />
        </tr>
      </thead>
      <tbody>
        {lines.map(([label, size]) => (
          <tr key={label}>
            <th scope="row">{label}</th>
            <SizeCells size={size} gpus={gpus} strong />
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/* One request at the picked length, part by part: what each part holds,
   where it sits on the GPUs, its three sizes and whether a later request
   can reuse it. */
function PartsAtLength({ sizes, gpus, tokens }) {
  const parts = sizes.rows.length > 1;
  return (
    <div className={s.scroll}>
      <table className={`${s.table} ${s.wide}`}>
        <thead>
          <tr>
            <th scope="col">Part</th>
            <th scope="col">At {tokens.toLocaleString("en-US")} tokens</th>
            {gpus > 1 && <th scope="col">On the GPUs</th>}
            <SizeHeads gpus={gpus} />
            <th scope="col" className={s.layout}>
              Prefix reuse
            </th>
          </tr>
        </thead>
        <tbody>
          {sizes.rows.map((row, index) => (
            <tr key={row.name}>
              <th scope="row">
                {parts && <i data-tone={index % TONES} aria-hidden="true" />}
                <code>{row.name}</code>
              </th>
              <td title={GROWTH_NOTE[row.kind]}>
                {row.holds}
                <small>
                  {bytesText(row.entry.gpu)} {row.perEntry}
                  {gpus > 1 && " per GPU"}
                </small>
              </td>
              {gpus > 1 && <td>{placementWords(row.placement, gpus)}</td>}
              <SizeCells size={row} gpus={gpus} strong />
              <td className={s.layout}>{row.reusable ? "Yes" : "No"}</td>
            </tr>
          ))}
        </tbody>
        {parts && (
          <tfoot>
            <tr>
              <th scope="row">Total</th>
              <td />
              {gpus > 1 && <td />}
              <SizeCells size={sizes.total} gpus={gpus} strong />
              <td className={s.layout} />
            </tr>
          </tfoot>
        )}
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
        {facts.oneCopy != null && (
          <div>
            <dt>One request, stored once</dt>
            <dd className={s.secondary}>{bytesText(facts.oneCopy)}</dd>
          </div>
        )}
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
      <Amplification facts={facts} />

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

/* How many times over the partition stores one request: the simulator's
   split into replication across GPUs and the engine's overhead. Hidden when
   the API sends no amplification. */
function Amplification({ facts }) {
  if (facts.oneCopy == null || facts.amplification == null) return null;
  const split = amplificationText(facts);
  const gpus = facts.gpusPerPartition;
  return (
    <p className={s.note}>
      {gpus > 1 ? `The partition's ${gpus} GPUs together hold` : "The GPU holds"}{" "}
      <b>{timesText(facts.amplification)}</b> the request stored once.
      {split && <> That is {split}.</>} Replication comes from parts that every GPU
      keeps whole instead of a slice of; engine overhead is vLLM's padding, extra
      state copies, block rounding and tokens in flight.
    </p>
  );
}

/* One request part by part: a bar of shares over the table that names them.
   A one-part cache needs no shares. */
function RequestParts({ rows, children }) {
  const parts = rows.length > 1;
  // One copy is the model's bytes over all its GPUs, not per GPU.
  const amplified = rows.every((row) => row.oneCopy != null);
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
            {amplified && (
              <th
                scope="col"
                className={s.num}
                title="The model's bytes, stored once"
              >
                One copy
              </th>
            )}
            {amplified && (
              <th
                scope="col"
                className={s.num}
                title="All its GPUs hold, over one copy"
              >
                Amplified
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
              {amplified && <td className={s.num}>{bytesText(row.oneCopy)}</td>}
              {amplified && <td className={s.num}>{timesText(row.amplified)}</td>}
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
  const placed = rows.every((row) => row.placement);
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
            {placed && <th scope="col">On its GPUs</th>}
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
              {placed && <td>{row.placement}</td>}
            </tr>
          ))}
        </tbody>
      </table>
      {engine && <p className={s.note}>{engine}</p>}
    </>
  );
}
