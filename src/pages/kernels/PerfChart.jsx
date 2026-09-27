import { ZoomIn, ZoomOut } from "lucide-react";
import { useEffect, useId, useMemo, useRef, useState } from "react";
import { formatDate, formatNumber, formatValue } from "./kernelData";
import s from "./PerfChart.module.css";

/* Categorical slots, stepped bright for the dark page so lines read at a
   glance. Slots 1-4 stay apart for every pair, colour-blind included (worst
   CVD ΔE 16.6 vs #0c0d0f); most kernels have one to three backends. */
export const SERIES_COLORS = [
  "#5ad8f5",
  "#ff6b6b",
  "#f4d35e",
  "#9d7dff",
  "#ff9f43",
  "#f06bd8",
  "#6fdc5a",
  "#62b6ff",
];

/* The colour of one value among the lines of a chart. Up to four values keep
   one colour each whatever the filters, and any four slots tell apart. Past
   that a fixed mapping can hand the lines on screen three neighbouring warm
   slots, so the lines shown take the slots in order. */
const STABLE_SLOTS = 4;
export const seriesColor = (all, present, value) =>
  SERIES_COLORS[
    all.length <= STABLE_SLOTS ? all.indexOf(value) : present.indexOf(value)
  ];

const MARGIN = { top: 40, right: 24, bottom: 56, left: 76 };
// Past this many points a series reads as a line; markers only mark the hover.
const MARKER_LIMIT = 40;
// How far one pixel of wheel travel zooms: 100px, a mouse notch, is ~14%.
const WHEEL_ZOOM = 0.0015;
// Wheel events closer together than this are one gesture, momentum included.
const WHEEL_GESTURE_GAP = 250;

function logTicks(min, max, base) {
  const ticks = [];
  const start = Math.floor(Math.log(min) / Math.log(base));
  const end = Math.ceil(Math.log(max) / Math.log(base));
  for (let e = start; e <= end; e += 1) {
    const value = base ** e;
    if (value >= min * 0.999 && value <= max * 1.001) ticks.push(value);
  }
  return ticks;
}

function decadeTicks(min, max) {
  const ticks = [];
  for (
    let e = Math.floor(Math.log10(min));
    e <= Math.ceil(Math.log10(max));
    e += 1
  ) {
    for (const m of [1, 2, 5]) {
      const value = m * 10 ** e;
      if (value >= min && value <= max) ticks.push(value);
    }
  }
  return ticks.length > 9
    ? ticks.filter((v) => /^1/.test(v.toExponential()))
    : ticks;
}

// Round steps between lo and hi, for an x axis that need not start at zero.
function rangeTicks(lo, hi) {
  const raw = (hi - lo) / 5 || Math.abs(hi) || 1;
  const step = [1, 2, 2.5, 5, 10]
    .map((m) => m * 10 ** Math.floor(Math.log10(raw)))
    .find((v) => v >= raw);
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step)
    ticks.push(Number(v.toPrecision(6)));
  return ticks;
}

function linearTicks(max) {
  const raw = max / 5;
  const step = [1, 2, 2.5, 5, 10]
    .map((m) => m * 10 ** Math.floor(Math.log10(raw)))
    .find((v) => v >= raw);
  const ticks = [];
  for (let v = 0; v < max + step; v += step) {
    ticks.push(Number(v.toPrecision(6)));
    if (v >= max) break;
  }
  return ticks;
}

// A zoomed edge falls between measured values; three figures say where.
const round = (x) => Number(x.toPrecision(3));

function thin(ticks, count) {
  if (ticks.length <= count) return ticks;
  const every = Math.ceil(ticks.length / count);
  return ticks.filter((_, i) => i % every === 0);
}

/* A series' points may include { x, y: null }: an x the chart covers but the
   series has no value at (a grid cell nobody measured). The line breaks
   there, the hover reads "not measured", and a point left alone between two
   gaps keeps its marker so it does not vanish.

   `cells` marks the simulator's grid along the bottom of the plot, one tick
   per cell: { x, status: "measured" | "missing" | "infeasible" }. */
export function PerfChart({
  series,
  xName,
  xLabel = xName,
  xUnit,
  colorName,
  yLabel,
  yUnit,
  logX,
  logY,
  peak,
  cells,
  describe,
  note = "Each point is one measured row; nothing between points is interpolated.",
  foot = (points) => <Provenance points={points} />,
}) {
  const wrapRef = useRef(null);
  // useId gives characters a url(#…) reference would have to escape.
  const clipId = `plot-clip-${useId().replace(/[^\w-]/g, "")}`;
  const [width, setWidth] = useState(900);
  // The x value under the pointer or keyboard, not an index: zooming changes
  // which points are on screen but not which one is being read.
  const [active, setActive] = useState(null);
  // A zoom is a range of x values. It belongs to one x dimension, so picking
  // another x axis shows that one whole.
  const [zoomState, setZoomState] = useState(null);
  const [panning, setPanning] = useState(false);
  // Pointers down on the plot, and the gesture they started: one pointer pans,
  // two pinch.
  const pointers = useRef(new Map());
  const gesture = useRef(null);
  // The range on screen as of the latest event, which can run ahead of the
  // last render while the wheel spins.
  const rangeRef = useRef(null);
  const wheelRef = useRef(null);
  // Who the current wheel gesture belongs to, the chart or the page, and when
  // its last event came.
  const wheelGesture = useRef({ owner: "page", last: -Infinity });
  useEffect(() => {
    const observer = new ResizeObserver(([entry]) =>
      setWidth(entry.contentRect.width),
    );
    observer.observe(wrapRef.current);
    // React listens to the wheel passively, and a zoom has to stop the page
    // scrolling under it.
    const el = wrapRef.current;
    const onWheel = (event) => wheelRef.current(event);
    el.addEventListener("wheel", onWheel, { passive: false });
    // A gesture that starts off the chart scrolls the page, and keeps doing so
    // when the chart scrolls in under the pointer.
    const onPageWheel = (event) => {
      if (el.contains(event.target)) return;
      const g = wheelGesture.current;
      if (event.timeStamp - g.last > WHEEL_GESTURE_GAP) g.owner = "page";
      g.last = event.timeStamp;
    };
    window.addEventListener("wheel", onPageWheel, {
      capture: true,
      passive: true,
    });
    return () => {
      observer.disconnect();
      el.removeEventListener("wheel", onWheel);
      window.removeEventListener("wheel", onPageWheel, { capture: true });
    };
  }, []);
  const height = width < 640 ? 320 : 440;
  const plotW = Math.max(120, width - MARGIN.left - MARGIN.right);
  const plotH = height - MARGIN.top - MARGIN.bottom;

  const geometry = useMemo(() => {
    const points = series.flatMap((serie) => serie.points);
    const allXs = [...new Set(points.map((p) => p.x))].sort((a, b) => a - b);
    const full = [allXs[0], allXs[allXs.length - 1]];
    const wanted = zoomState?.xName === xName ? zoomState.range : null;
    // A zoom outside what the current choices measured shows everything.
    const zoom =
      wanted && wanted[1] > full[0] && wanted[0] < full[1] ? wanted : null;
    const [xMin, xMax] = zoom ?? full;
    const xs = allXs.filter((x) => x >= xMin && x <= xMax);
    // The y range covers each line up to its first point past either side, so
    // a line that runs out of the frame leaves by a side, not the top.
    const ys = series
      .flatMap(({ points: line }) => {
        const first = Math.max(0, line.findIndex((p) => p.x >= xMin) - 1);
        let last = line.findLastIndex((p) => p.x <= xMax) + 1;
        if (last === 0) last = line.length - 1;
        return line.slice(first, last + 1);
      })
      .map((p) => p.y)
      .filter((y) => y > 0);
    if (peak) ys.push(peak.value);
    let yMin = Math.min(...ys);
    let yMax = Math.max(...ys);
    if (logY) {
      yMin = 10 ** (Math.floor(Math.log10(yMin) * 2) / 2);
      yMax = 10 ** (Math.ceil(Math.log10(yMax) * 2) / 2);
    } else {
      yMin = 0;
      const ticks = linearTicks(yMax * 1.05);
      yMax = ticks[ticks.length - 1];
    }
    // Zooming works in the scale on screen, so on a log axis a step in or
    // out covers the same number of doublings wherever it lands.
    const tx = (x) => (logX ? Math.log2(x) : x);
    const fromT = (v) => (logX ? 2 ** v : v);
    const sx = (x) => {
      if (xMin === xMax) return plotW / 2;
      return ((tx(x) - tx(xMin)) / (tx(xMax) - tx(xMin))) * plotW;
    };
    // The closest two measured points can fill the plot, and no closer.
    const minSpan = Math.min(...allXs.slice(1).map((x, i) => tx(x) - tx(allXs[i])));
    const sy = (y) =>
      logY
        ? plotH -
          ((Math.log10(y) - Math.log10(yMin)) /
            (Math.log10(yMax) - Math.log10(yMin))) *
            plotH
        : plotH - (y / yMax) * plotH;
    const tickCount = width < 640 ? 4 : 8;
    const xTicks = thin(
      logX ? logTicks(xMin, xMax, 2) : rangeTicks(xMin, xMax),
      tickCount,
    );
    const yTicks = logY ? decadeTicks(yMin, yMax) : linearTicks(yMax);
    return {
      allXs,
      full,
      zoom,
      xs,
      tx,
      fromT,
      minSpan,
      sx,
      sy,
      // Zoomed in past the powers of two, round steps label the axis.
      xTicks: xTicks.length >= 2 ? xTicks : thin(rangeTicks(xMin, xMax), tickCount),
      yTicks,
    };
  }, [series, peak, logX, logY, plotW, plotH, width, zoomState, xName]);

  const { full, zoom, xs, tx, fromT, minSpan, sx, sy, xTicks, yTicks } = geometry;
  const activeX = xs.includes(active) ? active : null;
  const [xMin, xMax] = zoom ?? full;
  const inView = (p) => p.x >= xMin && p.x <= xMax;
  rangeRef.current = [xMin, xMax].map(tx);
  const fullT = full.map(tx);
  const canZoomIn = full[0] !== full[1] && fullT[1] - fullT[0] > minSpan;

  /* Every zoom and pan lands here, as a range in the scale on screen. It keeps
     the range inside what was measured and no narrower than the closest two
     points; the whole range is no zoom at all. */
  function show([lo, hi]) {
    const fullSpan = fullT[1] - fullT[0];
    let span = hi - lo;
    if (span >= fullSpan * 0.999) {
      rangeRef.current = fullT;
      return setZoomState(null);
    }
    if (span < minSpan) {
      const mid = (lo + hi) / 2;
      span = minSpan;
      [lo, hi] = [mid - span / 2, mid + span / 2];
    }
    if (lo < fullT[0]) [lo, hi] = [fullT[0], fullT[0] + span];
    if (hi > fullT[1]) [lo, hi] = [fullT[1] - span, fullT[1]];
    rangeRef.current = [lo, hi];
    setZoomState({ xName, range: [fromT(lo), fromT(hi)] });
  }
  const setZoom = (range) => (range ? show(range.map(tx)) : show(fullT));
  // Scale the range by factor, keeping the value at c where it is on screen.
  function zoomAt(factor, c, base = rangeRef.current) {
    const [lo, hi] = base;
    show([c - (c - lo) * factor, c + (hi - c) * factor]);
  }
  const plotX = (clientX) =>
    clientX - wrapRef.current.getBoundingClientRect().left - MARGIN.left;
  const valueAt = (px, [lo, hi] = rangeRef.current) =>
    lo + (Math.min(plotW, Math.max(0, px)) / plotW) * (hi - lo);
  // The buttons and keys zoom around the point being read, else the middle.
  function zoomBy(factor) {
    const [lo, hi] = rangeRef.current;
    zoomAt(factor, activeX == null ? (lo + hi) / 2 : tx(activeX));
  }

  /* Scrolling is the page's: ctrl + scroll zooms around the pointer (a
     trackpad pinch arrives as the same), and a sideways swipe pans a zoomed
     chart. A pinch or ctrl + scroll over the chart never zooms the page.

     A sideways gesture belongs to whoever its first event went to, like a
     scroller inside a page: one that starts at an edge (or with nothing
     zoomed) is the page's, and one that starts panning stops at the edge
     instead of spilling its momentum into the page. */
  wheelRef.current = (event) => {
    const unit = [1, 16, plotH][event.deltaMode];
    const [lo, hi] = rangeRef.current;
    const g = wheelGesture.current;
    const fresh = event.timeStamp - g.last > WHEEL_GESTURE_GAP;
    g.last = event.timeStamp;

    if (event.ctrlKey || event.metaKey) {
      g.owner = "chart";
      event.preventDefault();
      // A pinch reports small steps; a mouse wheel with ctrl, whole notches.
      const delta = event.deltaY * unit * (Math.abs(event.deltaY) < 50 ? 5 : 1);
      if (delta < 0 && (!canZoomIn || hi - lo <= minSpan * 1.001)) return;
      return zoomAt(Math.exp(delta * WHEEL_ZOOM), valueAt(plotX(event.clientX)));
    }
    if (Math.abs(event.deltaX) <= Math.abs(event.deltaY)) {
      if (fresh) g.owner = "page";
      return;
    }
    const delta = event.deltaX * unit;
    const zoomed = hi - lo < fullT[1] - fullT[0] - 1e-9;
    const canPan =
      zoomed && (delta > 0 ? hi < fullT[1] - 1e-9 : lo > fullT[0] + 1e-9);
    if (fresh) g.owner = canPan ? "chart" : "page";
    if (g.owner === "page") return;
    event.preventDefault();
    if (!canPan) return;
    const d = (delta / plotW) * (hi - lo);
    show([lo + d, hi + d]);
  };

  function pick(clientX) {
    if (!xs.length) return setActive(null);
    const px = plotX(clientX);
    let best = xs[0];
    xs.forEach((x) => {
      if (Math.abs(sx(x) - px) < Math.abs(sx(best) - px)) best = x;
    });
    setActive(best);
  }

  // One pointer drags the zoomed range along; two pinch it around their middle.
  function startGesture() {
    const at = [...pointers.current.values()];
    const base = rangeRef.current;
    gesture.current =
      at.length >= 2
        ? {
            base,
            dist: Math.abs(at[0] - at[1]) || 1,
            c: valueAt(plotX((at[0] + at[1]) / 2)),
          }
        : at.length === 1
          ? { base, x: at[0] }
          : null;
  }
  function onPointerDown(event) {
    if (event.pointerType === "mouse" && event.button !== 0) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    pointers.current.set(event.pointerId, event.clientX);
    startGesture();
  }
  function onPointerMove(event) {
    const g = gesture.current;
    if (!g || !pointers.current.has(event.pointerId)) {
      if (event.pointerType === "mouse") pick(event.clientX);
      return;
    }
    pointers.current.set(event.pointerId, event.clientX);
    const at = [...pointers.current.values()];
    if (at.length >= 2) {
      const dist = Math.abs(at[0] - at[1]) || 1;
      return zoomAt(g.dist / dist, g.c, g.base);
    }
    const dx = event.clientX - g.x;
    if (!zoom || Math.abs(dx) < 3) return pick(event.clientX);
    setPanning(true);
    setActive(null);
    const d = (-dx / plotW) * (g.base[1] - g.base[0]);
    show([g.base[0] + d, g.base[1] + d]);
  }
  function onPointerUp(event) {
    pointers.current.delete(event.pointerId);
    setPanning(false);
    startGesture();
  }
  function onKeyDown(event) {
    const keys = [
      "ArrowLeft",
      "ArrowRight",
      "Home",
      "End",
      "Escape",
      "+",
      "=",
      "-",
      "0",
    ];
    if (!keys.includes(event.key)) return;
    event.preventDefault();
    if (event.key === "Escape") return setActive(null);
    if (event.key === "+" || event.key === "=") return zoomBy(0.5);
    if (event.key === "-") return zoomBy(2);
    if (event.key === "0") return setZoom(null);
    if (event.key === "Home") return setActive(xs[0]);
    if (event.key === "End") return setActive(xs[xs.length - 1]);
    const step = event.key === "ArrowRight" ? 1 : -1;
    const i = xs.indexOf(activeX);
    setActive(xs[i < 0 ? 0 : Math.min(xs.length - 1, Math.max(0, i + step))]);
  }

  // End labels only while they stay attached to their lines: at most four
  // series, and only when their ends sit at least one line height apart.
  const ends = series
    .filter((serie) => serie.points.some((p) => inView(p) && p.y != null))
    .map((serie) => {
      const last = serie.points.findLast((p) => inView(p) && p.y != null);
      return { serie, x: sx(last.x), y: sy(last.y) };
    })
    .sort((a, b) => a.y - b.y);
  const labelEnds =
    series.length > 1 &&
    series.length <= 4 &&
    width >= 640 &&
    ends.every((end, i) => i === 0 || end.y - ends[i - 1].y >= 18);

  // Past MARKER_LIMIT points on screen a series reads as a line.
  const dense = (serie) =>
    serie.points.filter((p) => p.y != null && inView(p)).length > MARKER_LIMIT;

  const hovered =
    activeX == null || panning
      ? []
      : series
          .map((serie) => ({
            serie,
            point: serie.points.find((p) => p.x === activeX),
          }))
          .filter((h) => h.point);
  const tooltipLeft = activeX == null ? 0 : MARGIN.left + sx(activeX);
  const flip = tooltipLeft > width * 0.6;

  return (
    <figure className={s.figure}>
      <div className={s.head}>
        {series.length === 1 && (
          <p className={s.single}>
            {colorName} <code>{series[0].label}</code>
          </p>
        )}
        {series.length > 1 && (
          <ul className={s.legend} aria-label="Series">
            {series.map((serie) => (
              <li key={serie.key}>
                <i style={{ background: serie.color }} aria-hidden="true" />
                {serie.label}
              </li>
            ))}
          </ul>
        )}
        <div className={s.zoom} role="group" aria-label="Zoom">
          {zoom && (
            <span className={s.zoomRange}>
              {xName} {formatValue(round(xMin), xUnit)} to{" "}
              {formatValue(round(xMax), xUnit)}
            </span>
          )}
          <button
            type="button"
            aria-label="Zoom in"
            title="Zoom in (+)"
            disabled={
              !canZoomIn ||
              (zoom && rangeRef.current[1] - rangeRef.current[0] <= minSpan * 1.001)
            }
            onClick={() => zoomBy(0.5)}
          >
            <ZoomIn size={16} aria-hidden="true" />
          </button>
          <button
            type="button"
            aria-label="Zoom out"
            title="Zoom out (−)"
            disabled={!zoom}
            onClick={() => zoomBy(2)}
          >
            <ZoomOut size={16} aria-hidden="true" />
          </button>
          <button
            type="button"
            className={s.showAll}
            title="Show the whole range (0)"
            disabled={!zoom}
            onClick={() => setZoom(null)}
          >
            Show all
          </button>
        </div>
      </div>
      <div
        ref={wrapRef}
        className={`${s.plot} ${zoom ? s.pannable : ""} ${panning ? s.panning : ""}`}
        tabIndex={0}
        role="img"
        aria-label={describe}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onPointerLeave={() => setActive(null)}
        onDoubleClick={() => setZoom(null)}
        onKeyDown={onKeyDown}
        onBlur={() => setActive(null)}
      >
        <svg width={width} height={height} aria-hidden="true">
          <defs>
            {/* Lines run on to the points past a zoomed range, cut at the
                plot's sides, so they leave the frame instead of stopping. */}
            <clipPath id={clipId}>
              <rect y={-12} width={plotW} height={plotH + 24} />
            </clipPath>
          </defs>
          <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
            {yTicks.map((tick) => (
              <g key={tick} transform={`translate(0,${sy(tick)})`}>
                <line className={s.grid} x2={plotW} />
                <text className={s.tick} x={-12} dy="0.32em" textAnchor="end">
                  {formatNumber(tick, 2)}
                </text>
              </g>
            ))}
            <line className={s.axis} y1={plotH} y2={plotH} x2={plotW} />
            {xTicks.map((tick) => (
              <text
                key={tick}
                className={s.tick}
                x={sx(tick)}
                y={plotH + 24}
                textAnchor="middle"
              >
                {formatValue(tick, xUnit)}
              </text>
            ))}
            <text className={s.axisLabel} x={plotW} y={plotH + 48} textAnchor="end">
              {xLabel}
              {logX ? ", log scale" : ""}
              {zoom ? ", zoomed" : ""}
            </text>
            <text className={s.axisLabel} x={-MARGIN.left} y={-24}>
              {yLabel} ({yUnit})
            </text>

            {peak && (
              <g transform={`translate(0,${sy(peak.value)})`}>
                <line className={s.peak} x2={plotW} />
                <text className={s.peakLabel} x={8} y={-8}>
                  {width < 640 ? peak.short : peak.label}
                </text>
              </g>
            )}

            {activeX != null && (
              <line
                className={s.crosshair}
                x1={sx(activeX)}
                x2={sx(activeX)}
                y2={plotH}
              />
            )}

            {/* A series may set its own stroke `width`; wider lines draw last,
                on top of the rest. */}
            {[...series]
              .sort((a, b) => (a.width ?? 0) - (b.width ?? 0))
              .map((serie) => (
                <g key={serie.key}>
                  <path
                    className={s.line}
                    clipPath={`url(#${clipId})`}
                    stroke={serie.color}
                    style={serie.width ? { strokeWidth: serie.width } : undefined}
                    d={linePath(serie.points, sx, sy)}
                  />
                  {serie.points
                    .filter(
                      (p, i, all) =>
                        p.y != null &&
                        inView(p) &&
                        (!dense(serie) ||
                          p.x === activeX ||
                          (all[i - 1]?.y == null && all[i + 1]?.y == null)),
                    )
                    .map((p) => (
                      <circle
                        key={p.x}
                        className={s.dot}
                        cx={sx(p.x)}
                        cy={sy(p.y)}
                        r={p.x === activeX ? 6.5 : 4.5}
                        fill={serie.color}
                      />
                    ))}
                </g>
              ))}

            {cells?.filter(inView).map((cell) => (
              <g
                key={cell.x}
                className={`${s.cell} ${s[cell.status]}`}
                transform={`translate(${sx(cell.x)},${plotH})`}
              >
                {cell.status === "infeasible" ? (
                  <path d="M-3,-10L3,-4M3,-10L-3,-4" />
                ) : (
                  <line y1={-9} y2={-2} />
                )}
              </g>
            ))}

            {labelEnds &&
              ends.map((end) => (
                <text
                  key={end.serie.key}
                  className={s.endLabel}
                  x={Math.min(end.x + 10, plotW)}
                  y={end.y}
                  dy="0.32em"
                  textAnchor={end.x + 10 > plotW - 80 ? "end" : "start"}
                  transform={
                    end.x + 10 > plotW - 80 ? "translate(-18,-14)" : undefined
                  }
                >
                  {end.serie.label}
                </text>
              ))}
          </g>
        </svg>

        {hovered.length > 0 && (
          <div
            className={s.tooltip}
            style={{
              left: flip ? undefined : tooltipLeft + 16,
              right: flip ? width - tooltipLeft + 16 : undefined,
              top: MARGIN.top,
            }}
          >
            <p className={s.tooltipHead}>
              {xName} = {formatValue(activeX, xUnit)}
            </p>
            <table>
              <tbody>
                {hovered.map(({ serie, point }) => (
                  <tr key={serie.key}>
                    <th scope="row">
                      <i style={{ background: serie.color }} aria-hidden="true" />
                      {serie.label}
                    </th>
                    <td>
                      {point.y == null
                        ? "not measured"
                        : `${formatNumber(point.y)} ${yUnit}`}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {foot(hovered.map((h) => h.point))}
          </div>
        )}
      </div>
      <p className={s.hint}>
        Hover or focus the chart and use the arrow keys to read values. Hold ctrl
        and scroll, or pinch, to zoom around the pointer (+ and − work too); drag or
        swipe sideways to move along; double-click or press 0 to see all of it.{" "}
        {note}
      </p>
    </figure>
  );
}

// A path through the points, lifting the pen at each point with no value.
function linePath(points, sx, sy) {
  let pen = "M";
  let d = "";
  for (const p of points) {
    if (p.y == null) {
      pen = "M";
      continue;
    }
    d += `${pen}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`;
    pen = "L";
  }
  return d;
}

function Provenance({ points }) {
  points = points.filter((p) => p.record?.provenance);
  if (!points.length) return null;
  const dates = [
    ...new Set(
      points.map((p) => p.record.provenance.profiler_run_at?.slice(0, 10)),
    ),
  ];
  const hashes = [
    ...new Set(
      points.map((p) => p.record.provenance.profiler_git_hash?.slice(0, 7)),
    ),
  ];
  const cuda = [...new Set(points.map((p) => p.record.provenance.cuda_version))];
  return (
    <p className={s.tooltipFoot}>
      Measured{" "}
      {dates.length === 1
        ? formatDate(points[0].record.provenance.profiler_run_at)
        : `on ${dates.length} dates`}
      , profiler {hashes.length === 1 ? hashes[0] : `${hashes.length} commits`},
      CUDA {cuda.join(" and ")}
    </p>
  );
}
