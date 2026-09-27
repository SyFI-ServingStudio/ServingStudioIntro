import { metricDoc } from "./kernelData";
import s from "./KernelDetail.module.css";

/* The metric on the y axis and the axis scales, above either chart. The
   choices live in the URL (y, scale, peak), so both views read one setting.

   A chart that can divide its lines by one of them passes `relative`:
   { on, reference, set }, whether it does, the reference line's label and
   how to change it. Log y and the spec-sheet peak mean nothing on a ratio,
   so they are off while it is on. */
export function ChartBar({
  kernel,
  metrics,
  y,
  logX,
  logY,
  positiveX,
  showPeak,
  relative,
  update,
}) {
  const ratio = Boolean(relative?.on);
  const scale = (x, yLog) => `${x ? "logx" : ""}${yLog ? "logy" : ""}` || "linear";
  return (
    <div className={s.chartBar}>
      <div className={s.metricSwitch} role="radiogroup" aria-label="Metric">
        {metrics.map((m) => (
          <button
            key={m}
            type="button"
            role="radio"
            aria-checked={m === y}
            onClick={() => update({ y: m })}
          >
            {metricDoc(kernel, m).label}
            <span>{metricDoc(kernel, m).unit}</span>
          </button>
        ))}
      </div>
      <div className={s.chartOptions}>
        {relative && (
          <div className={s.modeSwitch} role="radiogroup" aria-label="Y values">
            {[
              [false, "Absolute"],
              [true, `Relative to ${relative.reference}`],
            ].map(([on, text]) => (
              <button
                key={text}
                type="button"
                role="radio"
                aria-checked={on === ratio}
                onClick={() => relative.set(on)}
              >
                {text}
              </button>
            ))}
          </div>
        )}
        <label className={s.toggle}>
          <input
            type="checkbox"
            checked={logX}
            disabled={!positiveX}
            onChange={(event) =>
              update({ scale: scale(event.target.checked, logY) })
            }
          />
          Log x
        </label>
        <label className={s.toggle}>
          <input
            type="checkbox"
            checked={logY && !ratio}
            disabled={ratio}
            onChange={(event) =>
              update({ scale: scale(logX, event.target.checked) })
            }
          />
          Log y
        </label>
        <label className={s.toggle}>
          <input
            type="checkbox"
            checked={showPeak && !ratio}
            disabled={ratio}
            onChange={(event) =>
              update({ peak: event.target.checked ? "" : "off" })
            }
          />
          Spec-sheet peak
        </label>
      </div>
    </div>
  );
}
