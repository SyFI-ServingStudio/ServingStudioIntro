import { metricDoc } from "./kernelData";
import s from "./KernelDetail.module.css";

/* The metric on the y axis and the axis scales, above either chart. The
   choices live in the URL (y, scale, peak), so both views read one setting. */
export function ChartBar({
  kernel,
  metrics,
  y,
  logX,
  logY,
  positiveX,
  showPeak,
  update,
}) {
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
            checked={logY}
            onChange={(event) =>
              update({ scale: scale(logX, event.target.checked) })
            }
          />
          Log y
        </label>
        <label className={s.toggle}>
          <input
            type="checkbox"
            checked={showPeak}
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
