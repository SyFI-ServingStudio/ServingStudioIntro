import { useEffect, useRef, useState } from "react";
import { ArrowRight, ArrowUpRight, X } from "lucide-react";
import s from "./RealUseCases.module.css";
import qwenStory from "./qwenStory.json";
import { CaseInvestigation } from "./CaseInvestigation";
import qwenAfter from "../../public/case-studies/qwen-after-alignment.json";

const cases = [
  {
    id: "sglang-autotune",
    stack: "SGLang · GLM-5.2",
    title: "Autotune MoE kernels for prefill graphs.",
    summary:
      "ServingStudio Sim exposed slower-than-expected MoE kernels during SGLang prefill. Autotuning was enabled, but it did not cover the CUDA graph path used by prefill. Tuning that path increased input throughput by 5.6%.",
    metric: "+5.6%",
    metricLabel: "higher input throughput than the unpatched baseline",
    setup: "NVFP4 · 4 × B200 · TP4",
    comparison: {
      label: "Median workload duration",
      unit: "s",
      before: "44.873",
      after: "42.503",
      beforeLabel: "Baseline",
      afterLabel: "Patched",
    },
    link: {
      label: "Read the SGLang PR #38560",
      href: "https://github.com/sgl-project/sglang/pull/38560",
    },
  },
  {
    id: "spec5-graphs",
    stack: "vLLM · GLM-5.2 MTP",
    title: "Restore CUDA graph replay for speculative decoding.",
    summary:
      "With five MTP draft tokens, vLLM ran a common 2,048-token batch eagerly because its graph-size filter excluded that batch. Raising the scheduler and graph limits to 2,052 tokens restored CUDA graph replay. The updated configuration delivered 10.8% higher output throughput in a follow-up benchmark.",
    metric: "+10.8%",
    metricLabel: "higher output throughput than the 2,048-token baseline",
    setup: "NVFP4 · 4 × B200 · TP4 + EP4",
    comparison: {
      label: "Mean iteration time in the diagnostic window",
      unit: "ms",
      before: "141.048",
      after: "114.903",
      beforeLabel: "2,048 tokens",
      afterLabel: "2,052 tokens",
    },
    link: {
      label: "View the Spec5 implementation",
      href: "https://github.com/SyFI-ServingStudio/ServingStudioSim/pull/32",
    },
  },
  {
    id: "qwen-fusion",
    stack: "Mini-SGLang · Qwen3-235B",
    title: "Build and optimize Qwen3-235B in Mini-SGLang.",
    summary:
      "Mini-SGLang did not support MoE models, so we built a Qwen3-235B serving path and used ServingStudio Sim to guide kernel selection, communication design, and operator fusion. On the tested prefill-heavy workload, the final implementation delivered 25.6% higher output throughput than vLLM.",
    metric: "+25.6%",
    metricLabel: "higher output throughput than vLLM",
    setup: "FP8 · 4 × H200 · TP4 + EP4",
    comparison: {
      label: "Measured output throughput · 256 requests at concurrency 32",
      unit: "tok/s",
      before: "80.741",
      after: "101.406",
      beforeLabel: "vLLM",
      afterLabel: "Mini-SGLang",
    },
  },
];

function Comparison({ comparison }) {
  return (
    <figure className={s.comparison}>
      <figcaption>{comparison.label}</figcaption>
      <div className={s.readings}>
        <div>
          <span>{comparison.beforeLabel}</span>
          <strong>
            {comparison.before}
            <small>{comparison.unit}</small>
          </strong>
        </div>
        <ArrowRight size={22} aria-hidden="true" />
        <div>
          <span>{comparison.afterLabel}</span>
          <strong>
            {comparison.after}
            <small>{comparison.unit}</small>
          </strong>
        </div>
      </div>
    </figure>
  );
}

function KernelBreakdown({ rows, after = false }) {
  const maximum = Math.max(
    ...rows.flatMap((row) => [row.measured_ms, row.simulated_ms]),
  );
  return (
    <figure className={s.kernelChart}>
      <figcaption>
        {after
          ? "Kernel timings after switching to FA3 and native EP"
          : "Where the first implementation differed from the simulation"}
      </figcaption>
      <p className={s.chartKey}>
        Time per operation · ms ·{" "}
        {after
          ? "mean of three captured prefill iterations"
          : "mean of two steady prefill iterations"}
      </p>
      {rows.map((row) => (
        <div className={s.kernelRow} key={row.operation}>
          <h4>{row.label}</h4>
          {[
            ["simulated_ms", "Predicted"],
            ["measured_ms", "Measured"],
          ].map(([key, label]) => (
            <div className={s.kernelReading} key={key}>
              <span>{label}</span>
              <div className={s.barTrack} aria-hidden="true">
                <i
                  className={key === "measured_ms" ? s.measuredBar : s.predictedBar}
                  style={{ width: `${(row[key] / maximum) * 100}%` }}
                />
              </div>
              <strong>{row[key].toFixed(3)}</strong>
            </div>
          ))}
        </div>
      ))}
      <p className={s.chartKey}>
        {after ? (
          "Selected operation times after the changes, measured for one transformer layer during a 16K-token prefill on four H200 GPUs. The implementation no longer uses the old dispatch path, but the prediction still includes its cost. The chart therefore does not establish complete alignment."
        ) : (
          <>
            Selected operation times from the original alignment export, measured
            for one transformer layer during a 16K-token prefill on four H200 GPUs.
            The attention backends and MoE execution paths differed, which motivated
            the implementation changes.
          </>
        )}
      </p>
      <a
        className={s.sourceLink}
        href={`${import.meta.env.BASE_URL}case-studies/${after ? "qwen-after-alignment.json" : "qwen-kernel-breakdown.json"}`}
        download
      >
        {after
          ? "Download these operation timings"
          : "Download all 21 operation timings"}{" "}
        <ArrowUpRight size={18} aria-hidden="true" />
      </a>
    </figure>
  );
}

function QuantizationTable({ table }) {
  return (
    <figure className={s.evidenceTable}>
      <figcaption>{table.caption}</figcaption>
      <table>
        <thead>
          <tr>
            {table.headers.map((header) => (
              <th key={header} scope="col">
                {header === "Measured time" ? "Time" : header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {table.rows.map((row) => (
            <tr key={row[0]}>
              {row.map((cell, index) =>
                index === 0 ? (
                  <th scope="row" key={cell}>
                    {cell}
                  </th>
                ) : (
                  <td key={cell}>{cell}</td>
                ),
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </figure>
  );
}

export function RealUseCases() {
  const [active, setActive] = useState(null);
  const dialogRef = useRef(null);
  const openerRef = useRef(null);

  useEffect(() => {
    if (!active) return;
    const dialog = dialogRef.current;
    const previousOverflow = document.body.style.overflow;
    dialog.showModal();
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = previousOverflow;
      dialog.close();
      openerRef.current?.focus({ preventScroll: true });
    };
  }, [active]);

  return (
    <section
      className="section wrap"
      id="real-use-cases"
      aria-labelledby="real-use-cases-title"
    >
      <div className="section-intro" data-reveal>
        <h2 id="real-use-cases-title">
          From simulation to faster serving systems.
        </h2>
        <p>
          By comparing simulated and measured performance, we found opportunities
          to improve SGLang and vLLM and to optimize MoE models as we implemented
          them in Mini-SGLang. These case studies show what we changed and how much
          faster each system ran.
        </p>
      </div>
      <div className={s.grid}>
        {cases.map((item) => (
          <article className={s.card} key={item.id}>
            <p className={s.stack}>{item.stack}</p>
            <h3>{item.title}</h3>
            <p className={s.summary}>{item.summary}</p>
            <div className={s.metric}>
              <strong>{item.metric}</strong>
              <span>{item.metricLabel}</span>
            </div>
            <p className={s.setup}>{item.setup}</p>
            <button
              className={s.readMore}
              type="button"
              aria-haspopup="dialog"
              aria-label={`Read more: ${item.title}`}
              onClick={(event) => {
                openerRef.current = event.currentTarget;
                setActive(item);
              }}
            >
              Read more <ArrowUpRight size={19} aria-hidden="true" />
            </button>
          </article>
        ))}
      </div>
      {active && (
        <dialog
          ref={dialogRef}
          className={s.dialog}
          aria-labelledby="case-story-title"
          onCancel={() => setActive(null)}
          onClose={() => setActive(null)}
          onClick={(event) => {
            if (event.target === event.currentTarget) {
              const bounds = event.currentTarget.getBoundingClientRect();
              if (
                event.clientX < bounds.left ||
                event.clientX > bounds.right ||
                event.clientY < bounds.top ||
                event.clientY > bounds.bottom
              )
                setActive(null);
            }
          }}
        >
          <div className={s.dialogBar}>
            <span>{active.stack}</span>
            <button
              type="button"
              aria-label="Close case study"
              onClick={() => setActive(null)}
              autoFocus
            >
              <X size={24} />
            </button>
          </div>
          <article className={s.story}>
            <h2 id="case-story-title">{active.title}</h2>
            <p className={s.setup}>{active.setup}</p>
            <CaseInvestigation
              caseId={active.id}
              outcome={<Comparison comparison={active.comparison} />}
              charts={{
                kernels: (
                  <KernelBreakdown
                    rows={qwenStory.find((chapter) => chapter.chart).chart}
                  />
                ),
                after: <KernelBreakdown rows={qwenAfter.rows} after />,
                quantization: (
                  <QuantizationTable
                    table={qwenStory.find((chapter) => chapter.table).table}
                  />
                ),
              }}
            />
            {active.link && (
              <a
                className={s.sourceLink}
                href={active.link.href}
                target="_blank"
                rel="noreferrer"
              >
                {active.link.label}
                <ArrowUpRight size={18} aria-hidden="true" />
              </a>
            )}
          </article>
        </dialog>
      )}
    </section>
  );
}
