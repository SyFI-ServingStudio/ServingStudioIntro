import s from "./ProductIntroduction.module.css";

/* The simulation foundation and Agent workflow share the capabilities and
   evidence presented in the sections below. */
const productParts = [
  {
    name: "Simulator",
    role: "Predicts serving performance.",
    items: [
      [
        "Models",
        "Dense and mixture-of-experts architectures in BF16, FP8, and NVFP4",
      ],
      ["Deployments", "H200 and B200 GPUs with TP, EP, DP, and PP"],
      [
        "Execution",
        "Unified serving, prefill–decode or attention–FFN disaggregation, and speculative decoding",
      ],
      ["Results", "Throughput, TTFT, TPOT, and per-kernel timing"],
      ["Calibration", "Checked against measurements from vLLM and SGLang runs"],
    ],
  },
  {
    name: "Agent",
    role: "Runs the optimization loop.",
    items: [
      ["Designs", "Turns a serving question into an executable experiment"],
      ["Searches", "Sweeps configurations and compares their tradeoffs"],
      [
        "Explains",
        "Traces performance back to specific kernels and system behavior",
      ],
      ["Builds", "Implements the selected change in vLLM or SGLang"],
      [
        "Validates",
        "Benchmarks the change on real hardware and explains any remaining gap",
      ],
    ],
  },
];

export function ProductIntroduction() {
  return (
    <section
      className={`section ${s.productSection}`}
      id="product"
      aria-labelledby="product-title"
    >
      <div className="wrap">
        <div className="section-intro" data-reveal>
          <h2 id="product-title">The Simulator predicts. The Agent acts.</h2>
          <p>
            Grounded in measured GPU kernel timings, the Simulator predicts
            performance across models, hardware, and serving configurations. The
            Agent runs the optimization loop: it designs the experiment, sweeps
            configurations, explains the result, implements the change in vLLM or
            SGLang, and validates it on real hardware.
          </p>
        </div>
        <div className={s.productParts}>
          {productParts.map((part) => (
            <article className={s.productPart} key={part.name}>
              <h3>{part.name}</h3>
              <p className={s.productPartRole}>{part.role}</p>
              <dl>
                {part.items.map(([term, detail]) => (
                  <div key={term}>
                    <dt>{term}</dt>
                    <dd>{detail}</dd>
                  </div>
                ))}
              </dl>
            </article>
          ))}
        </div>
      </div>
    </section>
  );
}
