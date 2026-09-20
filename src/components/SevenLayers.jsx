import { useLayoutEffect, useRef, useState } from "react";
import { ArrowUpRight, Cpu, Layers, Activity, Route } from "lucide-react";
import s from "./SevenLayers.module.css";

const layers = [
  {
    n: 7,
    name: "Run infrastructure",
    group: "Stateful execution",
    icon: Activity,
    summary: "Drive the experiment.",
    detail:
      "Replays the arrival trace, advances one global clock, drives the deployment, and streams request and cost logs to Parquet.",
    examples: [
      [
        "Trace frontend",
        "Release requests according to the configured arrival schedule.",
      ],
      [
        "Simulation loop",
        "Advance the deployment, record request state and apply the stopping condition.",
      ],
    ],
  },
  {
    n: 6,
    name: "Orchestrator",
    group: "Stateful execution",
    icon: Route,
    summary: "Route across worker pools.",
    detail:
      "Coordinates pool-local placement and inter-pool flow, including transfers and barriers. A pool represents a serving role such as prefill or decode.",
    examples: [
      [
        "Prefill/decode flow",
        "Coordinate prefill and decode pools, including cross-pool KV transfer.",
      ],
      [
        "DP pool controller",
        "Place requests using least-queued or round-robin routing.",
      ],
    ],
  },
  {
    n: 5,
    name: "Worker",
    group: "Stateful execution",
    icon: Activity,
    summary: "Own the serving state.",
    detail:
      "Manages KV resources, admission, selection and execution cadence. Lowers the current batch into a model-cost query and advances local execution.",
    examples: [
      [
        "PD prefill worker",
        "Combine prefill admission, a full-attention KV store and whole-iteration execution.",
      ],
      [
        "PD decode worker",
        "Combine KV pulling and token decoding in a pull/decode worker.",
      ],
    ],
  },
  {
    n: 4,
    name: "Model architecture",
    group: "Cost composition",
    icon: Layers,
    summary: "Assemble the model.",
    detail:
      "Wires worklets into a full model or a disaggregated section. Compiles stable CostTree sections once and exposes typed cost queries to workers.",
    examples: [
      [
        "Llama3 dense TP",
        "Compose attention and MLP worklets with tensor-parallel synchronization.",
      ],
      [
        "Qwen3 MoE",
        "Compose DP attention and expert-parallel FFN sections, including dispatch and combine.",
      ],
    ],
  },
  {
    n: 3,
    name: "Worklet",
    group: "Cost composition",
    icon: Layers,
    summary: "Compose a model module.",
    detail:
      "Groups operations into attention, MLP, or another module-level synchronization section. It derives per-rank shapes from the configuration and composes sequential and overlapping costs.",
    examples: [
      [
        "Tensor-parallel attention",
        "Compose normalization, QKV projection, attention, output projection and all-reduce.",
      ],
      [
        "Tensor-parallel MLP",
        "Compose normalization, gate/up projection, SwiGLU, down projection and all-reduce.",
      ],
    ],
  },
  {
    n: 2,
    name: "Operation",
    group: "Cost composition",
    icon: Layers,
    summary: "Give kernels meaning.",
    detail:
      "Names a semantic operation, such as an output projection. An atomic operation wraps one kernel; a compound operation composes several kernel calls.",
    examples: [
      [
        "FP8 GEMM with input quantization",
        "Compose per-token-group quantization and a dense GEMM as one operation.",
      ],
      [
        "MoE dispatch",
        "Compose inter-node and intra-node communication costs for token dispatch.",
      ],
    ],
  },
  {
    n: 1,
    name: "Kernel / Profiling",
    group: "Measured foundation",
    icon: Cpu,
    summary: "Measure kernel costs on hardware.",
    detail:
      "Python profiles real kernels on real GPUs. Rust builds an interpolating timing cache from those samples and evaluates it for each kernel shape.",
    examples: [
      [
        "Single GEMM",
        "Use matrix dimensions, precision, and GPU configuration to query a cached kernel cost.",
      ],
      [
        "RMSNorm",
        "Query normalization cost by token count for a configured hidden size and precision.",
      ],
    ],
  },
];
const orderedLayers = [...layers].reverse();
export function SevenLayers() {
  const [selected, setSelected] = useState(1);
  const detailStack = useRef(null);
  useLayoutEffect(() => {
    const stack = detailStack.current;
    const contents = [...stack.querySelectorAll("[data-example-content]")];
    const measure = () => {
      const height = Math.ceil(
        Math.max(...contents.map((node) => node.getBoundingClientRect().height)),
      );
      stack.style.setProperty("--example-content-height", `${height}px`);
    };
    const observer = new ResizeObserver(measure);
    contents.forEach((node) => observer.observe(node));
    measure();
    return () => observer.disconnect();
  }, []);
  return (
    <div className={s.stackExplorer}>
      <div className={s.stackButtons} aria-label="Simulator layers">
        {orderedLayers.map((item, index) => (
          <div key={item.n}>
            {(index === 0 || orderedLayers[index - 1].group !== item.group) && (
              <p className={s.groupLabel}>{item.group}</p>
            )}
            <button
              aria-pressed={selected === item.n}
              aria-controls={`layer-detail-${item.n}`}
              onClick={() => setSelected(item.n)}
            >
              <span>L{item.n}</span>
              <strong>{item.name}</strong>
              <ArrowUpRight size={19} aria-hidden="true" />
            </button>
          </div>
        ))}
      </div>
      <div className={s.detailStack} ref={detailStack}>
        {orderedLayers.map((layer) => {
          const Icon = layer.icon;
          return (
            <article
              className={s.layerDetail}
              key={layer.n}
              id={`layer-detail-${layer.n}`}
              data-selected={selected === layer.n}
              aria-hidden={selected !== layer.n}
              aria-live="polite"
              aria-atomic="true"
            >
              <div className={s.layerMark}>
                <div className={s.layerIdentity}>
                  <span>L{layer.n}</span>
                  <strong>{layer.name}</strong>
                </div>
                <Icon size={36} aria-hidden="true" />
              </div>
              <p className={s.groupLabel}>{layer.group}</p>
              <h3>{layer.summary}</h3>
              <p>{layer.detail}</p>
              <div className={s.examples}>
                <p className={s.groupLabel}>Examples</p>
                <ul>
                  {layer.examples.map(([name, description]) => (
                    <li key={name}>
                      <div data-example-content>
                        <strong>{name}</strong>
                        <p>{description}</p>
                      </div>
                    </li>
                  ))}
                </ul>
              </div>
            </article>
          );
        })}
      </div>
    </div>
  );
}
