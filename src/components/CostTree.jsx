import { useLayoutEffect, useRef, useState } from "react";
import { GitBranch, Play } from "lucide-react";
import { Tabs } from "./Tabs";
import qwenData from "../data/qwenCostTree.json";
import llamaData from "../data/llamaCostTree.json";
import overview from "../sections/UseCases.module.css";
import s from "./CostTree.module.css";

const phases = ["Build", "Evaluate"];

const models = [
  llamaData,
  {
    ...qwenData,
    model: "Qwen3-235B",
    workload: "16,384-token prefill",
    hardware: "FP8 · 4 × H200 · TP4 / EP4",
    source:
      "Recorded timing prediction · case 2. One prefill request, no cached prefix, 94 decoder layers.",
  },
];

export function CostTree() {
  const [selected, setSelected] = useState(0);
  return (
    <div className={s.models}>
      <Tabs
        label="Cost tree model"
        items={models}
        selected={selected}
        onChange={setSelected}
        className={`tabs ${overview.exampleTabs} ${overview.caseSelector} ${s.modelSelector}`}
        tabId={(_, index) => `cost-model-${index}`}
        panelId="cost-model-panel"
        renderItem={(model) => model.model}
      />
      <div
        id="cost-model-panel"
        role="tabpanel"
        aria-labelledby={`cost-model-${selected}`}
      >
        <ModelCostTree data={models[selected]} key={selected} />
      </div>
    </div>
  );
}

function ModelCostTree({ data }) {
  const [phase, setPhase] = useState(0);
  const aside = useRef(null);
  const treeTitle = useRef(null);
  const [treeHeight, setTreeHeight] = useState(null);
  useLayoutEffect(() => {
    const element = aside.current;
    const title = treeTitle.current;
    const desktop = window.matchMedia("(min-width: 1024px)");
    const measure = () => {
      const titleHeight =
        title.getBoundingClientRect().height +
        parseFloat(getComputedStyle(title).marginBottom);
      setTreeHeight(
        desktop.matches
          ? Math.ceil(element.getBoundingClientRect().height - titleHeight)
          : 650,
      );
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    observer.observe(title);
    desktop.addEventListener("change", measure);
    return () => {
      observer.disconnect();
      desktop.removeEventListener("change", measure);
    };
  }, []);
  return (
    <div className={s.layout}>
      <div className={s.treeColumn}>
        <figure className={s.figure} data-phase={phase}>
          <div className={s.treeTitle} ref={treeTitle}>
            {data.model} · recorded cost tree
            <small>Scroll within the tree to inspect every operation.</small>
          </div>
          <div
            className={s.treeViewport}
            style={
              treeHeight ? { height: treeHeight, maxHeight: treeHeight } : undefined
            }
            tabIndex={0}
            role="region"
            aria-label="Full model cost tree"
          >
            <TreeRows rows={data.rows} />
          </div>
        </figure>
      </div>
      <aside className={s.aside} ref={aside}>
        <Tabs
          label="Cost tree phase"
          items={phases}
          selected={phase}
          onChange={setPhase}
          orientation="horizontal"
          className={s.tabs}
          labelFor={(label) => label}
          renderItem={(label, index) => {
            const Icon = index === 0 ? GitBranch : Play;
            return (
              <>
                <Icon size={30} strokeWidth={1.5} aria-hidden="true" />
                <span>{label}</span>
              </>
            );
          }}
          tabId={(_, i) => `cost-phase-${i}`}
          panelId="cost-phase-panel"
        />
        <div
          id="cost-phase-panel"
          role="tabpanel"
          aria-labelledby={`cost-phase-${phase}`}
          tabIndex={0}
          className={s.panel}
        >
          {phase === 0 ? (
            <p>
              The model’s operations are assembled into a cost tree once. Sum
              combines sequential work, Max combines parallel ranks, and Scale
              represents repeated layers.
            </p>
          ) : (
            <p>
              Each leaf queries its cache for the current input shape. The evaluator
              then combines these costs using the original tree, without rebuilding
              its structure.
            </p>
          )}
        </div>
        <div className={s.result}>
          <p>Predicted iteration time</p>
          <strong>
            {data.totalMs.toFixed(3)} <span>ms</span>
          </strong>
          <p>
            {data.workload}
            <br />
            {data.hardware}
          </p>
        </div>
        <p className={s.source}>{data.source}</p>
        <p className={s.sidenote} role="note">
          <span aria-hidden="true">* </span>
          This is the recorded model tree, from embedding through the decoder layers
          to the LM head. Operations shown as single leaves include their underlying
          kernel costs.
          {data.model === "Qwen3-235B" &&
            " Each multi-rank Max shows its slowest child once."}{" "}
          Timings are per invocation; the original tree determines the iteration
          total.
        </p>
      </aside>
    </div>
  );
}

const backendLabels = {
  deepgemm: "DeepGEMM",
  fa3: "FA3",
  flashinfer: "FlashInfer",
  flashinfer_trtllm: "FlashInfer TRT-LLM",
  nccl: "NCCL",
  nvshmem: "NVSHMEM",
  triton: "Triton",
  vllm_cuda: "vLLM CUDA",
  torch: "PyTorch",
};

function TreeRows({ rows }) {
  return (
    <ol className={s.tree}>
      {rows.map((node, index) => (
        <li
          key={node.id}
          className={node.kind === "Leaf" ? s.leaf : s.parent}
          data-node={node.id}
          style={{ "--depth": node.depth, "--order": index }}
        >
          <div className={s.row}>
            <b data-kind={node.kind}>{node.kind}</b>
            <strong>
              {node.name === "Expert rank 2" ? "Expert compute" : node.name}
              {node.parallelCount && (
                <span className={s.parallelNote}>
                  {node.parallelCount} parallel ranks · slowest rank shown
                </span>
              )}
            </strong>
            {node.kind === "Leaf" && (
              <div className={s.nodeMetrics}>
                <span className={s.time}>{node.ms.toFixed(3)} ms</span>
                <span className={s.backend}>
                  {node.backends
                    ?.map((name) => backendLabels[name] || name)
                    .join(" · ")}
                </span>
              </div>
            )}
          </div>
        </li>
      ))}
    </ol>
  );
}
