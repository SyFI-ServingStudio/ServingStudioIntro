import { useEffect, useState } from "react";
import {
  ArrowRight,
  SlidersHorizontal,
  Split,
  Layers,
  Database,
  FileCode2,
  FolderOpen,
} from "lucide-react";
import s from "./LauncherDemo.module.css";

// Deliberately focused excerpts, not standalone launcher presets.
const examples = [
  {
    title: "Sweep request rate",
    icon: SlidersHorizontal,
    detail: "Compare light and heavy traffic.",
    before: [
      "deployment: pd",
      "",
      "workload:",
      "  request_rate: 8  # requests / second",
    ],
    after: ["deployment: pd", "", "workload:", "  request_rate: ${rate}"],
    added: ["", "sweep:", "  rate: [8, 16, 32]"],
    note: "One parameter, three arrival rates.",
  },
  {
    title: "Allocate 8 GPUs",
    icon: Split,
    detail: "Vary the prefill/decode ratio.",
    before: [
      "# One GPU per replica",
      "pools:",
      "  prefill:",
      "    groups: [{replicas: 1}]",
      "  decode:",
      "    groups: [{replicas: 7}]",
    ],
    after: [
      "# One GPU per replica",
      "pools:",
      "  prefill:",
      '    groups: [{replicas: "${prefill_replicas}"}]',
      "  decode:",
      '    groups: [{replicas: "${decode_replicas}"}]',
    ],
    added: [
      "",
      "sweep:",
      "  prefill_replicas: [1, 2, 4, 8]",
      "",
      "derived:",
      '  decode_replicas: "8 - prefill_replicas"',
      "",
      "constraints:",
      '  - "decode_replicas >= 1"',
    ],
    note: "Keep eight GPUs in total. Exclude splits with no decode worker.",
  },
  {
    title: "Sweep backends",
    icon: Layers,
    detail: "Compare implementations of one kernel.",
    before: ["backends:", "  prefill/pd.attn_block.attn.prefill:", "    [fa2]"],
    after: [
      "backends:",
      "  prefill/pd.attn_block.attn.prefill:",
      "    ${attention_backend}",
    ],
    added: [
      "",
      "sweep:",
      "  attention_backend:",
      "    fa2: [fa2]",
      "    fa3: [fa3]",
      "    best: [fa2, fa3]",
    ],
    note: "Compare FA2, FA3, or the faster predicted backend for each input shape.",
  },
  {
    title: "Pair KV budgets",
    icon: Database,
    detail: "Change related parameters together.",
    before: [
      "pools:",
      "  prefill:",
      "    groups:",
      "      - worker:",
      "          type: pd_prefill",
      "          attn_gpu_memory_gb: 40",
      "  decode:",
      "    groups:",
      "      - worker:",
      "          type: pd_decode",
      "          attn_gpu_memory_gb: 80",
    ],
    after: [
      "pools:",
      "  prefill:",
      "    groups:",
      "      - worker:",
      "          type: pd_prefill",
      "          attn_gpu_memory_gb: ${prefill_mem_gb}",
      "  decode:",
      "    groups:",
      "      - worker:",
      "          type: pd_decode",
      "          attn_gpu_memory_gb: ${decode_mem_gb}",
    ],
    added: [
      "",
      "compound:",
      "  kv_budget:",
      "    small: {prefill_mem_gb: 40, decode_mem_gb: 80}",
      "    large: {prefill_mem_gb: 80, decode_mem_gb: 120}",
    ],
    note: "Each row sets both workers’ KV budgets together, in GB.",
  },
  {
    title: "Organize results",
    icon: FolderOpen,
    detail: "Give every experiment its own directory.",
    before: [
      "io:",
      "  log_dir: logs/llama3_pd",
      "",
      "workload:",
      "  request_rate: 8",
    ],
    after: [
      "io:",
      "  log_dir: logs/llama3_pd/rate_{rate}",
      "",
      "workload:",
      "  request_rate: ${rate}",
    ],
    added: [
      "",
      "sweep:",
      "  rate: [8, 16, 32]",
      "",
      "# logs/llama3_pd/rate_8",
      "# logs/llama3_pd/rate_16",
      "# logs/llama3_pd/rate_32",
    ],
    note: "The launcher substitutes each rate into the output path.",
  },
];

function Syntax({ text }) {
  if (text.trimStart().startsWith("#"))
    return <span className={s.comment}>{text}</span>;
  const match = text.match(/^(\s*(?:- )?)([^:]+:)(.*)$/);
  if (!match) return <span className={s.value}>{text || " "}</span>;
  return (
    <>
      {match[1]}
      <span className={s.key}>{match[2]}</span>
      <span className={s.value}>{match[3]}</span>
    </>
  );
}

function Snippet({ example }) {
  const [rows, setRows] = useState(example.before);
  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      setRows([...example.after, ...example.added]);
      return;
    }
    let timer;
    let count = 0;
    const advance = () => {
      // Rebuild from immutable source each step; never append to previous state.
      setRows([...example.after, ...example.added.slice(0, count)]);
      if (count < example.added.length) {
        count += 1;
        timer = window.setTimeout(advance, 160);
      }
    };
    timer = window.setTimeout(advance, 400);
    return () => window.clearTimeout(timer);
  }, [example]);

  return (
    <pre
      className={s.code}
      aria-label={`${example.title}: illustrative YAML excerpt`}
    >
      <code>
        {rows.map((text, index) => (
          <span
            key={index}
            className={`${s.line} ${index >= example.before.length ? s.inserted : ""}`}
          >
            <span className={s.lineInner}>
              <span
                key={text}
                className={text !== example.before[index] ? s.edited : undefined}
              >
                <Syntax text={text} />
              </span>
            </span>
          </span>
        ))}
      </code>
    </pre>
  );
}

export function LauncherDemo() {
  const [selection, setSelection] = useState({ index: 0, replay: 0 });
  const example = examples[selection.index];
  return (
    <div className={s.demo}>
      <div className={s.options} aria-label="Launcher demonstrations">
        {examples.map(({ title, icon: Icon, detail }, index) => (
          <button
            key={title}
            type="button"
            className={s.option}
            aria-pressed={selection.index === index}
            aria-controls="launcher-example"
            onClick={() =>
              setSelection(({ replay }) => ({ index, replay: replay + 1 }))
            }
          >
            <span className={s.optionTitle}>
              <Icon size={24} strokeWidth={1.5} aria-hidden="true" />
              <span>{title}</span>
              <ArrowRight className={s.arrow} size={20} aria-hidden="true" />
            </span>
            <span className={s.detail}>{detail}</span>
          </button>
        ))}
      </div>
      <div className={s.editor} id="launcher-example">
        <div className={s.toolbar}>
          <FileCode2 size={24} strokeWidth={1.5} aria-hidden="true" />
          <span>Launcher YAML</span>
          <span className={s.caption}>Illustrative excerpt</span>
        </div>
        <div className={s.stage}>
          {/* Full excerpts reserve the same maximum height through every animation. */}
          {examples.map((item) => (
            <div
              key={item.title}
              className={`${s.panel} ${s.reference}`}
              aria-hidden="true"
            >
              <pre className={s.code}>
                <code>
                  {[...item.after, ...item.added].map((text, index) => (
                    <span key={index} className={s.staticLine}>
                      <Syntax text={text} />
                    </span>
                  ))}
                </code>
              </pre>
              <p className={s.note}>{item.note}</p>
            </div>
          ))}
          <div className={s.panel} data-active-example="true">
            <Snippet
              key={`${selection.index}-${selection.replay}`}
              example={example}
            />
            <p className={s.note}>{example.note}</p>
          </div>
        </div>
      </div>
    </div>
  );
}
