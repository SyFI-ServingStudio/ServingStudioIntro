import { useState } from "react";
import { Boxes, Cpu, Shuffle, MessagesSquare, Network } from "lucide-react";
import { WorkerDetail } from "./WorkerDetail";
import { Tabs } from "./Tabs";
import s from "./ServingHierarchy.module.css";

const levels = [
  {
    name: "Worker",
    icon: Cpu,
    title: "A worker advances its assigned requests.",
    body: "A worker tracks local request and KV state, forms batches, and queries the model’s cost tree to determine when work finishes. One worker can span one GPU or a parallel group.",
  },
  {
    name: "Pool",
    icon: Boxes,
    title: "A pool places work across workers.",
    body: "A pool groups worker replicas with the same serving role. Its placement policy selects a worker for incoming work; each worker manages its own execution and state.",
  },
  {
    name: "Orchestrator",
    icon: Network,
    title: "An orchestrator coordinates pools.",
    body: "The orchestrator routes requests between pools and coordinates their handoffs. In this prefill/decode example, prefill completes before the request and KV state move to a decode worker.",
  },
];

function WorkerCard({ name, decode = false }) {
  return (
    <div className={s.workerCard}>
      <div className={s.workerHead}>
        <Cpu size={24} strokeWidth={1.5} aria-hidden="true" />
        <strong>{name}</strong>
      </div>
      <div className={s.workerSummary}>
        <span>{decode ? "KV pull" : "Admission"}</span>
        <span>KV</span>
        <span>Execution</span>
      </div>
    </div>
  );
}

function Pool({ decode = false, compact = false }) {
  const names = decode ? ["E", "F", "G", "H"] : ["A", "B", "C", "D"];
  return (
    <div className={compact ? s.pool : s.poolContents}>
      {compact && (
        <div className={s.poolHead}>
          <Boxes size={26} strokeWidth={1.5} aria-hidden="true" />
          <strong>{decode ? "Decode pool" : "Prefill pool"}</strong>
        </div>
      )}
      <div className={s.workers}>
        <div className={s.workBalancer}>
          <Shuffle size={24} strokeWidth={1.5} aria-hidden="true" />
          <strong>Work balancer</strong>
        </div>
        {names.map((name) => (
          <WorkerCard key={name} name={`Worker ${name}`} decode={decode} />
        ))}
      </div>
    </div>
  );
}

export function ServingHierarchy() {
  const [level, setLevel] = useState(0);
  const current = levels[level];
  return (
    <div className={s.component}>
      <div className={s.controls}>
        <Tabs
          label="Serving hierarchy"
          items={levels}
          selected={level}
          onChange={setLevel}
          className={s.levelTabs}
          tabId={(_, index) => `hierarchy-level-${index}`}
          panelId="hierarchy-panel"
          renderItem={({ name, icon: Icon }) => (
            <>
              <Icon size={24} strokeWidth={1.5} aria-hidden="true" />
              <span>{name}</span>
            </>
          )}
        />
      </div>
      <div
        id="hierarchy-panel"
        role="tabpanel"
        aria-labelledby={`hierarchy-level-${level}`}
      >
        <div className={s.explanations}>
          {levels.map((item, index) => (
            <div
              className={s.explanation}
              key={item.name}
              data-active={level === index}
              aria-hidden={level !== index}
            >
              <h3>{item.title}</h3>
              <p>{item.body}</p>
            </div>
          ))}
        </div>
        <div className={s.scene} data-level={level}>
          <div className={s.sceneHeader}>
            <current.icon size={28} strokeWidth={1.5} aria-hidden="true" />
            <strong>{["Worker A", "Prefill pool", "Orchestrator"][level]}</strong>
          </div>
          <div className={s.views}>
            <div
              className={s.view}
              data-active={level === 0}
              aria-hidden={level !== 0}
            >
              <WorkerDetail />
            </div>
            <div
              className={s.view}
              data-active={level === 1}
              aria-hidden={level !== 1}
            >
              <Pool />
            </div>
            <div
              className={s.view}
              data-active={level === 2}
              aria-hidden={level !== 2}
            >
              <div className={s.pools}>
                <div className={`${s.workBalancer} ${s.messageInterface}`}>
                  <MessagesSquare size={24} strokeWidth={1.5} aria-hidden="true" />
                  <strong>Message passing interface</strong>
                </div>
                <Pool compact />
                <Pool compact decode />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
