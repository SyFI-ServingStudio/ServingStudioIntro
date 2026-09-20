import s from "./WorkerDetail.module.css";

function Node({ name, note, span = false }) {
  return (
    <div
      data-tree-parent={span || undefined}
      className={span ? `${s.node} ${s.spanNode}` : s.node}
    >
      <strong>{name}</strong>
      {note ? <p>{note}</p> : null}
    </div>
  );
}

export function WorkerDetail() {
  return (
    <div className={s.workerScene}>
      <div className={s.scope}>
        <Node
          span
          name="Shell"
          note="Sets the execution cadence by forming a batch, building its input, and completing the iteration."
        />
        <div className={s.admissionLink} aria-hidden="true" />
        <div data-tree-children className={s.peers} style={{ "--n": 3 }}>
          <Node
            name="Admission"
            note="Chooses which pending request starts next and manages its lifecycle."
          />
          <Node
            name="KV"
            note="Owns cache capacity and each request’s residency, from reservation to release."
          />
          <Node
            name="Execution"
            note="Turns admitted work and KV state into the model input for this iteration."
          />
        </div>
        <div className={s.costRow} data-shell-cost>
          <Node
            name="Cost Tree"
            note="Returns the predicted cost of the model input, which sets the next compute completion time."
          />
        </div>
      </div>
    </div>
  );
}
