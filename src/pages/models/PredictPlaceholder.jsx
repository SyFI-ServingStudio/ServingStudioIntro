import { Plus, Timer } from "lucide-react";
import s from "./Models.module.css";

/* The next step, shown but switched off: a batch of requests for this
   parameter set, and a predicted time on every node of its tree. The public
   service does not predict yet, so nothing here sends anything. */
export function PredictPlaceholder() {
  return (
    <section className={s.predict} aria-labelledby="predict-title">
      <div className={s.predictHead}>
        <h2 id="predict-title">Predict iteration time</h2>
        <span className={s.soon}>Not available yet</span>
      </div>
      <p>
        Add the requests in a batch and the simulator will time every node of this
        tree for them.
      </p>
      <fieldset className={s.predictForm} disabled aria-describedby="predict-note">
        <legend className={s.visuallyHidden}>Requests in the batch</legend>
        <label>
          <span>Requests</span>
          <input type="number" inputMode="numeric" />
        </label>
        <label>
          <span>Prompt tokens each</span>
          <input type="number" inputMode="numeric" />
        </label>
        <label>
          <span>Context length each, for decode</span>
          <input type="number" inputMode="numeric" />
        </label>
        <div className={s.predictActions}>
          <button type="button">
            <Plus size={16} aria-hidden="true" />
            Add requests
          </button>
          <button type="button" className={s.primary}>
            <Timer size={16} aria-hidden="true" />
            Predict time
          </button>
        </div>
      </fieldset>
      <p id="predict-note" className={s.predictNote}>
        The data service answers the tree&apos;s structure only for now. Predictions
        will read measured rows alone, and a mixture-of-experts model will need its
        expert routing named.
      </p>
    </section>
  );
}
