import { CircleCheck, CircleDashed } from "lucide-react";
import s from "./Models.module.css";

/* One configuration's state in a line: a check when it runs, a dashed mark
   when `blocked`, and what to say about it. */
export function StatusLine({ blocked, children }) {
  return (
    <div className={s.memberStatus} data-state={blocked ? "blocked" : "ready"}>
      {blocked ? (
        <CircleDashed size={16} aria-hidden="true" />
      ) : (
        <CircleCheck size={16} aria-hidden="true" />
      )}
      <p>{children}</p>
    </div>
  );
}

/* Said when a link names values no configuration has (memberFromQuery's
   `unmatched`). */
export function UnmatchedNotice() {
  return (
    <p className={s.notice} role="status">
      The link named values no configuration of this deployment has, so the closest
      one is shown.
    </p>
  );
}
