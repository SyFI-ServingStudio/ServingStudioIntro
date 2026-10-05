import { Fragment } from "react";
import s from "./Models.module.css";

/* Pieces the cost tree, its kernel view and Live predict share. */

/* A time being computed: a sweeping bar where the number will be. */
export function Pending({ className }) {
  return <span className={`${s.pending} ${className ?? ""}`} aria-hidden="true" />;
}

// A dotted name may break after a dot on a narrow screen, not mid-word.
export const breakable = (name) =>
  name.includes(".")
    ? name.split(".").map((part, index) => (
        <Fragment key={index}>
          {index > 0 && (
            <>
              .<wbr />
            </>
          )}
          {part}
        </Fragment>
      ))
    : name;
