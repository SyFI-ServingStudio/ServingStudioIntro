import { withCode } from "./modelData";

/* A catalog sentence with its `identifiers` set as code. */
export function Prose({ text }) {
  return withCode(text).map((part, index) =>
    part.code != null ? <code key={index}>{part.code}</code> : part.text,
  );
}
