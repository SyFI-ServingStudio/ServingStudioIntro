/* Small helpers for the Markdown the build writes for crawlers and agents
   (scripts/agent-content.mjs, scripts/agent-detail-pages.mjs). */

/* Text placed in Markdown. The API's prose is written for Markdown readers
   (`code` spans included), so only what would open a link or an HTML tag is
   escaped, and nothing inside a code span, where a backslash would show. */
export const text = (value) =>
  String(value)
    .split(/(`[^`]*`)/)
    .map((part, index) => (index % 2 ? part : part.replace(/([\\[\]<>])/g, "\\$1")))
    .join("");
export const code = (value) => `\`${String(value).replace(/`/g, "")}\``;
export const join = (values) => values.join(", ");
export const plural = (count, word) =>
  `${count.toLocaleString("en-US")} ${word}${count === 1 ? "" : "s"}`;
// An axis value as the pages show it: numbers grouped, booleans as words.
export const axisValue = (value) =>
  typeof value === "number"
    ? value.toLocaleString("en-US")
    : typeof value === "boolean"
      ? value
        ? "on"
        : "off"
      : String(value);
export const axesText = (axes) =>
  axes
    .map((axis) => `${code(axis.name)} ${join(axis.values.map(axisValue))}`)
    .join("; ");

/* Whether a Models member can be timed: the simulator built it and every
   kernel it calls is measured (modelData.js `predictable`). */
export const predictable = (member) =>
  !member.error && member.missing != null && !Object.keys(member.missing).length;
