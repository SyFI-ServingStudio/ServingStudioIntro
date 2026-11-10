/* The text this build writes into a page's HTML for readers that run no
   JavaScript (scripts/agent-content.mjs), and how to take it out again. */

import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

/* Comments around everything agent-content.mjs writes into a page, so a
   plugin that reuses a page as a template (blog-content.mjs, for each post) can
   take it out again. */
export const INJECTED_START = "<!-- agent-content -->";
export const INJECTED_END = "<!-- /agent-content -->";
export const removeInjected = (html) =>
  html.replace(new RegExp(`${INJECTED_START}[\\s\\S]*?${INJECTED_END}`, "g"), "");

/* The Markdown as HTML inside #root. The style keeps it out of sight in any
   browser that runs scripts, so the page looks as it always did while the
   app loads; React then replaces #root's children. */
export function injectStaticContent(html, { markdown, markdownUrl }) {
  const body = renderToStaticMarkup(
    createElement(Markdown, { remarkPlugins: [remarkGfm] }, markdown),
  );
  return html
    .replace(
      "</head>",
      `  ${INJECTED_START}<link rel="alternate" type="text/markdown" href="${markdownUrl}" />\n    <style>@media (scripting: enabled) { .static-page { display: none; } }</style>${INJECTED_END}\n  </head>`,
    )
    .replace(
      '<div id="root"></div>',
      `<div id="root">${INJECTED_START}<main class="static-page wrap">${body}</main>${INJECTED_END}</div>`,
    );
}
