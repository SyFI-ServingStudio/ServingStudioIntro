/* The copy of the site's static pages (Overview, Features, Architecture and
   the Blog index) as Markdown, for readers that run no JavaScript.

   The pages' words live in their JSX, so this renders the real App at build
   time, once per page, with React's server renderer inside a Vite SSR server
   (the same JSX and CSS-module transforms as the browser build), and turns
   the HTML into Markdown. Nothing here writes copy of its own: a sentence
   changed in a component changes here on the next build.

   What a reader cannot read is left out: SVG, images, buttons, anything
   `aria-hidden` (but see isSkipped) or `hidden`, and the navigation (llms.txt lists the pages).
   An element whose own text is all hidden but that carries an `aria-label`
   (a typed-out question, a chart's summary) contributes that label. */

import { createElement } from "react";
import { prerenderToNodeStream } from "react-dom/static";
import { createServer } from "vite";
import react from "@vitejs/plugin-react";
import { blogContent } from "./blog-content.mjs";

/* ---------- HTML to Markdown ---------- */

const VOID_ELEMENTS = new Set([
  "area",
  "base",
  "br",
  "col",
  "embed",
  "hr",
  "img",
  "input",
  "link",
  "meta",
  "source",
  "track",
  "wbr",
]);
const SKIPPED_ELEMENTS = new Set([
  "svg",
  "script",
  "style",
  "noscript",
  "template",
  "button",
  "select",
  "textarea",
  "nav",
  "img",
  "picture",
  "video",
  "canvas",
]);
const BLOCK_ELEMENTS = new Set([
  "address",
  "article",
  "aside",
  "blockquote",
  "caption",
  "dd",
  "details",
  "div",
  "dl",
  "dt",
  "fieldset",
  "figcaption",
  "figure",
  "footer",
  "form",
  "h1",
  "h2",
  "h3",
  "h4",
  "h5",
  "h6",
  "header",
  "hr",
  "li",
  "main",
  "ol",
  "p",
  "pre",
  "section",
  "summary",
  "table",
  "ul",
]);

const decodeEntities = (value) =>
  value.replace(
    /&(#x[0-9a-f]+|#[0-9]+|amp|lt|gt|quot|apos|nbsp);/gi,
    (_, entity) => {
      const lower = entity.toLowerCase();
      if (lower.startsWith("#x"))
        return String.fromCodePoint(parseInt(lower.slice(2), 16));
      if (lower.startsWith("#"))
        return String.fromCodePoint(Number(lower.slice(1)));
      return { amp: "&", lt: "<", gt: ">", quot: '"', apos: "'", nbsp: " " }[lower];
    },
  );

/* A tree of { tag, attributes, children } and strings, from the well-formed
   HTML React's server renderer writes (every tag closed, attributes quoted). */
export function parseHtml(html) {
  const root = { tag: "#root", attributes: {}, children: [] };
  const stack = [root];
  const tokens =
    /<!--[\s\S]*?-->|<\/([a-zA-Z0-9-]+)\s*>|<([a-zA-Z0-9-]+)((?:\s+[^\s=>/]+(?:="[^"]*")?)*)\s*(\/?)>|([^<]+)/g;
  for (const match of html.matchAll(tokens)) {
    const [, closingTag, openingTag, attributeText, selfClosing, textValue] = match;
    const current = stack[stack.length - 1];
    if (textValue != null) current.children.push(decodeEntities(textValue));
    else if (closingTag) {
      const tag = closingTag.toLowerCase();
      const index = stack.findLastIndex((node) => node.tag === tag);
      if (index > 0) stack.length = index;
    } else if (openingTag) {
      const attributes = {};
      for (const [, name, value] of (attributeText ?? "").matchAll(
        /([^\s=]+)(?:="([^"]*)")?/g,
      ))
        attributes[name.toLowerCase()] = decodeEntities(value ?? "");
      const node = { tag: openingTag.toLowerCase(), attributes, children: [] };
      current.children.push(node);
      if (!selfClosing && !VOID_ELEMENTS.has(node.tag)) stack.push(node);
    }
  }
  return root;
}

// Plain text placed in Markdown: what would start markup is escaped.
const escapeMarkdown = (value) => value.replace(/([\\`*_[\]<>|])/g, "\\$1");

/* `aria-hidden` marks decoration and duplicates, except where it comes with
   `inert`: that is a staged reveal (UseCases.jsx `reveal`), copy a visitor
   sees once the replay reaches it. */
const isSkipped = (node) =>
  SKIPPED_ELEMENTS.has(node.tag) ||
  (node.attributes["aria-hidden"] === "true" && !("inert" in node.attributes)) ||
  "hidden" in node.attributes ||
  /\bskip-link\b/.test(node.attributes.class ?? "");

// Whether any text under `node` survives isSkipped.
const hasVisibleText = (node) =>
  typeof node === "string"
    ? node.trim() !== ""
    : !isSkipped(node) && node.children.some(hasVisibleText);

const containsBlock = (node) =>
  typeof node !== "string" &&
  node.children.some(
    (child) =>
      typeof child !== "string" &&
      (BLOCK_ELEMENTS.has(child.tag) || containsBlock(child)),
  );

/* Markdown for an HTML fragment. Links are made absolute against `origin`;
   in-page anchors keep only their text. */
export function htmlToMarkdown(html, { origin }) {
  const blocks = []; // { kind: "text" | "item" | "table", value }
  let buffer = "";
  let prefix = "";
  let kind = "text";

  const flush = () => {
    const value = buffer.replace(/\s+/g, " ").trim();
    if (value) blocks.push({ kind, value: `${prefix}${value}` });
    buffer = "";
    prefix = "";
    kind = "text";
  };
  // The inline Markdown of a node's children, without touching the blocks.
  const inline = (node) => {
    const saved = buffer;
    buffer = "";
    visitChildren(node);
    const value = buffer.replace(/\s+/g, " ").trim();
    buffer = saved;
    return value;
  };
  const absolute = (href) => {
    if (!href || href.startsWith("#") || /^(?:javascript|mailto):/i.test(href))
      return null;
    return new URL(href, `${origin}/`).href;
  };

  function visitTable(table) {
    const rows = [];
    const collect = (node) => {
      for (const child of node.children) {
        if (typeof child === "string" || isSkipped(child)) continue;
        if (child.tag === "tr")
          rows.push(
            child.children
              .filter(
                (cell) =>
                  typeof cell !== "string" && ["td", "th"].includes(cell.tag),
              )
              .map((cell) => (isSkipped(cell) ? "" : inline(cell))),
          );
        else collect(child);
      }
    };
    collect(table);
    if (!rows.length) return;
    const width = Math.max(...rows.map((row) => row.length));
    const line = (row) =>
      `| ${Array.from({ length: width }, (_, index) => row[index] ?? "").join(" | ")} |`;
    blocks.push({
      kind: "table",
      value: [
        line(rows[0]),
        line(rows[0].map(() => "---")),
        ...rows.slice(1).map(line),
      ].join("\n"),
    });
  }

  /* Two inline elements side by side, with no text between them, are
     usually laid out apart by CSS ("218.5" and "tok/s", a label and its
     value), so a space separates them. */
  function visitChildren(node) {
    node.children.forEach((child, index) => {
      const previous = node.children[index - 1];
      if (
        typeof child !== "string" &&
        previous != null &&
        typeof previous !== "string"
      )
        buffer += " ";
      visit(child);
    });
  }

  function visit(node) {
    if (typeof node === "string") {
      buffer += escapeMarkdown(node);
      return;
    }
    if (isSkipped(node)) return;
    const { tag, attributes } = node;
    // Hidden text with a spoken label: the label stands for it.
    const label = attributes["aria-label"]?.trim();
    if (label && tag !== "a" && !hasVisibleText(node)) {
      if (BLOCK_ELEMENTS.has(tag)) {
        flush();
        buffer = escapeMarkdown(label);
        flush();
      } else buffer += ` ${escapeMarkdown(label)} `;
      return;
    }
    if (tag === "br") buffer += " ";
    else if (tag === "table") {
      flush();
      visitTable(node);
    } else if (tag === "a" && !containsBlock(node)) {
      const label = inline(node) || attributes["aria-label"] || "";
      const href = absolute(attributes.href);
      if (label) buffer += href ? ` [${label}](${href}) ` : ` ${label} `;
    } else if (tag === "code" && !containsBlock(node)) {
      const value = node.children
        .filter((child) => typeof child === "string")
        .join("");
      if (value.trim()) buffer += `\`${value.replace(/`/g, "")}\``;
    } else if ((tag === "strong" || tag === "b") && !containsBlock(node)) {
      const value = inline(node);
      if (value) buffer += `**${value}**`;
    } else if (/^h[1-6]$/.test(tag)) {
      flush();
      prefix = `${"#".repeat(Number(tag[1]))} `;
      visitChildren(node);
      flush();
    } else if (tag === "li") {
      flush();
      kind = "item";
      prefix = "- ";
      visitChildren(node);
      flush();
    } else if (BLOCK_ELEMENTS.has(tag)) {
      flush();
      visitChildren(node);
      flush();
      // A link around whole blocks (a card): its target follows them.
    } else if (tag === "a") {
      flush();
      visitChildren(node);
      flush();
      const href = absolute(attributes.href);
      if (href) blocks.push({ kind: "text", value: `[${href}](${href})` });
    } else visitChildren(node);
  }

  visitChildren(parseHtml(html));
  flush();
  return blocks
    .map((block, index) =>
      index === 0
        ? block.value
        : `${block.kind === "item" && blocks[index - 1].kind === "item" ? "\n" : "\n\n"}${block.value}`,
    )
    .join("");
}

/* ---------- Rendering the pages ---------- */

/* Each page's App markup, keyed by its file ("" for the Overview). The App
   reads the page from window.location, so a stand-in location is set while
   each page renders; nothing else of `window` is read during a render.
   prerender waits for the lazily loaded Blog page. */
export async function prerenderPages({ root, origin, files }) {
  const server = await createServer({
    root,
    configFile: false,
    plugins: [react(), blogContent()],
    appType: "custom",
    logLevel: "error",
    optimizeDeps: { noDiscovery: true, include: [] },
    server: { middlewareMode: true, hmr: false, ws: false },
  });
  const rendered = {};
  try {
    const { default: App } = await server.ssrLoadModule("/src/App.jsx");
    for (const file of files) {
      globalThis.window = { location: new URL(`${origin}/${file}`) };
      try {
        const { prelude } = await prerenderToNodeStream(createElement(App));
        let html = "";
        for await (const chunk of prelude) html += chunk;
        rendered[file] = html;
      } finally {
        delete globalThis.window;
      }
    }
  } finally {
    await server.close();
  }
  return rendered;
}
