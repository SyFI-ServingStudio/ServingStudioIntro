import test from "node:test";
import assert from "node:assert/strict";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { htmlToMarkdown, prerenderPages } from "./prerender-pages.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const origin = "https://example.org";

test("headings, paragraphs, lists and links become Markdown", () => {
  const markdown = htmlToMarkdown(
    "<header><h1>Title<br/>here</h1><p>One <strong>bold</strong> and <code>code</code> &amp; more.</p></header>" +
      '<ul><li><a href="/features.html">Features</a></li><li><a href="#top">Top</a></li></ul>' +
      "<p>After</p>",
    { origin },
  );
  assert.equal(
    markdown,
    "# Title here\n\nOne **bold** and `code` & more.\n\n- [Features](https://example.org/features.html)\n- Top\n\nAfter",
  );
});

test("what a reader cannot read is left out, labels stand in for hidden text", () => {
  const markdown = htmlToMarkdown(
    '<nav><a href="/">Home</a></nav><a class="skip-link" href="#main">Skip</a>' +
      '<p>Shown<svg><text>chart</text></svg><button>Copy</button><span aria-hidden="true">icon</span></p>' +
      '<div hidden=""><p>Hidden</p></div>' +
      '<p aria-label="Typed question"><span aria-hidden="true">Typed</span><span aria-hidden="true"> question</span></p>' +
      '<div aria-hidden="true" inert=""><p>Revealed later</p></div>',
    { origin },
  );
  assert.equal(markdown, "Shown\n\nTyped question\n\nRevealed later");
});

test("side-by-side inline elements are spaced, tables keep their rows", () => {
  const markdown = htmlToMarkdown(
    "<p><span>218.5</span><span>tok/s</span></p>" +
      "<table><thead><tr><th>Depth</th><th>tok/s</th></tr></thead><tbody><tr><td>3</td><td>218.5</td></tr></tbody></table>",
    { origin },
  );
  assert.equal(
    markdown,
    "218.5 tok/s\n\n| Depth | tok/s |\n| --- | --- |\n| 3 | 218.5 |",
  );
});

test("the real App renders each static page's copy", async () => {
  const pages = await prerenderPages({
    root,
    origin,
    files: ["", "features.html", "architecture.html", "blog.html"],
  });
  const markdown = Object.fromEntries(
    Object.entries(pages).map(([file, html]) => [
      file,
      htmlToMarkdown(html, { origin }),
    ]),
  );
  assert.match(
    markdown[""],
    /^# Bridge Simulation and Agents for Speed-of-Light Serving$/m,
  );
  assert.match(
    markdown["features.html"],
    /^# Powerful simulation\. Zero manual coding\.$/m,
  );
  assert.match(
    markdown["architecture.html"],
    /^# How kernel measurements become serving predictions\.$/m,
  );
  assert.match(markdown["blog.html"], /^### Introducing ServingStudio/m);
  assert.equal(globalThis.window, undefined);
});
