import test from "node:test";
import assert from "node:assert/strict";
import { renderSitemap } from "./sitemap.mjs";

test("renders absolute URLs, optional lastmod, and escapes XML", () => {
  const xml = renderSitemap([
    { loc: "https://example.org/site/" },
    { loc: "https://example.org/site/blog/post/?a=1&b=2", lastmod: "2026-09-24" },
  ]);
  assert.match(
    xml,
    /^<\?xml version="1.0" encoding="UTF-8"\?>\n<urlset xmlns="http:\/\/www.sitemaps.org\/schemas\/sitemap\/0.9">/,
  );
  assert.match(
    xml,
    /<url>\n {4}<loc>https:\/\/example.org\/site\/<\/loc>\n {2}<\/url>/,
  );
  assert.match(
    xml,
    /<loc>https:\/\/example.org\/site\/blog\/post\/\?a=1&amp;b=2<\/loc>\n {4}<lastmod>2026-09-24<\/lastmod>/,
  );
  assert.equal(xml.match(/<url>/g).length, 2);
});

test("the sitemap lists every Markdown file already in the bundle", async () => {
  const { sitemap } = await import("./sitemap.mjs");
  const plugin = sitemap({ origin: "https://example.org", pages: [""] });
  plugin.configResolved({ base: "/", root: process.cwd() });
  let emitted;
  plugin.generateBundle.call(
    { emitFile: ({ source }) => (emitted = source) },
    {},
    { "kernels/x.md": {}, "index.md": {}, "index.html": {} },
  );
  assert.match(emitted, /<loc>https:\/\/example\.org\/index\.md<\/loc>/);
  assert.match(emitted, /<loc>https:\/\/example\.org\/kernels\/x\.md<\/loc>/);
  assert.doesNotMatch(emitted, /index\.html/);
});
