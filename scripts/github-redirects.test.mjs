import test from "node:test";
import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { buildRedirects, pagePath } from "./build-github-redirects.mjs";

test("serves a directory index at the directory", () => {
  assert.equal(pagePath("index.html"), "");
  assert.equal(pagePath("blog/post/index.html"), "blog/post/");
  assert.equal(pagePath("features.html"), "features.html");
});

test("writes a redirect for every page and a catch-all 404", () => {
  const root = mkdtempSync(join(tmpdir(), "redirects-"));
  const site = join(root, "dist");
  const out = join(root, "dist-redirect");
  mkdirSync(join(site, "blog/post"), { recursive: true });
  mkdirSync(join(site, "assets"));
  writeFileSync(join(site, "index.html"), "");
  writeFileSync(join(site, "features.html"), "");
  writeFileSync(join(site, "blog/post/index.html"), "");
  writeFileSync(join(site, "assets/main.js"), "");

  assert.deepEqual(buildRedirects(site, out).sort(), [
    "blog/post/index.html",
    "features.html",
    "index.html",
  ]);
  const post = readFileSync(join(out, "blog/post/index.html"), "utf8");
  const target = "https://servingstudio.cs.washington.edu/blog/post/";
  assert.match(post, new RegExp(`<link rel="canonical" href="${target}">`));
  assert.match(post, new RegExp(`content="0; url=${target}"`));
  assert.match(
    post,
    new RegExp(`location.replace\\("${target}" \\+ location.search`),
  );
  assert.match(
    readFileSync(join(out, "404.html"), "utf8"),
    /var base = "\/ServingStudioIntro\/";/,
  );
  assert.throws(() => readFileSync(join(out, "assets/main.js")));
});
