import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { readBlogPosts } from "./blog-content.mjs";

function fixture(t) {
  const root = mkdtempSync(path.join(os.tmpdir(), "intro-blog-"));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  return {
    root,
    post(slug, metadata = {}, markdown = "# Title\n\n## Overview\n\nContent.") {
      const folder = path.join(root, "content/blog", slug);
      mkdirSync(path.join(folder, "assets"), { recursive: true });
      writeFileSync(
        path.join(folder, "assets/cover.svg"),
        '<svg xmlns="http://www.w3.org/2000/svg"/>',
      );
      writeFileSync(path.join(folder, "index.md"), markdown);
      writeFileSync(
        path.join(folder, "metadata.json"),
        JSON.stringify({
          title: "A serving experiment",
          description: "Measurements and analysis.",
          authors: ["ServingStudio team"],
          tags: ["Release"],
          date: null,
          cover: "assets/cover.svg",
          coverAlt: "Experiment results",
          ...metadata,
        }),
      );
    },
  };
}

test("discovers added folders, sorts dated posts, and excludes drafts", (t) => {
  const f = fixture(t);
  f.post("older", { date: "2026-09-01" });
  f.post("newer", { date: "2026-09-22" });
  f.post("undated");
  f.post("unfinished", { draft: true });
  assert.deepEqual(
    readBlogPosts(f.root, "/ServingStudioIntro/").map((p) => p.slug),
    ["newer", "older", "undated"],
  );
});

test("resolves cover and Markdown assets under the deployment base", (t) => {
  const f = fixture(t);
  f.post(
    "experiment",
    {},
    "# Title\n\n![Results](./assets/cover.svg)\n\n[Download](assets/cover.svg)",
  );
  const [post] = readBlogPosts(f.root, "/ServingStudioIntro/");
  assert.equal(post.cover, "/ServingStudioIntro/blog/experiment/assets/cover.svg");
  assert.match(
    post.html,
    /src="\/ServingStudioIntro\/blog\/experiment\/assets\/cover.svg"/,
  );
  assert.match(
    post.html,
    /href="\/ServingStudioIntro\/blog\/experiment\/assets\/cover.svg"/,
  );
  assert.equal(post.assets[0].name, "blog/experiment/assets/cover.svg");
});

test("rejects tags outside the fixed categories", (t) => {
  const f = fixture(t);
  f.post("experiment", { tags: ["Release", "Simulation"] });
  assert.throws(() => readBlogPosts(f.root, "/"), /unknown tags Simulation/);
});

test("fails on missing assets and paths outside the post folder", (t) => {
  const f = fixture(t);
  f.post("experiment", {}, "![Results](assets/missing.png)");
  assert.throws(() => readBlogPosts(f.root, "/"), /missing post asset/);
  f.post("experiment", { cover: "../another-post/assets/cover.svg" });
  assert.throws(() => readBlogPosts(f.root, "/"), /missing post asset/);
});

test("renders GFM and unique section anchors without executing raw HTML", (t) => {
  const f = fixture(t);
  f.post(
    "experiment",
    {},
    "# Title\n\n## Results\n\n## Results\n\n| A | B |\n| - | - |\n| 1 | 2 |\n\n<script>alert(1)</script>\n\n[Unsafe](javascript:alert)",
  );
  const [post] = readBlogPosts(f.root, "/");
  assert.deepEqual(
    post.headings.map((h) => h.id),
    ["results", "results-2"],
  );
  assert.match(post.html, /<table>/);
  assert.doesNotMatch(post.html, /<h1|<script|href="javascript:/);
});

test("rejects malformed author metadata and impossible dates", (t) => {
  const f = fixture(t);
  f.post("experiment", { authors: [] });
  assert.throws(() => readBlogPosts(f.root, "/"), /metadata.authors/);
  f.post("experiment", { date: "2026-02-30" });
  assert.throws(() => readBlogPosts(f.root, "/"), /metadata.date/);
});
