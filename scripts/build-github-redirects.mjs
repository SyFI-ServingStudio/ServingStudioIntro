// Writes the GitHub Pages copy of the site: one redirect page per HTML page
// in dist/, each pointing at the same path on the CSE site, plus a 404.html
// that forwards any other old URL. GitHub Pages cannot send HTTP redirects, so
// each page redirects by script (keeping the query and hash), falls back to a
// meta refresh, and names the new URL as canonical for search engines.
//
//   npm run build && node scripts/build-github-redirects.mjs
import { mkdirSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join, relative } from "node:path";

export const origin = "https://servingstudio.cs.washington.edu";
// The path GitHub Pages serves this repository under.
export const oldBase = "/ServingStudioIntro/";

const escapeHtml = (value) =>
  value.replace(
    /[&<>"]/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c],
  );

// dist-relative HTML file to its URL path: a directory index is served at the
// directory, so blog/x/index.html becomes blog/x/.
export const pagePath = (file) => file.replace(/(^|\/)index\.html$/, "$1");

export function redirectPage(target) {
  const url = escapeHtml(target);
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ServingStudio has moved</title>
<link rel="canonical" href="${url}">
<script>location.replace(${JSON.stringify(target)} + location.search + location.hash);</script>
<meta http-equiv="refresh" content="0; url=${url}">
</head>
<body>
<p>ServingStudio has moved to <a href="${url}">${url}</a>.</p>
</body>
</html>
`;
}

// Served for every path with no page of its own. It strips the old base and
// sends the rest of the path to the same place on the new site.
export function notFoundPage() {
  const home = `${origin}/`;
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ServingStudio has moved</title>
<script>
  var path = location.pathname;
  var base = ${JSON.stringify(oldBase)};
  if (path.indexOf(base) === 0) path = "/" + path.slice(base.length);
  location.replace(${JSON.stringify(origin)} + path + location.search + location.hash);
</script>
<meta http-equiv="refresh" content="0; url=${home}">
</head>
<body>
<p>ServingStudio has moved to <a href="${home}">${home}</a>.</p>
</body>
</html>
`;
}

function htmlFiles(dir, root = dir) {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) return htmlFiles(path, root);
    return entry.name.endsWith(".html") ? [relative(root, path)] : [];
  });
}

export function buildRedirects(siteDir, outDir) {
  const files = htmlFiles(siteDir);
  if (!files.includes("index.html")) {
    throw new Error(`${siteDir} has no index.html; run npm run build first`);
  }
  rmSync(outDir, { recursive: true, force: true });
  for (const file of files) {
    const out = join(outDir, file);
    mkdirSync(dirname(out), { recursive: true });
    writeFileSync(out, redirectPage(`${origin}/${pagePath(file)}`));
  }
  writeFileSync(join(outDir, "404.html"), notFoundPage());
  return files;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const files = buildRedirects("dist", "dist-redirect");
  console.log(`dist-redirect/: ${files.length} redirect pages and 404.html`);
}
