import { readBlogPosts } from "./blog-content.mjs";

const escapeXml = (value) =>
  String(value).replace(
    /[&<>"']/g,
    (char) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&apos;" })[
        char
      ],
  );

// `entries` are { loc, lastmod? } with absolute URLs; lastmod is YYYY-MM-DD.
export function renderSitemap(entries) {
  const urls = entries
    .map(
      ({ loc, lastmod }) =>
        `  <url>\n    <loc>${escapeXml(loc)}</loc>\n${lastmod ? `    <lastmod>${lastmod}</lastmod>\n` : ""}  </url>`,
    )
    .join("\n");
  return `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${urls}\n</urlset>\n`;
}

// Emits sitemap.xml for search engines: the listed pages plus every published
// blog post. `origin` is the scheme and host; the Vite base supplies the path.
export function sitemap({ origin, pages }) {
  let config;
  return {
    name: "sitemap",
    apply: "build",
    configResolved(value) {
      config = value;
    },
    generateBundle() {
      const url = (path) => `${origin}${config.base}${path}`;
      const posts = readBlogPosts(config.root, config.base);
      this.emitFile({
        type: "asset",
        fileName: "sitemap.xml",
        source: renderSitemap([
          ...pages.map((page) => ({ loc: url(page) })),
          ...posts.map((post) => ({
            loc: `${origin}${post.href}`,
            lastmod: post.date ?? undefined,
          })),
        ]),
      });
    },
  };
}
