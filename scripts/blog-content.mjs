import { readdirSync, readFileSync, existsSync } from "node:fs";
import path from "node:path";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import Markdown, { defaultUrlTransform } from "react-markdown";
import remarkGfm from "remark-gfm";
import { removeInjected } from "./static-markup.mjs";

// Blog categories, in filter order. Every post tag must be one of these.
export const blogTags = ["Release", "Notes", "Model Perf", "Use Cases"];

const virtualId = "virtual:blog-posts";
const resolvedId = `\0${virtualId}`;
const escapeHtml = (value) =>
  String(value).replace(
    /[&<>"']/g,
    (char) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        char
      ],
  );

function filesIn(directory) {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const name = path.join(directory, entry.name);
    return entry.isDirectory() ? filesIn(name) : [name];
  });
}

export function readBlogPosts(root, base) {
  const directory = path.join(root, "content/blog");
  return readdirSync(directory, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map(({ name: slug }) => {
      if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(slug))
        throw new Error(`Invalid blog slug: ${slug}`);
      const folder = path.join(directory, slug);
      const metadata = JSON.parse(
        readFileSync(path.join(folder, "metadata.json"), "utf8"),
      );
      if (metadata.draft === true) return null;
      for (const field of ["title", "description", "cover", "coverAlt"]) {
        if (typeof metadata[field] !== "string" || !metadata[field].trim()) {
          throw new Error(`${slug}: metadata.${field} must be a nonempty string`);
        }
      }
      for (const field of ["authors", "tags"]) {
        if (
          !Array.isArray(metadata[field]) ||
          !metadata[field].length ||
          metadata[field].some(
            (value) => typeof value !== "string" || !value.trim(),
          )
        ) {
          throw new Error(
            `${slug}: metadata.${field} must be a nonempty list of strings`,
          );
        }
      }
      const unknownTags = metadata.tags.filter((tag) => !blogTags.includes(tag));
      if (unknownTags.length) {
        throw new Error(
          `${slug}: unknown tags ${unknownTags.join(", ")}; use ${blogTags.join(", ")}`,
        );
      }
      if (
        metadata.date != null &&
        (!/^\d{4}-\d{2}-\d{2}$/.test(metadata.date) ||
          new Date(metadata.date).toISOString().slice(0, 10) !== metadata.date)
      ) {
        throw new Error(`${slug}: metadata.date must be YYYY-MM-DD or null`);
      }
      const href = `${base}blog/${slug}/`;
      const assets = existsSync(path.join(folder, "assets"))
        ? filesIn(path.join(folder, "assets"))
        : [];
      const assetNames = new Set(
        assets.map((file) => path.relative(folder, file).split(path.sep).join("/")),
      );
      const localAsset = (url) => {
        const name = url.replace(/^\.\//, "").split(/[?#]/)[0];
        if (!assetNames.has(name))
          throw new Error(`${slug}: missing post asset ${url}`);
        return `${href}${url.replace(/^\.\//, "")}`;
      };
      const cover = localAsset(metadata.cover);
      const markdown = readFileSync(path.join(folder, "index.md"), "utf8");
      const interactive = metadata.interactive
        ? readInteractive(slug, folder, metadata.interactive, localAsset)
        : null;
      const headings = [];
      const seen = new Map();
      const textOf = (node) =>
        node.value || (node.children || []).map(textOf).join("");
      function articleHeadings() {
        return (tree) => {
          // Metadata owns the page title; keep the original heading in the source file.
          if (tree.children[0]?.type === "heading" && tree.children[0].depth === 1)
            tree.children.shift();
          for (const node of tree.children) {
            if (node.type !== "heading") continue;
            const text = textOf(node);
            const stem =
              text
                .toLowerCase()
                .replace(/[^\p{L}\p{N}]+/gu, "-")
                .replace(/^-|-$/g, "") || "section";
            const count = seen.get(stem) || 0;
            seen.set(stem, count + 1);
            const id = count ? `${stem}-${count + 1}` : stem;
            node.data = { ...node.data, hProperties: { id } };
            if (node.depth === 2) headings.push({ id, text });
          }
        };
      }
      // Raw HTML stays text, except a paired <u>…</u> inside one paragraph,
      // which becomes an underline.
      function underlineTags() {
        const visit = (node) => {
          const children = node.children;
          if (!children) return;
          for (let i = 0; i < children.length; i++) {
            if (children[i].type !== "html" || children[i].value !== "<u>")
              continue;
            const end = children.findIndex(
              (child, j) =>
                j > i && child.type === "html" && child.value === "</u>",
            );
            if (end < 0) continue;
            children.splice(i, end - i + 1, {
              type: "underline",
              data: { hName: "u" },
              children: children.slice(i + 1, end),
            });
          }
          children.forEach(visit);
        };
        return visit;
      }
      const html = renderToStaticMarkup(
        createElement(
          Markdown,
          {
            remarkPlugins: [remarkGfm, articleHeadings, underlineTags],
            components: {
              img({ node, ...props }) {
                return createElement("img", {
                  ...props,
                  loading: "lazy",
                  decoding: "async",
                });
              },
            },
            urlTransform(url) {
              const safe = defaultUrlTransform(url);
              if (!safe || /^(?:[a-z]+:|\/|#)/i.test(safe)) return safe;
              return localAsset(safe);
            },
          },
          markdown,
        ),
      );
      if (interactive) {
        // An interactive post's page shows its own HTML (with every figure's
        // data as text inside it); index.md stays its Markdown version.
        headings.length = 0;
        headings.push(...interactive.headings);
      }
      return {
        slug,
        href,
        title: metadata.title,
        description: metadata.description,
        authors: metadata.authors,
        date: metadata.date ?? null,
        tags: [...new Set(metadata.tags)],
        cover,
        coverAlt: metadata.coverAlt,
        readingMinutes: Math.max(
          1,
          Math.ceil((interactive?.prose ?? markdown).split(/\s+/).length / 220),
        ),
        html: interactive?.html ?? html,
        interactive: interactive && {
          styles: interactive.styles,
          scripts: interactive.scripts,
        },
        headings,
        assets: assets.map((file) => ({
          file,
          name: `blog/${slug}/${path.relative(folder, file).split(path.sep).join("/")}`,
        })),
      };
    })
    .filter(Boolean)
    .sort(
      (a, b) =>
        (b.date || "").localeCompare(a.date || "") || a.slug.localeCompare(b.slug),
    );
}

/* An interactive post (metadata.interactive = { body, styles, scripts }, each
   a path to one of the post's assets): the page shows `body`, an HTML
   fragment the post's own build wrote, instead of rendering index.md, and
   loads `styles` and then `scripts`, in order, once the article is on the
   page (Blog.jsx). The fragment is trusted like the site's own code. Its
   relative `src`/`href` must name post assets ("assets/…"), which resolve
   under the post's URL; its <h2 id> headings make the contents column.
   index.md is still the post's Markdown version, so crawlers and agents read
   the text (and the figures' data, if the post writes it there) without
   running the scripts. */
function readInteractive(slug, folder, spec, localAsset) {
  const list = (field) => {
    const value = spec[field] ?? [];
    if (!Array.isArray(value) || value.some((entry) => typeof entry !== "string"))
      throw new Error(
        `${slug}: metadata.interactive.${field} must be a list of asset paths`,
      );
    return value.map(localAsset);
  };
  if (typeof spec.body !== "string")
    throw new Error(`${slug}: metadata.interactive.body must be an asset path`);
  localAsset(spec.body);
  const body = readFileSync(path.join(folder, spec.body), "utf8");
  const html = body.replace(
    /\b(src|href)="(?![a-z]+:|\/|#)([^"]+)"/gi,
    (_, attribute, url) => `${attribute}="${localAsset(url)}"`,
  );
  const plain = (fragment) =>
    fragment
      .replace(/<[^>]+>/g, "")
      .replace(/&amp;/g, "&")
      .replace(/&lt;/g, "<")
      .replace(/&gt;/g, ">")
      .replace(/&quot;/g, '"')
      .replace(/&#0?39;/g, "'")
      .trim();
  const headings = [
    ...body.matchAll(/<h2\b[^>]*\bid="([^"]+)"[^>]*>([\s\S]*?)<\/h2>/g),
  ].map(([, id, text]) => ({ id, text: plain(text) }));
  // reading time counts the prose, not the figures' data tables
  const prose = plain(body.replace(/<details[\s\S]*?<\/details>/g, " "));
  return {
    html,
    headings,
    prose,
    styles: list("styles"),
    scripts: list("scripts"),
  };
}

export function blogContent() {
  let config;
  let cached;
  const getPosts = () => (cached ??= readBlogPosts(config.root, config.base));
  return {
    name: "blog-content",
    enforce: "post",
    configResolved(value) {
      config = value;
    },
    buildStart() {
      getPosts();
    },
    resolveId(id) {
      if (id === virtualId) return resolvedId;
    },
    load(id) {
      if (id !== resolvedId) return;
      const posts = getPosts().map(({ assets, ...post }) => post);
      return `export const tags = ${JSON.stringify(blogTags)};\nexport default ${JSON.stringify(posts)};`;
    },
    configureServer(server) {
      const contentDirectory = path.join(server.config.root, "content/blog");
      server.watcher.add(contentDirectory);
      const refresh = (file) => {
        if (!file.startsWith(`${contentDirectory}${path.sep}`)) return;
        cached = undefined;
        const module = server.moduleGraph.getModuleById(resolvedId);
        if (module) server.moduleGraph.invalidateModule(module);
        server.ws.send({ type: "full-reload" });
      };
      server.watcher.on("add", refresh).on("change", refresh).on("unlink", refresh);
      server.httpServer?.once("close", () => {
        server.watcher
          .off("add", refresh)
          .off("change", refresh)
          .off("unlink", refresh);
      });
      server.middlewares.use((req, res, next) => {
        const pathname = new URL(req.url, "http://localhost").pathname;
        for (const post of getPosts()) {
          if (pathname.startsWith(`${post.href}assets/`)) {
            req.url = `${config.base}content/blog/${post.slug}/${pathname.slice(post.href.length)}`;
            break;
          }
          if (pathname === post.href || pathname === `${post.href}index.html`) {
            req.url = `${config.base}blog.html`;
            break;
          }
        }
        next();
      });
    },
    generateBundle(_, bundle) {
      const template = bundle["blog.html"]?.source;
      if (!template) throw new Error("Blog HTML entry is missing from the build");
      for (const post of getPosts()) {
        for (const asset of post.assets)
          this.emitFile({
            type: "asset",
            fileName: asset.name,
            source: readFileSync(asset.file),
          });
        // The blog index's own text for crawlers (static-markup.mjs) is not
        // the post's; the post has its noscript article and its Markdown.
        let html = removeInjected(String(template))
          .replace(
            "</head>",
            `  <link rel="alternate" type="text/markdown" href="${post.href}index.md" />\n  </head>`,
          )
          .replace(
            /<title>[^<]*<\/title>/,
            `<title>${escapeHtml(post.title)} | ServingStudio</title>`,
          )
          .replace(
            /(<meta\s+name="description"\s+content=")[^"]*(")/,
            (_, opening, closing) =>
              `${opening}${escapeHtml(post.description)}${closing}`,
          )
          .replace(/<link\b[^>]*rel="preload"[^>]*>/g, "");
        // Readers without JavaScript still get the complete article, including its figures.
        html = html.replace(
          "</body>",
          `<noscript><article style="max-width:760px;margin:64px auto;padding:24px;line-height:1.8"><a href="${config.base}blog.html">All posts</a><h1>${escapeHtml(post.title)}</h1>${post.html}</article></noscript></body>`,
        );
        this.emitFile({
          type: "asset",
          fileName: `blog/${post.slug}/index.html`,
          source: html,
        });
      }
    },
  };
}
