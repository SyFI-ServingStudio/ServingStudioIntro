import { readdirSync, readFileSync, existsSync } from "node:fs";
import path from "node:path";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import Markdown, { defaultUrlTransform } from "react-markdown";
import remarkGfm from "remark-gfm";

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
      const html = renderToStaticMarkup(
        createElement(
          Markdown,
          {
            remarkPlugins: [remarkGfm, articleHeadings],
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
        readingMinutes: Math.max(1, Math.ceil(markdown.split(/\s+/).length / 220)),
        html,
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
        let html = String(template)
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
