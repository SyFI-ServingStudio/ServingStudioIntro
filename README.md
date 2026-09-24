# ServingStudio Intro

The introduction website for ServingStudio: its simulation foundation, Agent
workflows, performance analysis, and applications to real serving frameworks.

## Development

```bash
npm ci
npm run dev
```

The development server listens on all interfaces by default.

## Production build

```bash
npm run build
```

Vite writes the deployable site to `dist/`.

## Blog posts

The blog index is at `/ServingStudioIntro/blog.html`. Each post has one source
directory; its folder name becomes the URL slug:

```text
content/blog/introducing-servingstudio/
├── index.md
├── metadata.json
└── assets/
    ├── overview.png
    ├── overview.webp
    └── thumbnail.webp
```

To add a post, create a directory under `content/blog/` with a lowercase,
hyphen-separated name. Add the Markdown, metadata, and any figures to that
directory. The index updates automatically; no React or route changes are
needed. The development server also reloads when content changes.

`metadata.json` requires `title`, `description`, `authors` (an array), `tags`
(an array), `cover`, and `coverAlt`. Tags must come from the fixed categories
`Release`, `Notes`, `Model Perf`, and `Use Cases`, which appear as the index
filters in that order; the build rejects any other tag. To add a category, edit
`blogTags` in `scripts/blog-content.mjs`. Use a relative asset path such as
`assets/thumbnail.webp` for the cover. Set `date` to the publication date in
`YYYY-MM-DD` format, or `null` to omit it. Dated posts appear newest first,
followed by undated posts. Set `draft: true` to exclude a post and its assets
from the website build.

Write image and download links relative to `index.md`, for example
`![Performance comparison](assets/comparison.webp)`. Local files must live in
that post's `assets/` directory. External HTTPS links also work. Keep the original
PNG or other source figures alongside compressed web versions when needed.

Markdown supports tables, lists, code blocks, and links. The metadata supplies
the displayed title; an initial `#` heading in the Markdown is omitted when
rendering. Second-level headings populate the article's table of contents.
Raw HTML is displayed as text.

`npm run build` validates the metadata and local asset references, then generates
each article at `dist/blog/<slug>/index.html` with its assets. These URLs work on
GitHub Pages without a server-side router. The introduction article is available
at `/ServingStudioIntro/blog/introducing-servingstudio/`.

Run `npm test`, `npm run lint:css`, and `npm run build` before publishing.
The previous introduction draft now lives at
`content/blog/introducing-servingstudio/index.md`; historical drafts remain local.
The cover export template remains `blog-overview.html`. Save future captures to
the post's `assets/overview.png` and refresh both WebP versions.

## Single-file build

```bash
npm run build:single
```

This produces `dist/ServingStudioIntro.html` with the JavaScript, CSS, and logo
inlined. The file can be opened directly or uploaded to static hosting.
