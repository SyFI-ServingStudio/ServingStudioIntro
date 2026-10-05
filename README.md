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
SERVINGSTUDIO_UI_DIR=/path/to/ServingStudioUI npm run build
```

Vite writes the deployable site to `dist/`.

### Read more

A live prediction on the Models page and a finished run on the Simulate page
have a "Read more" button that opens ServingStudio UI's own result pages for
them. Those pages are compiled into this
site from a ServingStudioUI checkout (`app/src/embed`, on a branch that ships
the embedded viewer), named by `SERVINGSTUDIO_UI_DIR`; see
`scripts/servingstudio-ui.mjs`.

- `npm run dev` without `SERVINGSTUDIO_UI_DIR` runs without Read more and says
  so in its log.
- `npm run build` without it stops, so a deploy does not drop Read more by
  accident. `ALLOW_NO_READ_MORE=1 npm run build` builds the site without the
  button; the GitHub Actions check sets it.
- `npm run deploy:cse` runs a plain `npm run build`, so set one of the two
  before it: `SERVINGSTUDIO_UI_DIR=… npm run deploy:cse` for the full site.

The viewer reads the prediction or run through the public API's forwarded
Analyzer routes (`/api/public/v1/analyzer/predictions/…`, `…/analyzer/runs/…`
and `…/analyzer/kernel-kinds`); `src/pages/models/ReadMore.jsx` answers any
other read it makes with a 404 and logs it.

## Deployment

The site is served from the root of https://servingstudio.cs.washington.edu/.
Its files live in `/cse/web/research/servingstudio`, which the CSE web hosts
such as `bicycle` and `recycle` mount. With passwordless ssh to `bicycle`,
publish the current checkout with:

```bash
npm run deploy:cse              # build, then rsync dist/ to the server
npm run deploy:cse -- --dry-run # build, then list what would change
```

The sync deletes server files that are not in the build, so put anything the
server needs, such as an `.htaccess`, in `public/`.

The Models, Simulate and Kernels pages use ServingStudio Sim's public API at
`/api/public/v1` on the site's own origin: they read its catalogs, and Live
predict and Simulate send it work (`POST /predict`, `POST /simulate`,
`POST /workloads` for an uploaded trace). On the CSE site, `public/.htaccess`
proxies that path to the service on cayenne (`10.158.48.50:5220`); if it is not
running, the pages say the data service is not reachable. In development, point
Vite at a running service:

```bash
PUBLIC_API_PROXY_TARGET=http://127.0.0.1:<port> npm run dev
```

The old address, https://syfi-servingstudio.github.io/ServingStudioIntro/, now
serves only redirects. On every push to `main`, the GitHub Actions workflow
checks the build, runs `scripts/build-github-redirects.mjs` to write one
redirect page per site page plus a catch-all `404.html`, and publishes them.
Each old URL forwards to the same path on the CSE site. Pushing does not update
the CSE site; run `npm run deploy:cse` for that.

## Blog posts

The blog index is at `/blog.html`. Each post has one source
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
Raw HTML is displayed as text, except `<u>…</u>`, which underlines text
within a paragraph.

`npm run build` validates the metadata and local asset references, then generates
each article at `dist/blog/<slug>/index.html` with its assets. These URLs work on
any static host without a server-side router. The introduction article is
available at `/blog/introducing-servingstudio/`.

The build also writes `sitemap.xml` with the main pages and every published
post (with its date as `lastmod`). Submit
`https://servingstudio.cs.washington.edu/sitemap.xml` in Google Search Console; new posts are added automatically. A page in
`PAGES` (`src/sitePages.js`) is listed, as it is in the navigation.

Run `npm test`, `npm run lint:css`, and `SERVINGSTUDIO_UI_DIR=… npm run build`
before publishing.
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
