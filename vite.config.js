import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import { agentContent } from "./scripts/agent-content.mjs";
import { blogContent } from "./scripts/blog-content.mjs";
import { servingStudioUi } from "./scripts/servingstudio-ui.mjs";
import { sitemap } from "./scripts/sitemap.mjs";
import { PAGES } from "./src/sitePages.js";

/* The Models, Simulate and Kernels pages use ServingStudio Sim's public API.
   Pages request /api/public/v1 from their own origin and the server in front of
   the site forwards it: on the CSE site the .htaccess scripts/deploy-cse.sh
   writes, and here Vite, pointed at the service by PUBLIC_API_PROXY_TARGET
   (`uv run python -m public_api serve` in Sim). */
function publicApiProxy(env) {
  const target = env.PUBLIC_API_PROXY_TARGET;
  return target ? { "/api/public/v1": { target, changeOrigin: true } } : undefined;
}

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  return {
    /* The site is published at the root of https://servingstudio.cs.washington.edu/
       by scripts/deploy-cse.sh. Paths written as strings in JSX still read
       import.meta.env.BASE_URL, so moving the site under a subpath again only
       needs this value changed. */
    base: "/",
    plugins: [
      servingStudioUi({
        uiDir: env.SERVINGSTUDIO_UI_DIR,
        allowMissing: env.ALLOW_NO_READ_MORE === "1",
      }),
      react(),
      blogContent(),
      // Submit https://servingstudio.cs.washington.edu/sitemap.xml in Google
      // Search Console. The cover template blog-overview.html is omitted.
      sitemap({
        origin: "https://servingstudio.cs.washington.edu",
        pages: PAGES.map((p) => p.file),
      }),
      // Text for readers without JavaScript: the catalog pages' lists baked
      // into their HTML and as Markdown, llms.txt, robots.txt and the agent
      // skills. Reads the public API once; see scripts/agent-content.mjs.
      agentContent({
        origin: "https://servingstudio.cs.washington.edu",
        pages: PAGES,
        apiBase: env.PUBLIC_API_CATALOG_BASE,
        allowMissing: env.ALLOW_NO_API_CATALOG === "1",
        simDir: env.SERVINGSTUDIO_SIM_DIR,
        allowMissingSkills: env.ALLOW_NO_SKILLS === "1",
      }),
    ],
    server: { proxy: publicApiProxy(env) },
    preview: { proxy: publicApiProxy(env) },
    build: {
      rollupOptions: {
        input: {
          ...Object.fromEntries(PAGES.map((p) => [p.id, p.file || "index.html"])),
          blogOverview: "blog-overview.html",
        },
      },
    },
  };
});
