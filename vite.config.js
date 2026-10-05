import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import { blogContent } from "./scripts/blog-content.mjs";
import { servingStudioUi } from "./scripts/servingstudio-ui.mjs";
import { sitemap } from "./scripts/sitemap.mjs";

/* The Models and Kernels pages read ServingStudio Sim's read-only public API.
   Pages request /api/public/v1 from their own origin and the server in front of
   the site forwards it: public/.htaccess on the CSE site, and here Vite, pointed
   at the service by PUBLIC_API_PROXY_TARGET (`uv run python -m public_api serve`
   in Sim). */
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
        pages: [
          "",
          "features.html",
          "architecture.html",
          "models.html",
          "simulate.html",
          "kernels.html",
          "blog.html",
        ],
      }),
    ],
    server: { proxy: publicApiProxy(env) },
    preview: { proxy: publicApiProxy(env) },
    build: {
      rollupOptions: {
        input: {
          overview: "index.html",
          blogOverview: "blog-overview.html",
          blog: "blog.html",
          features: "features.html",
          architecture: "architecture.html",
          models: "models.html",
          simulate: "simulate.html",
          kernels: "kernels.html",
        },
      },
    },
  };
});
