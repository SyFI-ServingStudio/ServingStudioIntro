import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import { blogContent } from "./scripts/blog-content.mjs";
import { sitemap } from "./scripts/sitemap.mjs";

/* The Kernel Library reads ServingStudio Sim's read-only public API. Pages
   request /api/public/v1 from their own origin and the server in front of the
   site forwards it; here that is Vite, pointed at the service by
   PUBLIC_API_PROXY_TARGET (`uv run python -m public_api serve` in Sim). */
function publicApiProxy(env) {
  const target = env.PUBLIC_API_PROXY_TARGET;
  return target ? { "/api/public/v1": { target, changeOrigin: true } } : undefined;
}

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  return {
    /* The site is published at https://syfi-servingstudio.github.io/ServingStudioIntro/, so
       every asset URL has to carry that prefix. Vite rewrites the ones it can
       see: relative imports, and the absolute /fonts and /images references in
       CSS. It cannot rewrite a path written as a string in JSX, so those read
       import.meta.env.BASE_URL instead. */
    base: "/ServingStudioIntro/",
    plugins: [
      react(),
      blogContent(),
      // Submit https://syfi-servingstudio.github.io/ServingStudioIntro/sitemap.xml
      // in Google Search Console. The cover template blog-overview.html is omitted.
      sitemap({
        origin: "https://syfi-servingstudio.github.io",
        pages: [
          "",
          "features.html",
          "architecture.html",
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
          kernels: "kernels.html",
        },
      },
    },
  };
});
