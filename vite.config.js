import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { blogContent } from "./scripts/blog-content.mjs";

export default defineConfig({
  /* The site is published at https://syfi-servingstudio.github.io/ServingStudioIntro/, so
     every asset URL has to carry that prefix. Vite rewrites the ones it can
     see: relative imports, and the absolute /fonts and /images references in
     CSS. It cannot rewrite a path written as a string in JSX, so those read
     import.meta.env.BASE_URL instead. */
  base: "/ServingStudioIntro/",
  plugins: [react(), blogContent()],
  build: {
    rollupOptions: {
      input: {
        overview: "index.html",
        blogOverview: "blog-overview.html",
        blog: "blog.html",
        features: "features.html",
        architecture: "architecture.html",
      },
    },
  },
});
