import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { blogContent } from "./scripts/blog-content.mjs";
import { sitemap } from "./scripts/sitemap.mjs";

export default defineConfig({
  /* The site is published at the root of https://servingstudio.cs.washington.edu/
     by scripts/deploy-cse.sh. Paths written as strings in JSX still read
     import.meta.env.BASE_URL, so moving the site under a subpath again only
     needs this value changed. */
  base: "/",
  plugins: [
    react(),
    blogContent(),
    // Submit https://servingstudio.cs.washington.edu/sitemap.xml in Google
    // Search Console. The cover template blog-overview.html is omitted.
    sitemap({
      origin: "https://servingstudio.cs.washington.edu",
      pages: ["", "features.html", "architecture.html", "blog.html"],
    }),
  ],
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
