/* The Models page's "Read more": ServingStudio UI's own result pages
   (app/src/embed in a ServingStudioUI checkout), compiled into this site from
   source, in the page's React tree. SERVINGSTUDIO_UI_DIR names the checkout.

   The UI's imports of its dependencies resolve here, from this site's
   node_modules (package.json carries the UI's runtime dependencies at its
   versions), so the page has one React, one emotion and one MUI. Pages read
   whether the viewer is there, and from which UI commit, from
   `virtual:servingstudio-ui`; `@servingstudio/ui/embed` is the viewer itself.

   Without a checkout development goes on without Read more; a build stops
   instead, unless ALLOW_NO_READ_MORE=1 asks for a site without it. */

import { execFileSync } from "node:child_process";
import { existsSync } from "node:fs";
import { join, resolve } from "node:path";
import { searchForWorkspaceRoot } from "vite";

const INFO = "virtual:servingstudio-ui";
const EMBED = "@servingstudio/ui/embed";
const MISSING = "\0servingstudio-ui-missing";

// Every package the UI's source imports at run time.
export const UI_DEPENDENCIES = [
  "react",
  "react-dom",
  "@emotion/cache",
  "@emotion/react",
  "@emotion/styled",
  "@mui/icons-material",
  "@mui/material",
  "@tanstack/react-query",
  "echarts",
  "echarts-for-react",
  "highlight.js",
  "motion",
  "zod",
];

/* The viewer's entry and the checkout's commit, or why there is none. */
export function uiSource({ uiDir }) {
  if (!uiDir)
    return {
      reason:
        "SERVINGSTUDIO_UI_DIR is not set; point it at a ServingStudioUI checkout.",
    };
  const dir = resolve(uiDir);
  const entry = join(dir, "app", "src", "embed", "index.ts");
  if (!existsSync(entry))
    return {
      reason: `${dir} has no app/src/embed/index.ts: check out a ServingStudioUI that ships the embedded viewer.`,
    };
  return { dir, entry, commit: commitOf(dir) };
}

function commitOf(dir) {
  const git = (...args) =>
    execFileSync("git", ["-C", dir, ...args], { encoding: "utf8" }).trim();
  const head = git("rev-parse", "HEAD");
  return git("status", "--porcelain", "--", "app").length ? `${head}-dirty` : head;
}

export function servingStudioUi({ uiDir, allowMissing = false, log = console }) {
  const from = uiSource({ uiDir });
  return {
    name: "servingstudio-ui",
    enforce: "pre",
    config(_, { command }) {
      if (!from.dir) {
        if (command === "build" && !allowMissing)
          throw new Error(
            `No ServingStudio UI for Read more: ${from.reason} To build the site without it, set ALLOW_NO_READ_MORE=1.`,
          );
        log.warn(`[servingstudio-ui] no Read more: ${from.reason}`);
        return undefined;
      }
      log.log(`[servingstudio-ui] Read more from ${from.dir} (${from.commit})`);
      return {
        resolve: { dedupe: UI_DEPENDENCIES },
        server: {
          fs: { allow: [searchForWorkspaceRoot(process.cwd()), from.dir] },
        },
      };
    },
    resolveId(id) {
      if (id === INFO) return `\0${INFO}`;
      if (id === EMBED) return from.entry ?? MISSING;
      return null;
    },
    load(id) {
      if (id === `\0${INFO}`)
        return `export default ${JSON.stringify(
          from.dir
            ? { available: true, commit: from.commit }
            : { available: false, reason: from.reason },
        )};`;
      if (id === MISSING) return "export const ResultViewer = null;";
      return null;
    },
  };
}
