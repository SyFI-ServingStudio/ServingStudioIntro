import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { UI_DEPENDENCIES, servingStudioUi, uiSource } from "./servingstudio-ui.mjs";

const quiet = { log() {}, warn() {} };

function uiCheckout({ embed = true } = {}) {
  const dir = mkdtempSync(join(tmpdir(), "servingstudio-ui-"));
  const git = (...args) =>
    execFileSync("git", ["-C", dir, ...args], { stdio: "pipe" });
  git("init", "-q");
  mkdirSync(join(dir, "app", "src", "embed"), { recursive: true });
  if (embed)
    writeFileSync(join(dir, "app", "src", "embed", "index.ts"), "export {};\n");
  writeFileSync(join(dir, "README.md"), "UI\n");
  git("add", ".");
  git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "ui");
  const head = execFileSync("git", ["-C", dir, "rev-parse", "HEAD"], {
    encoding: "utf8",
  }).trim();
  return { dir, head };
}

const info = (plugin) =>
  JSON.parse(
    plugin
      .load(plugin.resolveId("virtual:servingstudio-ui"))
      .replace(/^export default /, "")
      .replace(/;$/, ""),
  );

test("a checkout's viewer is the embed entry, at its commit", () => {
  const { dir, head } = uiCheckout();
  const plugin = servingStudioUi({ uiDir: dir, log: quiet });
  assert.equal(
    plugin.resolveId("@servingstudio/ui/embed"),
    join(dir, "app", "src", "embed", "index.ts"),
  );
  assert.deepEqual(info(plugin), { available: true });
  const config = plugin.config({}, { command: "build" });
  assert.deepEqual(config.resolve.dedupe, UI_DEPENDENCIES);
  assert.ok(config.server.fs.allow.includes(dir));
});

test("a checkout with uncommitted UI changes is marked dirty", () => {
  const { dir, head } = uiCheckout();
  writeFileSync(join(dir, "app", "src", "embed", "index.ts"), "export {}; //\n");
  assert.equal(uiSource({ uiDir: dir }).commit, `${head}-dirty`);
});

test("without a checkout there is no viewer, and the log says why", () => {
  const logged = [];
  const log = { log() {}, warn: (line) => logged.push(line) };
  const plugin = servingStudioUi({ log });
  assert.deepEqual(info(plugin), { available: false });
  assert.equal(
    plugin.load(plugin.resolveId("@servingstudio/ui/embed")),
    "export const ResultViewer = null;",
  );
  assert.equal(plugin.config({}, { command: "serve" }), undefined);
  assert.match(logged.join("\n"), /SERVINGSTUDIO_UI_DIR is not set/);
});

test("a build without a checkout stops unless asked not to", () => {
  const { dir } = uiCheckout({ embed: false });
  assert.throws(
    () =>
      servingStudioUi({ uiDir: dir, log: quiet }).config({}, { command: "build" }),
    /has no app\/src\/embed\/index\.ts.*ALLOW_NO_READ_MORE=1/,
  );
  const allowed = servingStudioUi({ uiDir: dir, allowMissing: true, log: quiet });
  assert.equal(allowed.config({}, { command: "build" }), undefined);
  assert.equal(info(allowed).available, false);
});

test("every package deduplicated for the UI is the site's own", () => {
  const site = JSON.parse(
    readFileSync(new URL("../package.json", import.meta.url), "utf8"),
  );
  for (const name of UI_DEPENDENCIES)
    assert.ok(site.dependencies[name], `package.json lacks ${name}`);
});
