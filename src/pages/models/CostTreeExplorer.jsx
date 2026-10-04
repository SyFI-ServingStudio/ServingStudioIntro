import { ArrowUpRight, CircleCheck, CircleDashed } from "lucide-react";
import { Fragment, useEffect, useMemo, useState } from "react";
import { TreeRows, backendLabels } from "../../components/CostTree";
import { formatValue } from "../kernels/kernelData";
import {
  allIds,
  formatMs,
  initiallyOpen,
  kernelLink,
  nestSection,
  paramValue,
  prepareTree,
  shortReference,
  visible,
} from "./modelData";
import detail from "../kernels/KernelDetail.module.css";
import s from "./Models.module.css";

const KIND = { sum: "Sum", max: "Max", scale: "Scale", leaf: "Leaf" };
// Past this many, a leaf's shape reads "…"; the kernel page has the rest.
const SHAPE_ARGS = 5;

/* One member's cost tree, as /models/.../tree gives it: how the simulator
   puts an iteration's time together from kernel calls. A leaf says which
   kernel it calls and the config it reads rows with. `times`, once Live
   predict has timed a batch, puts a time on every node: { section: time per
   node }, indexed by the tree's node ids. `kernels` maps a kind to its
   catalog entry, for the leaf's title and link. */
export function CostTreeExplorer({ tree, error, kernels, times }) {
  if (error)
    return (
      <section className={s.treePanel}>
        <p role="alert">The cost tree did not load ({error.message}).</p>
      </section>
    );
  if (!tree)
    return (
      <section className={s.treePanel} aria-busy="true">
        <p role="status" className={s.treeLoading}>
          Loading the cost tree…
        </p>
      </section>
    );
  if (tree.error)
    return (
      <section className={s.treePanel}>
        <p role="alert">The simulator could not build this tree: {tree.error}</p>
      </section>
    );
  return (
    <Tree
      key={JSON.stringify([tree.preset, tree.params])}
      tree={tree}
      kernels={kernels ?? new Map()}
      times={times}
    />
  );
}

// Params every tree has and no reader picks.
const HIDDEN = ["type", "model_config", "num_layers", "sim_num_layers"];

/* The arch block the preset builds this member with, every param filled. */
function BuiltWith({ arch }) {
  const items = Object.entries(arch).filter(([name]) => !HIDDEN.includes(name));
  if (!items.length) return null;
  return (
    <p className={s.defaults}>
      Built with{" "}
      {items.map(([name, value], index) => (
        <span key={name} title={typeof value === "string" ? value : undefined}>
          {index > 0 && ", "}
          <code>{name}</code>{" "}
          <span className={s.runValue}>
            {typeof value === "string" ? shortReference(value) : paramValue(value)}
          </span>
        </span>
      ))}
      .
    </p>
  );
}

function Tree({ tree, kernels, times }) {
  const [sectionIndex, setSection] = useState(0);
  const section = tree.sections[Math.min(sectionIndex, tree.sections.length - 1)];
  const root = useMemo(() => prepareTree(nestSection(section)), [section]);
  const timed = times?.[section.section] ?? null;
  const [open, setOpen] = useState(() => initiallyOpen(root));
  useEffect(() => setOpen(initiallyOpen(root)), [root]);
  const toggle = (id) =>
    setOpen((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const rows = [...visible(root, open)].map((node) => {
    const time = timed && <NodeTime ms={timed[node.id]} repeated={node.repeated} />;
    return node.kind === "leaf"
      ? leafRow(tree, kernels, node, time)
      : compositeRow(node, open, toggle, time);
  });
  const calls = tree.sections.reduce((sum, item) => sum + item.slots.length, 0);
  const configs = Object.keys(tree.configs).length;
  const lacking = Object.values(tree.missing ?? {}).reduce((a, b) => a + b, 0);

  return (
    <section className={s.treePanel} aria-labelledby="tree-title">
      <div className={s.treeHead}>
        <h2 id="tree-title">Cost tree</h2>
        <dl className={s.stats}>
          <div>
            <dt>Kernel calls</dt>
            <dd>{calls.toLocaleString("en-US")}</dd>
          </div>
          <div>
            <dt>Kernel configs</dt>
            <dd>{configs.toLocaleString("en-US")}</dd>
          </div>
          <div>
            <dt>Rows lacking</dt>
            <dd>
              {tree.missing == null ? "unchecked" : lacking.toLocaleString("en-US")}
            </dd>
          </div>
          <div>
            <dt>GPUs per replica</dt>
            <dd>{tree.gpus_per_replica ?? "unknown"}</dd>
          </div>
        </dl>
        <BuiltWith arch={tree.arch} />
      </div>

      <div className={s.treeBar}>
        {tree.sections.length > 1 ? (
          <div className={detail.viewSwitch} role="radiogroup" aria-label="Section">
            {tree.sections.map((item, index) => (
              <button
                key={item.section}
                type="button"
                role="radio"
                aria-checked={item === section}
                onClick={() => setSection(index)}
              >
                {item.section.replaceAll("_", " ")}
              </button>
            ))}
          </div>
        ) : (
          <span />
        )}
        <div className={s.treeActions}>
          <button type="button" onClick={() => setOpen(new Set(allIds(root)))}>
            Expand all
          </button>
          <button type="button" onClick={() => setOpen(new Set())}>
            Collapse all
          </button>
        </div>
      </div>

      <div className={s.treeScroll}>
        <TreeRows rows={rows} className={s.tree} label="Cost tree nodes" />
      </div>
    </section>
  );
}

// A dotted name may break after a dot on a narrow screen, not mid-word.
const breakable = (name) =>
  name.includes(".")
    ? name.split(".").map((part, index) => (
        <Fragment key={index}>
          {index > 0 && (
            <>
              .<wbr />
            </>
          )}
          {part}
        </Fragment>
      ))
    : name;

/* A node's predicted time. Inside a Scale it is one repeat's: the Scale
   row holds the repeats' total. */
function NodeTime({ ms, repeated }) {
  if (ms == null) return null;
  return (
    <span className={s.nodeTime}>
      {formatMs(ms)} ms
      {repeated && <small> per repeat</small>}
    </span>
  );
}

function compositeRow(node, open, toggle, time) {
  const after =
    node.kind === "scale" ? (
      <span className={s.times}>×{node.n.toLocaleString("en-US")}</span>
    ) : node.kind === "max" ? (
      <span className={s.ranks}>
        {node.copies
          ? `${node.copies} alike, one shown`
          : `slowest of ${node.width}`}
        {node.overlap !== 1 && `, ÷ ${node.overlap}`}
      </span>
    ) : null;
  return {
    id: node.id,
    kind: KIND[node.kind],
    depth: node.depth,
    name: breakable(node.name),
    note: node.note,
    toggle: () => toggle(node.id),
    open: open.has(node.id),
    after,
    meta: (
      <>
        {time}
        <span className={s.nodeCount}>
          {node.leaves.toLocaleString("en-US")}{" "}
          {node.leaves === 1 ? "call" : "calls"}
        </span>
      </>
    ),
  };
}

/* A config's scalar identity, the first few fields. */
function shapeText(config) {
  if (!config) return null;
  const entries = Object.entries(config.identity);
  const parts = entries
    .slice(0, SHAPE_ARGS)
    .map(([name, value]) => `${name} ${formatValue(value)}`);
  if (entries.length > SHAPE_ARGS || config.structured.length) parts.push("…");
  return parts.join(", ");
}

function leafRow(tree, kernels, node, time) {
  const { slot } = node;
  const kernel = kernels.get(slot.kernel);
  const href = kernelLink(kernel, slot, tree);
  const title = kernel?.title ?? slot.kernel;
  // The rows this member asks of the leaf's config that profile.db lacks:
  // 0 when every one is measured, null when unchecked.
  const lacking = tree.configs[slot.config]?.missing;
  const checked = lacking != null;
  return {
    id: node.id,
    kind: "Leaf",
    depth: node.depth,
    name: breakable(node.name),
    note: shapeText(tree.configs[slot.config]),
    meta: (
      <>
        {time}
        {href ? (
          <a className={s.kernelLink} href={href}>
            {title}
            <ArrowUpRight size={15} aria-hidden="true" />
          </a>
        ) : (
          <span className={s.kernelPlain} title="This kernel has no page yet">
            {title}
          </span>
        )}
        <span
          className={s.coverage}
          data-state={checked ? (lacking ? "unmeasured" : "measured") : undefined}
          title={
            lacking
              ? `profile.db lacks ${lacking.toLocaleString("en-US")} of the rows this parameter set reads of this config; the kernel page shows its cells`
              : checked
                ? "Every row this parameter set reads of this config is measured"
                : undefined
          }
        >
          {checked &&
            (lacking ? (
              <CircleDashed size={14} aria-hidden="true" />
            ) : (
              <CircleCheck size={14} aria-hidden="true" />
            ))}
          {lacking ? "Lacks rows" : checked ? "Measured" : null}
          <span className={s.backends}>
            {slot.backends.map((b) => backendLabels[b] || b).join(", ")}
          </span>
        </span>
      </>
    ),
  };
}
