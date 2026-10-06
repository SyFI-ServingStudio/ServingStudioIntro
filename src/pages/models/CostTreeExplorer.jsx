import { ArrowUpRight } from "lucide-react";
import { useEffect, useState } from "react";
import { TreeRows } from "../../components/CostTree";
import { KernelShare } from "./KernelShare";
import { Pending, breakable } from "./TreeParts";
import { formatValue } from "../kernels/kernelData";
import {
  formatMs,
  formatPercent,
  kernelLink,
  paramValue,
  parents,
  sectionName,
  shortReference,
  visibleRows,
} from "./modelData";
import detail from "../kernels/KernelDetail.module.css";
import s from "./Models.module.css";

const KIND = {
  sum: "Sum",
  max: "Max",
  parallel: "Parallel",
  scale: "Scale",
  leaf: "Leaf",
};
// Past this many, a leaf's shape reads "…"; the kernel page has the rest.
const SHAPE_ARGS = 5;

/* One member's cost tree: how the simulator puts an iteration's time
   together from kernel calls. `tree` (/models/.../tree) names each section's
   kernel calls, their kernels and configs; `member` (/models) counts them. `times` is the Analyzer's tree of
   the batch Live predict last timed, { section: nodes }: every node's time
   for one call and its share of the section, after its repeats. `kernels`
   maps a kind to its catalog entry, for the leaf's title and link. While
   `pending`, a newer batch is being timed: the rows keep their place and
   every time is a placeholder, never the last batch's number. `view` picks
   the tree, or the Analyzer's ranking of the batch (`share`, KernelShare) by
   kernel or by kernel type. `aside` (Live predict) sits beside the rows,
   level with their top, under the panel's heading. */
export function CostTreeExplorer({
  member,
  tree,
  error,
  kernels,
  times,
  share,
  pending,
  view,
  onView,
  aside,
}) {
  if (error)
    return (
      <section className={s.treePanel}>
        <TreeLayout aside={aside}>
          <p role="alert">The cost tree did not load ({error.message}).</p>
        </TreeLayout>
      </section>
    );
  if (!tree)
    return (
      <section className={s.treePanel} aria-busy="true">
        <TreeLayout aside={aside}>
          <p role="status" className={s.treeLoading}>
            Loading the cost tree…
          </p>
        </TreeLayout>
      </section>
    );
  if (tree.error)
    return (
      <section className={s.treePanel}>
        <TreeLayout aside={aside}>
          <p role="alert">The simulator could not build this tree: {tree.error}</p>
        </TreeLayout>
      </section>
    );
  return (
    <Tree
      key={JSON.stringify([tree.preset, tree.params])}
      member={member}
      tree={tree}
      kernels={kernels ?? new Map()}
      times={times}
      share={share}
      pending={pending}
      view={view}
      onView={onView}
      aside={aside}
    />
  );
}

/* The rows on the left, `aside` beside them on a wide screen and under them
   otherwise; the row is as tall as the taller of the two. */
function TreeLayout({ aside, children }) {
  return (
    <div className={s.treeLayout}>
      <div className={s.treeBody}>{children}</div>
      {aside}
    </div>
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

const VIEWS = [
  ["tree", "Tree"],
  ["kernels", "By kernel"],
  ["kinds", "By kernel type"],
];

function Tree({
  member,
  tree,
  kernels,
  times,
  share,
  pending,
  view,
  onView,
  aside,
}) {
  const [sectionIndex, setSection] = useState(0);
  const section = tree.sections[Math.min(sectionIndex, tree.sections.length - 1)];
  const nodes = times?.[section.section] ?? null;
  const [open, setOpen] = useState(() => new Set(nodes ? parents(nodes, 2) : []));
  // A new tree opens to its blocks; a new time on the same tree keeps the
  // reader's folding.
  const shape =
    nodes && JSON.stringify(nodes.map((node) => [node.node, node.depth]));
  useEffect(() => {
    if (nodes) setOpen(new Set(parents(nodes, 2)));
  }, [shape]); // eslint-disable-line react-hooks/exhaustive-deps
  const toggle = (id) =>
    setOpen((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const total = nodes?.[0].total_ms;
  const rows = nodes
    ? visibleRows(nodes, open).map((node) => {
        const time = <NodeTime node={node} pending={pending} />;
        return node.kind === "leaf"
          ? leafRow(tree, section, kernels, node, time)
          : compositeRow(node, open, toggle, time);
      })
    : [];

  return (
    <section className={s.treePanel} aria-labelledby="tree-title">
      <div className={s.treeHead}>
        <h2 id="tree-title">Cost tree</h2>
        <dl className={s.stats}>
          <div>
            <dt>Kernel calls</dt>
            <dd>{member.leaves.toLocaleString("en-US")}</dd>
          </div>
          <div>
            <dt>Kernel configs</dt>
            <dd>{member.configs.toLocaleString("en-US")}</dd>
          </div>
          {nodes && (
            <div>
              <dt>Critical path</dt>
              <dd>
                {pending ? (
                  <Pending className={s.pendingStat} />
                ) : (
                  <>{formatMs(total)} ms</>
                )}
              </dd>
            </div>
          )}
          <div>
            <dt>GPUs per replica</dt>
            <dd>{member.gpus_per_replica}</dd>
          </div>
        </dl>
        <BuiltWith arch={tree.arch} />
      </div>

      <TreeLayout aside={aside}>
        <div className={s.treeBar}>
          <div className={s.treeViews}>
            <div className={detail.viewSwitch} role="radiogroup" aria-label="View">
              {VIEWS.map(([id, label]) => (
                <button
                  key={id}
                  type="button"
                  role="radio"
                  aria-checked={view === id}
                  onClick={() => onView(id)}
                >
                  {label}
                </button>
              ))}
            </div>
            {view === "tree" && tree.sections.length > 1 && (
              <SectionSwitch
                sections={tree.sections}
                section={section}
                onPick={setSection}
              />
            )}
          </div>
          {view === "tree" && (
            <div className={s.treeActions}>
              <button
                type="button"
                onClick={() => setOpen(new Set(parents(nodes ?? [])))}
              >
                Expand all
              </button>
              <button type="button" onClick={() => setOpen(new Set())}>
                Collapse all
              </button>
            </div>
          )}
        </div>

        {view !== "tree" ? (
          <KernelShare
            tree={tree}
            kernels={kernels}
            share={share}
            pending={pending}
            byKind={view === "kinds"}
          />
        ) : (
          <div className={s.treeScroll}>
            {nodes ? (
              <TreeRows rows={rows} className={s.tree} label="Cost tree nodes" />
            ) : (
              <p role="status" className={s.treeLoading}>
                The tree appears once Live predict has timed a batch.
              </p>
            )}
          </div>
        )}
      </TreeLayout>
    </section>
  );
}

function SectionSwitch({ sections, section, onPick }) {
  return (
    <div className={detail.viewSwitch} role="radiogroup" aria-label="Section">
      {sections.map((item, index) => (
        <button
          key={item.section}
          type="button"
          role="radio"
          aria-checked={item === section}
          onClick={() => onPick(index)}
        >
          {sectionName(item.section)}
        </button>
      ))}
    </div>
  );
}

/* A node's time for one call, then its share of the section once its
   repeats are counted. */

function NodeTime({ node, pending }) {
  if (pending)
    return (
      <span className={s.nodeTime}>
        <Pending className={s.pendingNode} />
      </span>
    );
  return (
    <span
      className={s.nodeTime}
      title={`One call ${formatMs(node.ms)} ms; ${formatMs(node.total_ms)} ms with its repeats`}
    >
      {formatMs(node.ms)} ms <small>{formatPercent(node.pct)}</small>
    </span>
  );
}

function compositeRow(node, open, toggle, time) {
  return {
    id: node.node,
    kind: KIND[node.kind],
    depth: node.depth,
    name: breakable(node.label),
    toggle: node.parent ? () => toggle(node.node) : undefined,
    open: open.has(node.node),
    after: copiesText(node),
    meta: time,
  };
}

// Identical siblings show once; the row says how many it stands for.
const copiesText = (node) =>
  node.copies ? (
    <span
      className={s.ranks}
      title={
        node.avg_total_ms == null
          ? undefined
          : `The slowest of ${node.copies} is shown; their average is ${formatMs(node.avg_total_ms)} ms`
      }
    >
      ×{node.copies}
    </span>
  ) : null;

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

function leafRow(tree, section, kernels, node, time) {
  const slot = section.slots[node.slot];
  const kernel = kernels.get(slot.kernel);
  const href = kernelLink(kernel, slot, tree);
  const title = kernel?.title ?? slot.kernel;
  return {
    id: node.node,
    kind: "Leaf",
    depth: node.depth,
    name: breakable(node.label),
    after: copiesText(node),
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
      </>
    ),
  };
}
