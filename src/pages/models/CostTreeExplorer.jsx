import { ArrowUpRight, CircleCheck, CircleDashed, CircleDot } from "lucide-react";
import { Fragment, useEffect, useMemo, useState } from "react";
import { TreeRows, backendLabels } from "../../components/CostTree";
import { formatValue } from "../kernels/kernelData";
import {
  allIds,
  coverage,
  initiallyOpen,
  kernelLink,
  loadCostTree,
  paramValue,
  prepareTree,
  visible,
} from "./modelData";
import detail from "../kernels/KernelDetail.module.css";
import s from "./Models.module.css";

const KIND = { sum: "Sum", max: "Max", scale: "Scale", leaf: "Leaf" };
// Past this many, a leaf's shape reads "…"; the kernel page has the rest.
const SHAPE_ARGS = 5;

/* One parameter set's cost tree, read from the API: how the simulator puts an
   iteration's time together from kernel calls. It carries no times; a leaf
   says which kernel it calls, the shape it asks for and whether profile.db
   measures that shape. */
export function CostTreeExplorer({ arch, member }) {
  const [tree, setTree] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    loadCostTree(arch.arch, member.query).then(setTree, setError);
  }, [arch.arch, member.query]);

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
  return <Tree tree={tree} />;
}

function Tree({ tree }) {
  const [sectionIndex, setSection] = useState(0);
  const section = tree.sections[Math.min(sectionIndex, tree.sections.length - 1)];
  const root = useMemo(() => prepareTree(section.root), [section]);
  const [open, setOpen] = useState(() => initiallyOpen(root));
  useEffect(() => setOpen(initiallyOpen(root)), [root]);
  const toggle = (id) =>
    setOpen((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const rows = [...visible(root, open)].map((node) =>
    node.kind === "leaf" ? leafRow(tree, node) : compositeRow(node, open, toggle),
  );
  const { counts } = tree;
  const defaults = Object.entries(tree.defaults).filter(
    ([name]) => !["num_layers", "sim_num_layers"].includes(name),
  );
  // Params the schema marks as traffic (MoE routing): no default stands in.
  const predicting = tree.set_when_predicting ?? [];

  return (
    <section className={s.treePanel} aria-labelledby="tree-title">
      <div className={s.treeHead}>
        <h2 id="tree-title">Cost tree</h2>
        <dl className={s.stats}>
          <div>
            <dt>Kernel calls</dt>
            <dd>{counts.leaves.toLocaleString("en-US")}</dd>
          </div>
          <div>
            <dt>Distinct shapes</dt>
            <dd>{counts.configs.toLocaleString("en-US")}</dd>
          </div>
          <div>
            <dt>Shapes measured</dt>
            <dd>
              {counts.measured}
              <span> of {counts.configs}</span>
            </dd>
          </div>
          <div>
            <dt>GPUs per replica</dt>
            <dd>{tree.gpus_per_replica ?? "unknown"}</dd>
          </div>
        </dl>
        {(defaults.length > 0 || predicting.length > 0) && (
          <p className={s.defaults}>
            {defaults.length > 0 && (
              <>
                Built with the other params at their defaults:{" "}
                {defaults.map(([name, value], index) => (
                  <span key={name}>
                    {index > 0 && ", "}
                    <code>{name}</code> {paramValue(value)}
                  </span>
                ))}
                .{predicting.length > 0 && " "}
              </>
            )}
            {predicting.length > 0 && (
              <>
                {predicting.map((name, index) => (
                  <span key={name}>
                    {index > 0 &&
                      (index === predicting.length - 1 ? " and " : ", ")}
                    <code>{name}</code>
                  </span>
                ))}{" "}
                {predicting.length > 1 ? "are" : "is"} set when predicting.
              </>
            )}
          </p>
        )}
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

function compositeRow(node, open, toggle) {
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
      <span className={s.nodeCount}>
        {node.leaves.toLocaleString("en-US")} {node.leaves === 1 ? "call" : "calls"}
      </span>
    ),
  };
}

function shapeText(config) {
  if (!config) return null;
  const entries = Object.entries(config.args);
  const parts = entries
    .slice(0, SHAPE_ARGS)
    .map(([name, value]) => `${name} ${formatValue(value)}`);
  if (entries.length > SHAPE_ARGS) parts.push("…");
  return parts.join(", ");
}

const COVERAGE = {
  measured: [CircleCheck, "Measured"],
  partial: [CircleDot, "Partly measured"],
  unmeasured: [CircleDashed, "Not yet measured"],
  unregistered: [CircleDashed, "Not yet measured"],
};

function leafRow(tree, node) {
  const { slot } = node;
  const config = tree.configs[slot.config_hash];
  const kernel = tree.kernels[slot.kind];
  const href = kernelLink(tree, slot);
  const cov = coverage(config);
  const [Icon, text] = COVERAGE[cov.state];
  const detailText =
    cov.state === "partial"
      ? `${text}, ${cov.best} of ${cov.runnable} grid cells`
      : cov.state === "unregistered"
        ? `${text}: no simulator grid is registered for this shape`
        : text;
  const title = kernel?.title ?? slot.kind;
  return {
    id: node.id,
    kind: "Leaf",
    depth: node.depth,
    name: breakable(node.name),
    note: shapeText(config),
    meta: (
      <>
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
        <span className={s.coverage} data-state={cov.state} title={detailText}>
          <Icon size={14} aria-hidden="true" />
          {cov.state === "partial" ? `${cov.best}/${cov.runnable}` : text}
          <span className={s.backends}>
            {slot.backends.map((b) => backendLabels[b] || b).join(", ")}
          </span>
        </span>
      </>
    ),
  };
}
