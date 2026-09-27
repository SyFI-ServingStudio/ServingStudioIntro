import { ArrowUpRight, CircleCheck, CircleDashed, CircleDot } from "lucide-react";
import { Fragment, useEffect, useMemo, useState } from "react";
import { TreeRows, backendLabels } from "../../components/CostTree";
import { formatValue } from "../kernels/kernelData";
import {
  allIds,
  coverage,
  initiallyOpen,
  kernelLink,
  paramValue,
  prepareTree,
  routingText,
  sourceText,
  visible,
} from "./modelData";
import detail from "../kernels/KernelDetail.module.css";
import s from "./Models.module.css";

const KIND = { sum: "Sum", max: "Max", scale: "Scale", leaf: "Leaf" };
// Past this many, a leaf's shape reads "…"; the kernel page has the rest.
const SHAPE_ARGS = 5;

/* One parameter set's cost tree, as ModelDetail loaded it: how the simulator
   puts an iteration's time together from kernel calls. It carries no times; a
   leaf says which kernel it calls, the shape it asks for and whether
   profile.db measures that shape. */
export function CostTreeExplorer({ tree, error }) {
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
  return <Tree key={JSON.stringify(tree.run?.query ?? tree.query)} tree={tree} />;
}

const HIDDEN = ["num_layers", "sim_num_layers"];

/* "Built as in <source>: <params>." for a tree built as a registered run,
   then the other sources that recorded the same run. */
function BuiltAs({ run, defaults }) {
  // Sources named alike (two prediction configs of one name on another
  // machine) read once, with how many there are.
  const named = new Map();
  for (const source of run.sources) {
    const text = sourceText(source);
    named.set(text, { source, count: (named.get(text)?.count ?? 0) + 1 });
  }
  const [first, ...others] = [...named.values()];
  const routingKeys = new Set(
    run.pickers.find((p) => p.name === "routing")?.keys ?? [],
  );
  const routing = Object.fromEntries(
    Object.entries(run.params).filter(([name]) => routingKeys.has(name)),
  );
  const params = Object.entries(run.params).filter(
    ([name]) => !routingKeys.has(name) && !HIDDEN.includes(name),
  );
  const items = [
    ...params.map(([name, value]) => [name, paramValue(value), name in defaults]),
    ...(routingKeys.size
      ? [["routing", routingText(routing, run.routing), false]]
      : []),
  ].sort(([a], [b]) => order(a) - order(b));
  return (
    <p className={s.defaults}>
      Built as in <SourceName {...first} />:{" "}
      {items.map(([name, text, isDefault], index) => (
        <span
          key={name}
          title={name === "routing" ? run.routing?.label : undefined}
        >
          {index > 0 && ", "}
          <code>{name}</code> <span className={s.runValue}>{text}</span>
          {isDefault && <span className={s.soft}> (default)</span>}
        </span>
      ))}
      .
      {others.length > 0 && (
        <>
          {" "}
          The same run was also recorded by{" "}
          {others.map((entry, index) => (
            <Fragment key={entry.source.id}>
              {index > 0 && (index === others.length - 1 ? " and " : ", ")}
              <SourceName {...entry} />
            </Fragment>
          ))}
          .
        </>
      )}
      {run.skipped.length > 0 && (
        <>
          {" "}
          <span className={s.soft}>
            {run.skipped.length} more registered{" "}
            {run.skipped.length === 1 ? "run is" : "runs are"} not offered:{" "}
            {[...new Set(run.skipped.map((skip) => skip.error))].join("; ")}.
          </span>
        </>
      )}
    </p>
  );
}

function SourceName({ source, count }) {
  return (
    <span
      className={s.source}
      title={source.cases?.length ? source.cases.join(", ") : undefined}
    >
      {sourceText(source)}
      {count > 1 && <span className={s.soft}> ({count} of them)</span>}
    </span>
  );
}

// The params a reader looks for first lead the line.
const LEAD = ["max_model_len", "routing", "draft_tokens", "mtp_mode"];
const order = (name) => (LEAD.includes(name) ? LEAD.indexOf(name) : LEAD.length);

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
  const registry = tree.run?.basis === "registry";
  const defaults = Object.entries(tree.defaults).filter(
    ([name]) => !HIDDEN.includes(name),
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
        {registry && <BuiltAs run={tree.run} defaults={tree.defaults} />}
        {!registry && (defaults.length > 0 || predicting.length > 0) && (
          <p className={s.defaults}>
            No registered run matches this set, so it is built at the schema
            defaults.{" "}
            {defaults.length > 0 && (
              <>
                The other params:{" "}
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
