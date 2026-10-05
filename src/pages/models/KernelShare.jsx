import { ArrowUpRight, ChevronRight } from "lucide-react";
import { useState } from "react";
import { formatMs, kernelLink } from "./modelData";
import { Pending, breakable } from "./TreeParts";
import s from "./Models.module.css";

// Rows shown before "Show all".
const FIRST = 25;
// Placeholder rows while the batch is timed.
const PLACEHOLDERS = 8;

/* The Analyzer's ranking of the batch Live predict last timed (`share`, the
   `kernel_time_share` /predict answers with), not recomputed here. By kernel:
   every kernel call site, its layers summed, largest first; under a Max only
   the slowest branch counts, so the shares add up to the whole time. By
   kernel type (`byKind`): the Analyzer's `kinds`, each opening on its call
   sites, which are the segments of that kind in the Analyzer's order. A call
   site is a slot name of the member's tree, which gives its kernel page. */
export function KernelShare({ tree, kernels, share, pending, byKind }) {
  if (pending) return <Placeholders />;
  if (!share)
    return (
      <p role="status" className={s.treeLoading}>
        The ranking appears once Live predict has timed a batch.
      </p>
    );
  if (!share.segments)
    return (
      <p role="status" className={s.treeLoading}>
        The Analyzer has no ranking for this batch.
      </p>
    );
  return byKind ? (
    <KindRanking tree={tree} kernels={kernels} share={share} />
  ) : (
    <SiteRanking tree={tree} kernels={kernels} share={share} />
  );
}

function Placeholders() {
  return (
    <ol className={s.share} aria-label="Kernels by time" aria-busy="true">
      {Array.from({ length: PLACEHOLDERS }, (_, index) => (
        <li key={index} className={s.shareRow}>
          <span className={s.shareRank}>{index + 1}</span>
          <span className={s.shareName}>
            <Pending className={s.pendingName} />
          </span>
          <span className={s.shareTime}>
            <Pending className={s.pendingNode} />
          </span>
        </li>
      ))}
    </ol>
  );
}

function SiteRanking({ tree, kernels, share }) {
  const [all, setAll] = useState(false);
  const slots = slotsByName(tree);
  const top = share.segments[0]?.kernel_time_ms ?? 0;
  const shown = all ? share.segments : share.segments.slice(0, FIRST);
  return (
    <>
      <ol className={s.share} aria-label="Kernels by time">
        {shown.map((segment, index) => (
          <SiteRow
            key={segment.position}
            rank={index + 1}
            segment={segment}
            top={top}
            whole={share.kernel_time_ms}
            slot={slots.get(segment.position)}
            kernels={kernels}
            tree={tree}
          />
        ))}
      </ol>
      {share.segments.length > FIRST && (
        <button
          type="button"
          className={`${s.addButton} ${s.shareMore}`}
          onClick={() => setAll(!all)}
        >
          {all
            ? `Show the first ${FIRST}`
            : `Show all ${share.segments.length.toLocaleString("en-US")}`}
        </button>
      )}
    </>
  );
}

function KindRanking({ tree, kernels, share }) {
  const [open, setOpen] = useState(() => new Set());
  const toggle = (kind) =>
    setOpen((current) => {
      const next = new Set(current);
      if (next.has(kind)) next.delete(kind);
      else next.add(kind);
      return next;
    });
  const slots = slotsByName(tree);
  const top = share.kinds[0]?.kernel_time_ms ?? 0;
  return (
    <ol className={s.share} aria-label="Kernel types by time">
      {share.kinds.map((item, index) => {
        const kernel = kernels.get(item.kind);
        const expanded = open.has(item.kind);
        const sites = expanded
          ? share.segments.filter((segment) => segment.kind === item.kind)
          : [];
        return (
          <li key={item.kind} className={s.shareGroup}>
            <div className={s.shareRow}>
              <span className={s.shareRank}>{index + 1}</span>
              <span className={s.shareName}>
                <button
                  type="button"
                  className={s.shareToggle}
                  aria-expanded={expanded}
                  onClick={() => toggle(item.kind)}
                >
                  <ChevronRight size={16} aria-hidden="true" />
                  {kernel?.title ?? item.kind}
                </button>
                <span className={s.shareMeta}>
                  <code>{item.kind}</code> ·{" "}
                  {item.positions.toLocaleString("en-US")}{" "}
                  {item.positions === 1 ? "call site" : "call sites"}
                </span>
              </span>
              <Time item={item} whole={share.kernel_time_ms} />
              <Bar value={item.kernel_time_ms} top={top} />
            </div>
            {expanded && (
              <ol className={s.shareSites} aria-label={`${item.kind} call sites`}>
                {sites.map((segment, siteIndex) => (
                  <SiteRow
                    key={segment.position}
                    rank={`${index + 1}.${siteIndex + 1}`}
                    segment={segment}
                    top={top}
                    whole={share.kernel_time_ms}
                    slot={slots.get(segment.position)}
                    kernels={kernels}
                    tree={tree}
                    plain
                  />
                ))}
              </ol>
            )}
          </li>
        );
      })}
    </ol>
  );
}

/* One call site: its slot name and kernel page (left out under its own
   kernel type, `plain`), its time and share, and a bar against `top`. */
function SiteRow({ rank, segment, top, whole, slot, kernels, tree, plain }) {
  const kernel = kernels.get(segment.kind);
  const href = slot && kernelLink(kernel, slot, tree);
  const title = kernel?.title ?? segment.kind;
  return (
    <li className={s.shareRow}>
      <span className={s.shareRank}>{rank}</span>
      <span className={s.shareName}>
        <strong>{breakable(segment.position)}</strong>
        {href ? (
          <a className={s.kernelLink} href={href}>
            {plain ? "Kernel page" : title}
            <ArrowUpRight size={15} aria-hidden="true" />
          </a>
        ) : (
          !plain && <span className={s.kernelPlain}>{title}</span>
        )}
      </span>
      <Time item={segment} whole={whole} />
      <Bar value={segment.kernel_time_ms} top={top} />
    </li>
  );
}

function Time({ item, whole }) {
  return (
    <span
      className={s.shareTime}
      title={`${item.kernel_time_ms} ms of ${whole} ms`}
    >
      {formatMs(item.kernel_time_ms)} ms <small>{percent(item.share_pct)}</small>
    </span>
  );
}

function Bar({ value, top }) {
  return (
    <span
      className={s.shareBar}
      style={{ "--share": top ? value / top : 0 }}
      aria-hidden="true"
    />
  );
}

const percent = (pct) => (pct > 0 && pct < 0.1 ? "<0.1%" : `${pct.toFixed(1)}%`);

/* Each slot of the member's tree by name; a call site's slots share one. */
function slotsByName(tree) {
  const slots = new Map();
  for (const section of tree.sections)
    for (const slot of section.slots)
      if (!slots.has(slot.name)) slots.set(slot.name, slot);
  return slots;
}
