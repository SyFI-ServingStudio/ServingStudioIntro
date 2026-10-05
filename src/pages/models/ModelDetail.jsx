import { ArrowLeft, CircleCheck, CircleDashed } from "lucide-react";
import {
  useCallback,
  useEffect,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";
import {
  loadCatalog,
  readQuery,
  setQuery,
  shortGpu,
  subscribeUrl,
} from "../kernels/kernelData";
import { Tag, ToggleTag } from "../kernels/Tag";
import {
  CONTRACTS,
  loadTree,
  memberFromQuery,
  missingText,
  modelHref,
  predictable,
  shortReference,
} from "./modelData";
import { openModel } from "./Models";
import { CostTreeExplorer } from "./CostTreeExplorer";
import { LivePredict } from "./LivePredict";
import { MemberPicker } from "./MemberPicker";
import detail from "../kernels/KernelDetail.module.css";
import picker from "../kernels/ConfigPicker.module.css";
import s from "./Models.module.css";

const search = () => window.location.search;

/* One checkpoint: pick one of its public presets and a value per axis of the
   preset, then read that member's cost tree and time a batch on it. */
export function ModelDetail({ catalog, checkpoint, preset }) {
  useSyncExternalStore(subscribeUrl, search);
  const title = checkpoint.name ?? checkpoint.checkpoint;
  useEffect(() => {
    document.title = `${title} | Models | ServingStudio`;
  }, [title]);
  const family = checkpoint.family;

  return (
    <div className={detail.page}>
      <header className={`wrap ${detail.header}`}>
        <a
          className={detail.back}
          href={modelHref({})}
          onClick={(event) => openModel(event, {})}
        >
          <ArrowLeft size={18} aria-hidden="true" />
          All models
        </a>
        <h1>{title}</h1>
        <p className={detail.identity}>
          <code>{checkpoint.checkpoint}</code>
          {family && <Tag type="family" value={family} />}
          <Tag type="category" value={CONTRACTS[preset.contract]}>
            {CONTRACTS[preset.contract] ?? preset.contract ?? "Does not build"}
          </Tag>
        </p>
      </header>

      <div className={`wrap ${s.detailBody}`}>
        <Explorer catalog={catalog} checkpoint={checkpoint} preset={preset} />
      </div>
    </div>
  );
}

const memberFromUrl = (preset, url) => memberFromQuery(preset, url, predictable);

function Explorer({ catalog, checkpoint, preset }) {
  const url = readQuery();
  const { member, unmatched } = memberFromUrl(preset, url);
  const pick = (params, id = preset.id) => setQuery({ preset: id, ...params });
  const tree = useTree(preset.id, member);
  const kernels = useKernelCatalog();
  // Live predict's time per node of the tree, once it has costed a batch,
  // the Analyzer's kernel ranking of that batch, and whether a newer batch
  // is being timed (its old times are stale).
  const [timed, setTimed] = useState({ times: null, share: null });
  const onTimes = useCallback(
    (times, share = null) => setTimed({ times, share }),
    [],
  );
  const [pending, setPending] = useState(false);
  // Which view of the costed batch the tree panel shows.
  const [view, setView] = useState("tree");
  // The batch a reader built, kept across members.
  const [batch, setBatch] = useState(null);

  const workload = preset.axes.find((axis) => axis.rows);
  const aside = useStickyTop();

  return (
    <>
      {unmatched && (
        <p className={s.notice} role="status">
          The link named values no configuration of this deployment has, so the
          closest one is shown.
        </p>
      )}
      <section
        className={`${detail.controls} ${s.picker}`}
        aria-label="Configuration"
      >
        <div
          className={`${picker.level} ${s.wrapLevel}`}
          role="group"
          aria-label="Deployment"
        >
          <span className={picker.levelName}>Deployment</span>
          <div className={picker.choices}>
            {checkpoint.presets.map((p) => (
              <ToggleTag
                key={p.id}
                type="choice"
                value={p.arch}
                pressed={p === preset}
                title={[p.arch, CONTRACTS[p.contract] ?? p.contract]
                  .filter(Boolean)
                  .join("\n")}
                onClick={() =>
                  pick(memberFromUrl(p, member.params).member.params, p.id)
                }
              >
                {p.arch_name ?? p.arch}
              </ToggleTag>
            ))}
          </div>
        </div>
        <div className={picker.level} role="group" aria-label="GPU">
          <span className={picker.levelName}>GPU</span>
          <div className={picker.choices}>
            <Tag type="gpu" value={shortGpu(preset.gpu)} />
          </div>
        </div>
        {preset.axes.length > 0 && (
          <div className={s.axes}>
            <MemberPicker
              axes={preset.axes}
              members={preset.members}
              current={member.params}
              onPick={(params) => pick(params)}
              blocked={blockedReason}
              valueTitle={(axis, value) =>
                axis.rows ? routingTitle(axis.rows[value]) : undefined
              }
            />
          </div>
        )}
        <MemberStatus member={member} workload={workload} />
      </section>

      <Legend />

      <CostTreeExplorer
        tree={tree.data}
        error={tree.error}
        kernels={kernels}
        times={timed.times}
        share={timed.share}
        pending={pending}
        view={view}
        onView={setView}
        aside={
          <aside className={s.aside} ref={aside}>
            <LivePredict
              preset={preset}
              member={member}
              tree={tree.data}
              caseFields={catalog.case_fields}
              batch={batch}
              setBatch={setBatch}
              onPick={(params) => pick(params)}
              onTimes={onTimes}
              onPending={setPending}
            />
          </aside>
        }
      />
    </>
  );
}

/* Why a member cannot be predicted, or null when it can. */
function blockedReason(member) {
  if (member.error) return `The simulator cannot build it: ${member.error}`;
  if (member.missing == null) return "Its measurements are not checked yet";
  if (Object.keys(member.missing).length)
    return `Cannot be timed yet; measurements missing for ${missingText(member.missing)}`;
  return null;
}

/* A workload row binds the routing and the capture it reads. */
const routingTitle = (row) =>
  row &&
  Object.entries(row)
    .map(([name, value]) => `${name}: ${shortReference(value)}`)
    .join("\n");

/* What the picked member is: its size, and whether Live predict can time it.
   One that cannot is still shown, with the reason. */
function MemberStatus({ member, workload }) {
  const reason = blockedReason(member);
  const routing = workload && workload.rows[member.params[workload.name]];
  return (
    <div className={s.memberStatus} data-state={reason ? "blocked" : "ready"}>
      {reason ? (
        <CircleDashed size={16} aria-hidden="true" />
      ) : (
        <CircleCheck size={16} aria-hidden="true" />
      )}
      <p>
        {member.error ? (
          <>The simulator could not build this configuration: {member.error}</>
        ) : (
          <>
            {member.leaves.toLocaleString("en-US")} kernel calls over{" "}
            {member.configs.toLocaleString("en-US")} kernel configs on{" "}
            {member.gpus_per_replica}{" "}
            {member.gpus_per_replica === 1 ? "GPU" : "GPUs"}.
            {reason && (
              <>
                {" "}
                <b>Cannot be timed yet:</b> measurements missing for{" "}
                {Object.entries(member.missing ?? {}).map(
                  ([kind, count], index) => (
                    <span key={kind}>
                      {index > 0 && ", "}
                      <code>{kind}</code> ({count.toLocaleString("en-US")})
                    </span>
                  ),
                )}
                {member.missing == null && "kernels not yet checked"}.
              </>
            )}
          </>
        )}
        {routing && (
          <>
            {" "}
            Routing: {routing.routing}
            {Object.entries(routing)
              .filter(([name]) => name !== "routing")
              .map(([name, value]) => (
                <span key={name} title={String(value)}>
                  , <code>{shortReference(value)}</code>
                </span>
              ))}
            .
          </>
        )}
      </p>
    </div>
  );
}

/* Where a sticky panel sticks: under the nav when it fits in the window,
   otherwise by its bottom, so its last line stays reachable. */
const NAV_CLEARANCE = 108;
const BOTTOM_GAP = 24;

function useStickyTop() {
  const release = useRef(null);
  // A callback ref: the panel remounts with the tree panel around it.
  return useCallback((node) => {
    release.current?.();
    release.current = null;
    if (!node) return;
    const place = () => {
      const top = Math.min(
        NAV_CLEARANCE,
        window.innerHeight - node.offsetHeight - BOTTOM_GAP,
      );
      node.style.setProperty("--stick-top", `${top}px`);
    };
    const observer = new ResizeObserver(place);
    observer.observe(node);
    window.addEventListener("resize", place);
    release.current = () => {
      observer.disconnect();
      window.removeEventListener("resize", place);
    };
  }, []);
}

/* The member's cost tree. */
function useTree(presetId, member) {
  const key = JSON.stringify([presetId, member.params]);
  const [state, setState] = useState({ key: null });
  useEffect(() => {
    let live = true;
    loadTree(presetId, member.params).then(
      (data) => live && setState({ key, data }),
      (error) => live && setState({ key, error }),
    );
    return () => {
      live = false;
    };
    // `key` stands for the preset and the member's params.
  }, [key]);
  return state.key === key ? state : {};
}

/* The kernel catalog names each leaf's kind and says which have a page. A
   catalog that fails to load leaves the leaves unlinked. */
function useKernelCatalog() {
  const [kernels, setKernels] = useState(null);
  useEffect(() => {
    loadCatalog().then(
      (catalog) =>
        setKernels(new Map(catalog.kernels.map((kernel) => [kernel.kind, kernel]))),
      () => setKernels(new Map()),
    );
  }, []);
  return kernels;
}

const LEGEND = [
  ["Sum", "Children run one after another. Their times add."],
  ["Max", "Children run in parallel, one per rank. The slowest sets the time."],
  ["Scale", "One child repeated n times, such as the decoder layers. Costed once."],
  ["Leaf", "One kernel call, timed from measured rows. Opens the kernel's page."],
];

function Legend() {
  return (
    <section className={s.legend} aria-labelledby="legend-title">
      <h2 id="legend-title">Reading the tree</h2>
      <dl>
        {LEGEND.map(([kind, text]) => (
          <div key={kind}>
            <dt>
              <b data-kind={kind} className={s.badge}>
                {kind}
              </b>
            </dt>
            <dd>{text}</dd>
          </div>
        ))}
      </dl>
      <p className={s.legendNote}>
        <CircleDashed size={14} aria-hidden="true" /> marks a configuration that
        cannot be timed yet, because some of its kernel measurements are missing.
      </p>
    </section>
  );
}
