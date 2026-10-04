import { ArrowLeft, CircleCheck, CircleDashed } from "lucide-react";
import { useEffect, useState, useSyncExternalStore } from "react";
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
  memberMatches,
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

/* The member the link names; one it does not fully name opens on the first
   predictable member that has the values it gives, and says so. */
function memberFromUrl(preset, url) {
  const names = preset.axes.map((axis) => axis.name);
  const named = preset.members.find((m) => memberMatches(m, url, names));
  if (named) return { member: named };
  const given = names.filter((name) => url[name] != null);
  const fits = preset.members.filter((m) => memberMatches(m, url, given));
  const pool = fits.length ? fits : preset.members;
  return {
    member: pool.find(predictable) ?? pool[0],
    unmatched: given.length > 0,
  };
}

function Explorer({ catalog, checkpoint, preset }) {
  const url = readQuery();
  const { member, unmatched } = memberFromUrl(preset, url);
  const pick = (params, id = preset.id) => setQuery({ preset: id, ...params });
  const tree = useTree(preset.id, member);
  const kernels = useKernelCatalog();
  // Live predict's time per node of the tree, once it has costed a batch.
  const [times, setTimes] = useState(null);
  // The batch a reader built, kept across members.
  const [batch, setBatch] = useState(null);

  const siblings = catalog.checkpoints.filter(
    (c) => c.family === checkpoint.family && c.presets.length,
  );
  // Another checkpoint opens on the same arch when it has one.
  const openCheckpoint = (other) => {
    const same = other.presets.find((p) => p.arch === preset.arch);
    const target = same ?? other.presets[0];
    pick(memberFromUrl(target, member.params).member.params, target.id);
  };
  const workload = preset.axes.find((axis) => axis.rows);

  return (
    <>
      {unmatched && (
        <p className={s.notice} role="status">
          The link named values no parameter set of this deployment has, so the
          closest one is shown.
        </p>
      )}
      <section
        className={`${detail.controls} ${s.picker}`}
        aria-label="Parameter set"
      >
        {siblings.length > 1 && (
          <div
            className={`${picker.level} ${s.wrapLevel}`}
            role="group"
            aria-label="Model"
          >
            <span className={picker.levelName}>Model</span>
            <div className={picker.choices}>
              {siblings.map((c) => (
                <ToggleTag
                  key={c.checkpoint}
                  type="family"
                  value={checkpoint.family}
                  pressed={c === checkpoint}
                  title={c.checkpoint}
                  onClick={() => openCheckpoint(c)}
                >
                  {c.name ?? c.checkpoint}
                </ToggleTag>
              ))}
            </div>
          </div>
        )}
        <div
          className={`${picker.level} ${s.wrapLevel}`}
          role="group"
          aria-label="Deployment"
        >
          <span className={picker.levelName}>
            Deployment
            <span className={picker.levelCount}>
              {checkpoint.presets.length === 1
                ? "one public preset"
                : `${checkpoint.presets.length} public presets`}
            </span>
          </span>
          <div className={picker.choices}>
            {checkpoint.presets.map((p) => (
              <ToggleTag
                key={p.id}
                type="choice"
                value={p.arch}
                pressed={p === preset}
                title={`${CONTRACTS[p.contract] ?? p.contract ?? ""}\npresets/public/${p.id}.yaml`}
                onClick={() =>
                  pick(memberFromUrl(p, member.params).member.params, p.id)
                }
              >
                <span className={s.axisName}>{p.arch}</span>
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

      <div className={s.treeLayout}>
        <CostTreeExplorer
          tree={tree.data}
          error={tree.error}
          kernels={kernels}
          times={times}
        />
        <aside className={s.aside}>
          <Legend />
        </aside>
      </div>

      <LivePredict
        preset={preset}
        member={member}
        tree={tree.data}
        caseFields={catalog.case_fields}
        batch={batch}
        setBatch={setBatch}
        onPick={(params) => pick(params)}
        onTimes={setTimes}
      />
    </>
  );
}

/* Why a member cannot be predicted, or null when it can. */
function blockedReason(member) {
  if (member.error) return `Does not build: ${member.error}`;
  if (member.missing == null) return "Not checked against profile.db";
  if (Object.keys(member.missing).length)
    return `Not predictable: profile.db lacks rows of ${missingText(member.missing)}`;
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
          <>The simulator could not build this parameter set: {member.error}</>
        ) : (
          <>
            {member.leaves.toLocaleString("en-US")} kernel calls over{" "}
            {member.configs.toLocaleString("en-US")} kernel configs on{" "}
            {member.gpus_per_replica}{" "}
            {member.gpus_per_replica === 1 ? "GPU" : "GPUs"}.{" "}
            {reason ? (
              <>
                <b>Not predictable:</b> profile.db lacks measured rows of{" "}
                {Object.entries(member.missing ?? {}).map(
                  ([kind, count], index) => (
                    <span key={kind}>
                      {index > 0 && ", "}
                      <code>{kind}</code> ({count.toLocaleString("en-US")})
                    </span>
                  ),
                )}
                {member.missing == null && "kinds not yet checked"}.
              </>
            ) : (
              "Every row its kernels read is measured, so Live predict below can time it."
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
        <CircleDashed size={14} aria-hidden="true" /> marks a parameter set, or a
        kernel config in the tree, that profile.db lacks rows for.
      </p>
    </section>
  );
}
