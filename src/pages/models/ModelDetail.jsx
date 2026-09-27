import { ArrowLeft } from "lucide-react";
import { useEffect, useMemo, useState, useSyncExternalStore } from "react";
import { readQuery, setQuery, shortGpu, subscribeUrl } from "../kernels/kernelData";
import { Tag, ToggleTag } from "../kernels/Tag";
import {
  CONTRACTS,
  loadArch,
  loadCostTree,
  memberFromUrl,
  paramValue,
  routingText,
} from "./modelData";
import { modelHref, openModel } from "./Models";
import { CostTreeExplorer } from "./CostTreeExplorer";
import { PredictPlaceholder } from "./PredictPlaceholder";
import { Prose } from "./Prose";
import detail from "../kernels/KernelDetail.module.css";
import picker from "../kernels/ConfigPicker.module.css";
import s from "./Models.module.css";

const search = () => window.location.search;

/* One arch: pick a model, a GPU and one of the parameter sets its
   #[supported] rows allow, then read that set's cost tree. */
export function ModelDetail({ catalog, entry }) {
  useSyncExternalStore(subscribeUrl, search);
  const [arch, setArch] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    loadArch(entry.arch).then(setArch, setError);
  }, [entry.arch]);
  const title = entry.name ?? entry.arch;
  useEffect(() => {
    document.title = `${title} | Models | ServingStudio`;
  }, [title]);
  const family = entry.families[0];

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
          <code>{entry.arch}</code>
          {family && <Tag type="family" value={family} />}
          <Tag type="category" value={CONTRACTS[entry.contract]}>
            {CONTRACTS[entry.contract] ?? entry.contract}
          </Tag>
        </p>
        {entry.summary && (
          <p className={`${detail.summary} ${s.prose}`}>
            <Prose text={entry.summary} />
          </p>
        )}
      </header>

      <div className={`wrap ${s.detailBody}`}>
        {error ? (
          <p role="alert">
            This model&apos;s parameters did not load ({error.message}). Reload the
            page to try again.
          </p>
        ) : !arch ? (
          <p role="status" className={detail.loading}>
            Loading parameter sets…
          </p>
        ) : arch.param_sets.length === 0 ? (
          <p className={s.notice}>
            No #[supported] row names a parameter set for this arch yet, so there is
            no cost tree to show.
          </p>
        ) : (
          <Explorer arch={arch} catalog={catalog} family={family} />
        )}
      </div>
    </div>
  );
}

/* The params that tell `members` apart; all of them when nothing does. */
function distinguishing(members, names) {
  const differ = names.filter(
    (name) =>
      new Set(members.map(({ member }) => String(member.query[name]))).size > 1,
  );
  return differ.length ? differ : names;
}

function Explorer({ arch, family }) {
  const url = readQuery();
  const members = useMemo(
    () =>
      arch.param_sets.flatMap((set) =>
        set.members.map((member) => ({ set, member })),
      ),
    [arch],
  );
  const named = memberFromUrl(arch, url);
  const asked = arch.query.some((name) => url[name] != null);
  // A link that names no supported set opens on the first one the model and
  // GPU it names allow, and says so.
  const current =
    named ??
    members.find(
      ({ member }) =>
        (!url.model || member.query.model === url.model) &&
        (!url.gpu || member.query.gpu === url.gpu),
    ) ??
    members[0];
  const { model, gpu } = current.member.query;
  const pick = (next) => setQuery({ arch: arch.arch, ...next.member.query });
  // Every other key in the link picks one of the set's registered runs.
  const runQuery = Object.fromEntries(
    Object.entries(url).filter(
      ([key]) => key !== "arch" && !arch.query.includes(key),
    ),
  );
  const tree = useTree(arch.arch, current.member.query, runQuery);

  const models = arch.models.filter((m) =>
    members.some(({ member }) => member.query.model === m.model_config),
  );
  const gpus = [
    ...new Set(
      members
        .filter(({ member }) => member.query.model === model)
        .map((m) => m.member.query.gpu),
    ),
  ];
  const here = members.filter(
    ({ member }) => member.query.model === model && member.query.gpu === gpu,
  );
  const params = arch.query.filter((name) => name !== "gpu" && name !== "model");
  const shown = distinguishing(here, params);
  const docs = new Map(arch.params.map((p) => [p.name, p]));

  return (
    <>
      {asked && !named && (
        <p className={s.notice} role="status">
          The link named a parameter set this arch does not support, so the first
          supported one is shown.
        </p>
      )}
      <section
        className={`${detail.controls} ${s.picker}`}
        aria-label="Parameter set"
      >
        <div className={picker.level}>
          <span className={picker.levelName}>Model</span>
          <div className={picker.choices}>
            {models.map((m) => (
              <ToggleTag
                key={m.model_config}
                type="family"
                value={family}
                pressed={m.model_config === model}
                title={m.checkpoint ?? undefined}
                onClick={() =>
                  pick(
                    members.find(
                      ({ member }) => member.query.model === m.model_config,
                    ),
                  )
                }
              >
                {m.name ?? m.model_config}
              </ToggleTag>
            ))}
          </div>
        </div>
        <div className={picker.level}>
          <span className={picker.levelName}>GPU</span>
          <div className={picker.choices}>
            {gpus.map((name) => (
              <ToggleTag
                key={name}
                type="gpu"
                value={shortGpu(name)}
                pressed={name === gpu}
                onClick={() =>
                  pick(
                    members.find(
                      ({ member }) =>
                        member.query.model === model && member.query.gpu === name,
                    ),
                  )
                }
              />
            ))}
          </div>
        </div>
        <div className={`${picker.level} ${s.setLevel}`}>
          <span className={picker.levelName}>
            Parameter set
            <span className={picker.levelCount}>
              {here.length} supported
              {shown.length < params.length && `, told apart by ${listed(shown)}`}
            </span>
          </span>
          <div className={picker.choices}>
            {here.map((choice) => (
              <ToggleTag
                key={choice.member.label}
                type="choice"
                value={choice.member.label}
                pressed={choice.member === current.member}
                title={choice.member.label}
                onClick={() => pick(choice)}
              >
                <span className={s.setChip}>
                  {shown.map((name) => (
                    <span key={name}>
                      {name} <b>{paramValue(choice.member.query[name])}</b>
                    </span>
                  ))}
                  {!shown.length && "Default parameters"}
                </span>
              </ToggleTag>
            ))}
          </div>
        </div>
        {shown.length < params.length && (
          <p className={s.fixed}>
            Same in every set here:{" "}
            {params
              .filter((name) => !shown.includes(name))
              .map((name, index) => (
                <span key={name} title={docs.get(name)?.description || undefined}>
                  {index > 0 && ", "}
                  <code>{name}</code> {paramValue(current.member.query[name])}
                </span>
              ))}
          </p>
        )}
      </section>

      {tree.stale && (
        <p className={s.notice} role="status">
          No registered run of this set used the run configuration the link named,
          so the best-measured run is shown.
        </p>
      )}
      {tree.data?.run?.basis === "registry" && (
        <RunPicker
          run={tree.data.run}
          onPick={(params) =>
            setQuery({ arch: arch.arch, ...current.member.query, ...params })
          }
        />
      )}

      <div className={s.treeLayout}>
        <CostTreeExplorer tree={tree.data} error={tree.error} />
        <aside className={s.aside}>
          <Legend />
          <PredictPlaceholder />
        </aside>
      </div>
    </>
  );
}

/* The set's cost tree for the run the link names. A link naming run params
   no registered run used opens on the best-measured run instead, and says so. */
function useTree(archName, setQuery, runQuery) {
  const key = JSON.stringify([archName, setQuery, runQuery]);
  const [state, setState] = useState({ key: null });
  useEffect(() => {
    let live = true;
    const hasRun = Object.keys(runQuery).length > 0;
    loadCostTree(archName, { ...setQuery, ...runQuery })
      .catch((error) => {
        if (!hasRun || error.status !== 404) throw error;
        return loadCostTree(archName, setQuery).then((data) => ({
          data,
          stale: true,
        }));
      })
      .then(
        (result) =>
          live &&
          setState(result.stale ? { key, ...result } : { key, data: result }),
        (error) => live && setState({ key, error }),
      );
    return () => {
      live = false;
    };
    // `key` stands for the three inputs.
  }, [key]);
  return state.key === key ? state : {};
}

/* The params the set leaves open, as its registered runs set them: one picker
   per param that the runs set differently, the rest as text. A value no run
   pairs with the current others is dimmed; picking it moves to the best run
   that has it. */
function RunPicker({ run, onPick }) {
  const open = run.pickers.filter((p) => !p.fixed);
  const fixed = run.pickers.filter((p) => p.fixed);
  const pickOption = (picker, option) => {
    if (!option.compatible) return onPick(option.value);
    const rest = Object.fromEntries(
      Object.entries(run.params).filter(([name]) => !picker.keys.includes(name)),
    );
    onPick({ ...rest, ...option.value });
  };
  return (
    <section
      className={`${detail.controls} ${s.picker}`}
      aria-label="Run configuration"
    >
      <div className={`${picker.level} ${s.setLevel}`}>
        <span className={picker.levelName}>
          Run configuration
          <span className={picker.levelCount}>
            {run.combinations} registered{" "}
            {run.combinations === 1 ? "run uses" : "runs use"} this set
          </span>
        </span>
      </div>
      {open.map((p) => (
        <div key={p.name} className={`${picker.level} ${s.runLevel}`}>
          <span className={picker.levelName}>
            <code className={s.runName}>{p.name}</code>
          </span>
          <div className={picker.choices}>
            {p.options.map((option) => (
              <ToggleTag
                key={JSON.stringify(option.value)}
                type="choice"
                value={optionText(p, option)}
                pressed={option.selected}
                faint={!option.compatible}
                title={optionTitle(option)}
                onClick={() => pickOption(p, option)}
              >
                <span className={s.runChip}>
                  {optionText(p, option)}
                  <small>
                    {option.counts.measured}/{option.counts.configs}
                  </small>
                </span>
              </ToggleTag>
            ))}
          </div>
        </div>
      ))}
      {fixed.length > 0 && (
        <p className={s.fixed}>
          Same in every run here:{" "}
          {fixed.map((p, index) => (
            <span key={p.name}>
              {index > 0 && ", "}
              <code>{p.name}</code> {optionText(p, p.options[0])}
            </span>
          ))}
        </p>
      )}
    </section>
  );
}

const optionText = (p, option) =>
  p.name === "routing"
    ? routingText(option.value, option.routing)
    : paramValue(option.value[p.name]);

const optionTitle = (option) =>
  `${option.counts.measured} of ${option.counts.configs} shapes measured` +
  (option.compatible ? "" : "; picking it changes the other run params too");

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
    </section>
  );
}

// "a", "a and b", "a, b and c".
const listed = (names) =>
  names.length < 3
    ? names.join(" and ")
    : `${names.slice(0, -1).join(", ")} and ${names.at(-1)}`;
