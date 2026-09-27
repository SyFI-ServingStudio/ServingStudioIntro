import { ArrowLeft } from "lucide-react";
import { useEffect, useMemo, useState, useSyncExternalStore } from "react";
import { readQuery, setQuery, shortGpu, subscribeUrl } from "../kernels/kernelData";
import { Tag, ToggleTag } from "../kernels/Tag";
import { CONTRACTS, loadArch, memberFromUrl, paramValue } from "./modelData";
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

      <div className={s.treeLayout}>
        <CostTreeExplorer
          key={JSON.stringify(current.member.query)}
          arch={arch}
          member={current.member}
        />
        <aside className={s.aside}>
          <Legend />
          <PredictPlaceholder />
        </aside>
      </div>
    </>
  );
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
    </section>
  );
}

// "a", "a and b", "a, b and c".
const listed = (names) =>
  names.length < 3
    ? names.join(" and ")
    : `${names.slice(0, -1).join(", ")} and ${names.at(-1)}`;
