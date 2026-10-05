import { ArrowRight } from "lucide-react";
import { useEffect } from "react";
import { PageHero } from "../../components/PageHero";
import { shortGpu } from "../kernels/kernelData";
import { openInPage, pageHref, setQuery, useQuery } from "../../url";
import { Tag, ToggleTag, tagColor } from "../kernels/Tag";
import { archLabel, CONTRACTS, predictable } from "./modelData";
import s from "./Models.module.css";

const OTHER = "Other";
const familyOf = (checkpoint) => checkpoint.family ?? OTHER;

/* Every checkpoint of the model catalog, grouped by family in the catalog's
   order. A checkpoint opens on its public presets: each one way of deploying
   it (an arch), with every supported value of its parameters. */
export function ModelCatalog({ catalog, unknownPreset }) {
  const query = useQuery();
  useEffect(() => {
    document.title = "Models | ServingStudio";
  }, []);
  const picked = query.family ?? "";
  const checkpoints = catalog.checkpoints;
  const families = [...new Set(checkpoints.map(familyOf))];
  const shown = checkpoints.filter((c) => !picked || familyOf(c) === picked);
  const groups = families
    .map((family) => ({
      family,
      checkpoints: shown.filter((c) => familyOf(c) === family),
    }))
    .filter((g) => g.checkpoints.length);

  return (
    <div className={s.page}>
      <PageHero
        compact
        title="Models"
        description="Each model runs as an execution graph, which the simulator costs as a tree of measured kernels. Pick a model and how it is deployed to see how an iteration's time is put together, then time a batch of your own."
        image="hero-models.webp"
      />

      <div className={`wrap ${s.catalog}`}>
        {unknownPreset && (
          <p className={s.notice} role="status">
            No public deployment is named <code>{unknownPreset}</code>. These are
            the ones that are.
          </p>
        )}
        <div className={s.filters} role="group" aria-label="Model family">
          <ToggleTag
            type="family"
            value="Any"
            pressed={!picked}
            count={checkpoints.length}
            onClick={() => setQuery({ ...query, family: "" })}
          >
            All
          </ToggleTag>
          {families.map((family) => (
            <ToggleTag
              key={family}
              type="family"
              value={family}
              pressed={picked === family}
              count={checkpoints.filter((c) => familyOf(c) === family).length}
              onClick={() =>
                setQuery({ ...query, family: picked === family ? "" : family })
              }
            />
          ))}
        </div>

        {groups.map(({ family, checkpoints: rows }) => (
          <section
            key={family}
            className={s.family}
            style={{
              "--family": tagColor("family", family) ?? "var(--c-line-600)",
            }}
            aria-labelledby={`family-${family}`}
          >
            <h2 id={`family-${family}`}>{family}</h2>
            <ul className={s.archList}>
              {rows.map((checkpoint) => (
                <li key={checkpoint.checkpoint}>
                  <CheckpointRow checkpoint={checkpoint} />
                </li>
              ))}
            </ul>
          </section>
        ))}
      </div>
    </div>
  );
}

function CheckpointRow({ checkpoint }) {
  const { presets } = checkpoint;
  const members = presets.flatMap((p) => p.members);
  const ready = members.filter(predictable).length;
  const gpus = [...new Set(presets.map((p) => p.gpu))];
  const body = (
    <>
      <span className={s.archMain}>
        <span className={s.archName}>
          {checkpoint.name ?? checkpoint.checkpoint}
        </span>
        <code className={s.archTag}>{checkpoint.checkpoint}</code>
      </span>
      <span className={s.archFacts}>
        <span className={s.fact}>
          <span className={s.factName}>Deployments</span>
          <span className={s.factValue}>
            {presets.length ? (
              presets.map((p) => (
                <span
                  key={p.id}
                  className={s.deploymentName}
                  title={[p.arch, CONTRACTS[p.contract]].filter(Boolean).join("\n")}
                >
                  {archLabel(p)}
                </span>
              ))
            ) : (
              <span className={s.contract}>None yet</span>
            )}
          </span>
        </span>
        {gpus.length > 0 && (
          <span className={s.fact}>
            <span className={s.factName}>GPU</span>
            <span className={s.factValue}>
              {gpus.map((gpu) => (
                <Tag key={gpu} type="gpu" value={shortGpu(gpu)} />
              ))}
            </span>
          </span>
        )}
        {members.length > 0 && (
          <span className={s.fact}>
            <span className={s.factName}>Configurations</span>
            <span className={s.factValue}>
              <span className={s.figure}>{members.length}</span>
              <span className={s.contract}>{ready} predictable</span>
            </span>
          </span>
        )}
      </span>
    </>
  );
  if (!presets.length) return <div className={s.archRow}>{body}</div>;
  const params = { preset: presets[0].id };
  return (
    <a
      className={s.archRow}
      href={pageHref("models", params)}
      onClick={(event) => openInPage(event, params)}
    >
      {body}
      <ArrowRight className={s.archArrow} size={20} aria-hidden="true" />
    </a>
  );
}
