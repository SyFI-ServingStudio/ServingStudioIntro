import { ArrowRight } from "lucide-react";
import { useEffect, useSyncExternalStore } from "react";
import { readQuery, setQuery, shortGpu, subscribeUrl } from "../kernels/kernelData";
import { Tag, ToggleTag, tagColor } from "../kernels/Tag";
import { CONTRACTS } from "./modelData";
import { Prose } from "./Prose";
import { modelHref, openModel } from "./Models";
import detail from "../kernels/KernelDetail.module.css";
import s from "./Models.module.css";

const search = () => window.location.search;
const OTHER = "Other";
const familyOf = (arch) => arch.families[0] ?? OTHER;

/* Every arch the simulator supports, grouped by model family in the model
   catalog's order. An arch is a model's execution graph under one way of
   deploying it; each opens on its supported parameter sets. */
export function ModelCatalog({ catalog, unknownArch }) {
  useSyncExternalStore(subscribeUrl, search);
  useEffect(() => {
    document.title = "Models | ServingStudio";
  }, []);
  const query = readQuery();
  const picked = query.family ?? "";
  const models = new Map(catalog.models.map((m) => [m.model_config, m]));
  const families = [...new Set(catalog.archs.map(familyOf))];
  const shown = catalog.archs.filter((a) => !picked || familyOf(a) === picked);
  const groups = families
    .map((family) => ({
      family,
      archs: shown.filter((a) => familyOf(a) === family),
    }))
    .filter((g) => g.archs.length);

  return (
    <div className={detail.page}>
      <header className={`wrap ${detail.header}`}>
        <h1>Models</h1>
        <p className={detail.summary}>
          Each model runs as an execution graph, which the simulator costs as a tree
          of measured kernels. Pick one and a supported set of parameters to see how
          an iteration&apos;s time is put together.
        </p>
      </header>

      <div className={`wrap ${s.catalog}`}>
        {unknownArch && (
          <p className={s.notice} role="status">
            No model runs as <code>{unknownArch}</code>. These are the ones that do.
          </p>
        )}
        <div className={s.filters} role="group" aria-label="Model family">
          <ToggleTag
            type="family"
            value="Any"
            pressed={!picked}
            count={catalog.archs.length}
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
              count={catalog.archs.filter((a) => familyOf(a) === family).length}
              onClick={() =>
                setQuery({ ...query, family: picked === family ? "" : family })
              }
            />
          ))}
        </div>

        {groups.map(({ family, archs }) => (
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
              {archs.map((arch) => (
                <li key={arch.arch}>
                  <a
                    className={s.archRow}
                    href={modelHref({ arch: arch.arch })}
                    onClick={(event) => openModel(event, { arch: arch.arch })}
                  >
                    <span className={s.archMain}>
                      <span className={s.archName}>{arch.name ?? arch.arch}</span>
                      <code className={s.archTag}>{arch.arch}</code>
                      {arch.summary && (
                        <span className={s.archSummary}>
                          <Prose text={arch.summary} />
                        </span>
                      )}
                    </span>
                    <span className={s.archFacts}>
                      <span className={s.fact}>
                        <span className={s.factName}>Models</span>
                        <span className={s.factValue}>
                          {arch.models.map((stem) => (
                            <Tag key={stem} type="family" value={family}>
                              {models.get(stem)?.name ?? stem}
                            </Tag>
                          ))}
                        </span>
                      </span>
                      <span className={s.fact}>
                        <span className={s.factName}>GPU</span>
                        <span className={s.factValue}>
                          {arch.gpus.map((gpu) => (
                            <Tag key={gpu} type="gpu" value={shortGpu(gpu)} />
                          ))}
                        </span>
                      </span>
                      <span className={s.fact}>
                        <span className={s.factName}>Parameter sets</span>
                        <span className={s.factValue}>
                          <span className={s.figure}>{arch.combinations}</span>
                          {arch.contract !== "iter_wise" && (
                            <span className={s.contract}>
                              {CONTRACTS[arch.contract]}
                            </span>
                          )}
                        </span>
                      </span>
                    </span>
                    <ArrowRight
                      className={s.archArrow}
                      size={20}
                      aria-hidden="true"
                    />
                  </a>
                </li>
              ))}
            </ul>
          </section>
        ))}
      </div>
    </div>
  );
}
