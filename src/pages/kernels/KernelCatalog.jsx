import { Search, X } from "lucide-react";
import { useEffect, useSyncExternalStore } from "react";
import { PageHero } from "../../components/PageHero";
import {
  groupModels,
  presetArch,
  presetCheckpoint,
  usedModels,
  readQuery,
  setQuery,
  shortGpu,
  subscribeUrl,
} from "./kernelData";
import { kernelHref, openKernel } from "./Kernels";
import { SkillInstall } from "./SkillInstall";
import { Tag, TagList, ToggleTag, tagColor } from "./Tag";
import s from "./KernelCatalog.module.css";

const search = () => window.location.search;
// Kinds without a DOC have no category yet; they close the list.
const UNDOCUMENTED = "Not yet documented";
const categoryOf = (kernel) => kernel.category ?? UNDOCUMENTED;
const titleOf = (kernel) => kernel.title ?? kernel.kind;
const list = (value) => (value ? value.split(",").filter(Boolean) : []);

/* Filters live in the column they filter. Within a column choices are OR'd,
   across columns AND'd. GPU and precision are judged together: B200 with FP8
   keeps a kernel only if it was measured at FP8 on a B200. */
function matcher(query, catalog) {
  const models = new Set(list(query.model));
  const gpus = new Set(list(query.gpu));
  const precisions = new Set(list(query.precision));
  const needle = (query.q || "").trim().toLowerCase();
  const covered = (k, over = {}) => {
    const g = over.gpus ?? gpus;
    const p = over.precisions ?? precisions;
    return k.coverage.filter(
      (c) => (!g.size || g.has(shortGpu(c.gpu))) && (!p.size || p.has(c.precision)),
    );
  };
  const matches = (k, over = {}) => {
    const m = over.models ?? models;
    if (m.size && !usedModels(k).some((key) => m.has(key))) return false;
    if (!covered(k, over).length) return false;
    if (!needle) return true;
    return [
      k.kind,
      titleOf(k),
      categoryOf(k),
      k.subcategory ?? "",
      ...k.used_by,
      ...groupModels(usedModels(k), catalog.models).flatMap((g) => [
        g.family,
        ...g.names,
      ]),
      ...k.coverage.map((c) => c.backend),
    ]
      .join(" ")
      .toLowerCase()
      .includes(needle);
  };
  return { models, gpus, precisions, covered, matches };
}

const toggle = (set, values, on) => {
  const next = new Set(set);
  values.forEach((v) => (on ? next.add(v) : next.delete(v)));
  return [...next].join(",");
};

export function KernelCatalog({ catalog, unknownKind }) {
  useSyncExternalStore(subscribeUrl, search);
  // A detail page renames the tab; coming back has to name it again.
  useEffect(() => {
    document.title = "Kernels | ServingStudio";
  }, []);
  const query = readQuery();
  const { q = "", cat = "", sub = "" } = query;
  const update = (patch) => setQuery({ ...query, ...patch });
  const m = matcher(query, catalog);

  // The models some kernel is used by: every other one would match nothing.
  const usedBy = [...new Set(catalog.kernels.flatMap(usedModels))];
  const families = groupModels(usedBy, catalog.models);
  const gpuNames = catalog.gpus.map((g) => shortGpu(g.name));
  // The count on a choice is how many kernels have it, given the other
  // columns' filters. Choices in its own column don't move it, so picking FP8
  // leaves the number on BF16 alone.
  const countWith = (over) =>
    catalog.kernels.filter((k) => m.matches(k, over)).length;

  const matched = catalog.kernels.filter((k) => m.matches(k));
  const categories = [...catalog.categories, UNDOCUMENTED];
  const counts = Object.fromEntries(categories.map((c) => [c, 0]));
  matched.forEach((k) => (counts[categoryOf(k)] += 1));
  // Attention is split by the attention a layer runs (MHA / GQA, MLA, DSA,
  // Gated DeltaNet); other categories are one section.
  const sections = (category) =>
    catalog.subcategories[category] ?? [{ name: null, summary: null }];
  const subs = cat ? (catalog.subcategories[cat] ?? []) : [];
  const subCounts = {};
  matched.forEach((k) => {
    if (k.category === cat)
      subCounts[k.subcategory] = (subCounts[k.subcategory] ?? 0) + 1;
  });
  const shown = matched.filter(
    (k) => (!cat || categoryOf(k) === cat) && (!sub || k.subcategory === sub),
  );
  const order = (a, b) =>
    usedModels(b).length - usedModels(a).length ||
    titleOf(a).localeCompare(titleOf(b));
  const groups = categories
    .map((category) => [
      category,
      sections(category)
        .map((section) => ({
          ...section,
          kernels: shown
            .filter(
              (k) => categoryOf(k) === category && k.subcategory === section.name,
            )
            .sort(order),
        }))
        .filter((section) => section.kernels.length),
    ])
    .filter(([, parts]) => parts.length);
  const filtered = q || m.models.size || m.gpus.size || m.precisions.size;
  return (
    <div className={s.page}>
      <PageHero
        compact
        title="Kernel Library"
        description={
          <>
            <span className={s.lede}>
              Explore measured GPU performance across the kernels, shapes, and
              backends that power LLM serving.
            </span>
            <SkillInstall />
          </>
        }
        image="hero-kernels.webp"
      />
      <div className={`wrap ${s.catalog}`}>
        {unknownKind && (
          <p className={s.notice} role="status">
            There is no documented kernel named <code>{unknownKind}</code>. Search
            the list below instead.
          </p>
        )}

        <div className={s.categories} role="group" aria-label="Category">
          <button
            type="button"
            aria-pressed={!cat}
            onClick={() => update({ cat: "", sub: "" })}
          >
            All <span>{matched.length}</span>
          </button>
          {categories.map((category) => (
            <button
              key={category}
              type="button"
              aria-pressed={cat === category}
              disabled={!counts[category]}
              style={{ "--band": tagColor("category", category) }}
              onClick={() =>
                update({ cat: cat === category ? "" : category, sub: "" })
              }
            >
              <i className={s.dot} aria-hidden="true" />
              {category} <span>{counts[category]}</span>
            </button>
          ))}
        </div>
        {subs.length > 0 && (
          <div className={s.subcategories} role="group" aria-label={`${cat} kind`}>
            {subs.map(({ name }) => (
              <button
                key={name}
                type="button"
                aria-pressed={sub === name}
                disabled={!subCounts[name]}
                onClick={() => update({ sub: sub === name ? "" : name })}
              >
                {name} <span>{subCounts[name] ?? 0}</span>
              </button>
            ))}
          </div>
        )}

        <div className={s.board}>
          <div className={s.filterBar} role="group" aria-label="Filter kernels">
            <div className={s.filterCell}>
              <span className={s.columnName}>Kernel</span>
              <label className={s.search}>
                <Search size={18} aria-hidden="true" />
                <input
                  type="search"
                  aria-label="Search kernels"
                  value={q}
                  placeholder="Search kernels or backends"
                  onChange={(event) => update({ q: event.target.value })}
                />
              </label>
              {filtered ? (
                <button
                  type="button"
                  className={s.clear}
                  onClick={() => setQuery({ cat, sub })}
                >
                  <X size={16} aria-hidden="true" />
                  Clear filters
                </button>
              ) : null}
            </div>

            <div className={s.filterCell}>
              <span className={s.columnName}>Used by</span>
              <div className={s.choices}>
                {families.map(({ family, keys: stems, names }) => {
                  const picked = stems.filter((n) => m.models.has(n));
                  const state =
                    picked.length === 0
                      ? false
                      : picked.length === stems.length
                        ? true
                        : "mixed";
                  return (
                    <ToggleTag
                      key={family}
                      type="family"
                      value={family}
                      pressed={state}
                      count={countWith({ models: new Set(stems) })}
                      title={names.join(", ")}
                      onClick={() =>
                        update({ model: toggle(m.models, stems, state !== true) })
                      }
                    />
                  );
                })}
              </div>
              {families
                .filter(
                  ({ keys }) =>
                    keys.length > 1 && keys.some((n) => m.models.has(n)),
                )
                .map(({ family, keys: stems, names }) => (
                  <div
                    key={family}
                    className={s.variants}
                    aria-label={`${family} models`}
                  >
                    {stems.map((stem, i) => (
                      <ToggleTag
                        key={stem}
                        type="family"
                        value={family}
                        small
                        pressed={m.models.has(stem)}
                        onClick={() =>
                          update({
                            model: toggle(m.models, [stem], !m.models.has(stem)),
                          })
                        }
                      >
                        {names[i]}
                      </ToggleTag>
                    ))}
                  </div>
                ))}
            </div>

            <div className={s.filterCell}>
              <span className={s.columnName}>Precision</span>
              <div className={s.choices}>
                {catalog.precisions.map((p) => (
                  <ToggleTag
                    key={p}
                    type="precision"
                    value={p}
                    pressed={m.precisions.has(p)}
                    count={countWith({ precisions: new Set([p]) })}
                    onClick={() =>
                      update({
                        precision: toggle(m.precisions, [p], !m.precisions.has(p)),
                      })
                    }
                  />
                ))}
              </div>
            </div>

            <div className={s.filterCell}>
              <span className={s.columnName}>GPUs</span>
              <div className={s.choices}>
                {gpuNames.map((g) => (
                  <ToggleTag
                    key={g}
                    type="gpu"
                    value={g}
                    pressed={m.gpus.has(g)}
                    count={countWith({ gpus: new Set([g]) })}
                    onClick={() =>
                      update({ gpu: toggle(m.gpus, [g], !m.gpus.has(g)) })
                    }
                  />
                ))}
              </div>
            </div>
          </div>

          {groups.length === 0 ? (
            <div className={s.empty}>
              <p>No kernel matches these filters.</p>
              <button type="button" onClick={() => setQuery({})}>
                Clear filters
              </button>
            </div>
          ) : (
            <table className={s.table}>
              <colgroup>
                <col className={s.colKernel} />
                <col className={s.colModels} />
                <col className={s.colPrecision} />
                <col />
              </colgroup>
              <caption className="visually-hidden">
                Kernels grouped by category, with the models that use them and the
                precisions and GPUs they were measured at
              </caption>
              <thead className="visually-hidden">
                <tr>
                  <th scope="col">Kernel</th>
                  <th scope="col">Used by</th>
                  <th scope="col">Precision</th>
                  <th scope="col">GPUs</th>
                </tr>
              </thead>
              {groups.map(([category, parts]) => (
                <tbody
                  key={category}
                  style={{ "--band": tagColor("category", category) }}
                >
                  <tr className={s.groupRow}>
                    <th scope="rowgroup" colSpan={4}>
                      <h2 className={s.band}>{category}</h2>
                    </th>
                  </tr>
                  {parts.map(({ name, summary, kernels }) => [
                    name && (
                      <tr key={name} className={s.subRow}>
                        <th scope="rowgroup" colSpan={4}>
                          <div className={s.subInner}>
                            <h3>{name}</h3>
                            <p>{summary}</p>
                          </div>
                        </th>
                      </tr>
                    ),
                    ...kernels.map((kernel) => (
                      <KernelRow
                        key={kernel.kind}
                        kernel={kernel}
                        catalog={catalog}
                        coverage={m.covered(kernel)}
                        usedBy={usedBy.length}
                      />
                    )),
                  ])}
                </tbody>
              ))}
            </table>
          )}
        </div>
      </div>
    </div>
  );
}

/* The deployments (archs) of one model that build a config of a kernel. */
const deployments = (kernel, key) =>
  kernel.used_by
    .filter((preset) => presetCheckpoint(preset) === key)
    .map(presetArch)
    .join(", ");

/* The precision and GPU cells follow the filters: with B200 picked, the
   precision cell lists what was measured on B200, and the other way round. */
function KernelRow({ kernel, catalog, coverage, usedBy }) {
  const models = catalog.models;
  const used = usedModels(kernel);
  const precisions = catalog.precisions.filter((p) =>
    coverage.some((c) => c.precision === p),
  );
  const gpus = catalog.gpus
    .map((g) => g.name)
    .filter((name) => coverage.some((c) => c.gpu === name));
  const href = kernel.documented ? kernelHref({ kind: kernel.kind }) : null;
  // The name is the link for keyboards and "open in new tab"; a click anywhere
  // else on the row follows it too, unless it ends a text selection.
  const openRow = (event) => {
    if (!href || event.target.closest("a")) return;
    if (window.getSelection()?.toString()) return;
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.button === 1) {
      window.open(href, "_blank", "noopener");
      return;
    }
    if (event.button === 0) openKernel(event, { kind: kernel.kind });
  };
  return (
    <tr
      className={href ? s.linkRow : undefined}
      onClick={openRow}
      onAuxClick={openRow}
    >
      <th scope="row" className={s.kernelCell}>
        {href ? (
          <a
            href={href}
            onClick={(event) => openKernel(event, { kind: kernel.kind })}
          >
            {kernel.title}
          </a>
        ) : (
          <span>{titleOf(kernel)}</span>
        )}
        {kernel.title && <code>{kernel.kind}</code>}
      </th>
      <td data-label="Used by">
        {usedBy > 1 && used.length === usedBy ? (
          <Tag
            type="family"
            value="all"
            title="Every model with a public deployment uses this kernel"
          >
            All models
          </Tag>
        ) : used.length ? (
          <TagList label="Used by">
            {groupModels(used, models).map(({ family, keys, names }) => (
              <Tag
                key={family}
                type="family"
                value={family}
                title={keys
                  .map((key, i) => `${names[i]}: ${deployments(kernel, key)}`)
                  .join("\n")}
              />
            ))}
          </TagList>
        ) : null}
      </td>
      <td data-label="Precision">
        <TagList label="Precision">
          {precisions.map((p) => (
            <Tag key={p} type="precision" value={p} />
          ))}
        </TagList>
      </td>
      <td data-label="GPUs">
        <TagList label="GPUs">
          {gpus.map((name) => (
            <Tag key={name} type="gpu" value={shortGpu(name)} />
          ))}
        </TagList>
      </td>
    </tr>
  );
}
