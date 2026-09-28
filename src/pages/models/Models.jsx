import { useEffect, useState, useSyncExternalStore } from "react";
import { readQuery, setQuery, subscribeUrl } from "../kernels/kernelData";
import { loadArchs } from "./modelData";
import { ModelCatalog } from "./ModelCatalog";
import { ModelDetail } from "./ModelDetail";

const search = () => window.location.search;

/* The Models page: the list of archs, or one arch (`?arch=`) with a supported
   parameter set picked and its cost tree. */
export default function Models() {
  const query = useSyncExternalStore(subscribeUrl, search);
  const [catalog, setCatalog] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    loadArchs().then(setCatalog, setError);
  }, []);

  if (error)
    return (
      <p className="wrap blog-loading" role="alert">
        The model list did not load ({error.message}). The model data service may be
        down; reload the page to try again.
      </p>
    );
  if (!catalog)
    return (
      <p className="wrap blog-loading" role="status">
        Loading models…
      </p>
    );
  const { arch } = readQuery(query);
  const entry = arch && catalog.archs.find((a) => a.arch === arch);
  return entry ? (
    <ModelDetail key={arch} catalog={catalog} entry={entry} />
  ) : (
    <ModelCatalog catalog={catalog} unknownArch={arch && !entry ? arch : null} />
  );
}

/* A link inside the page changes the query string without a page load. */
export function openModel(event, params) {
  if (event.metaKey || event.ctrlKey || event.shiftKey || event.button !== 0)
    return;
  event.preventDefault();
  setQuery(params, { push: true });
  window.scrollTo({ top: 0, behavior: "instant" });
}

export function modelHref(params) {
  const text = new URLSearchParams(params).toString();
  return `${import.meta.env.BASE_URL}models.html${text ? `?${text}` : ""}`;
}
