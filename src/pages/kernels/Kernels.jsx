import { useEffect, useState, useSyncExternalStore } from "react";
import { loadCatalog, readQuery, setQuery, subscribeUrl } from "./kernelData";
import { KernelCatalog } from "./KernelCatalog";
import { KernelDetail } from "./KernelDetail";

const search = () => window.location.search;

export default function Kernels() {
  const query = useSyncExternalStore(subscribeUrl, search);
  const [catalog, setCatalog] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    loadCatalog().then(setCatalog, setError);
  }, []);

  if (error)
    return (
      <p className="wrap blog-loading" role="alert">
        The kernel catalog did not load ({error.message}). The kernel data service
        may be down; reload the page to try again.
      </p>
    );
  if (!catalog)
    return (
      <p className="wrap blog-loading" role="status">
        Loading kernels…
      </p>
    );
  const { kind } = readQuery(query);
  // Only documented kinds have a detail page.
  const entry =
    kind && catalog.kernels.find((k) => k.kind === kind && k.documented);
  return entry ? (
    <KernelDetail key={kind} catalog={catalog} entry={entry} />
  ) : (
    <KernelCatalog catalog={catalog} unknownKind={kind && !entry ? kind : null} />
  );
}

/* A link inside the library changes the query string without a page load. */
export function openKernel(event, params) {
  if (event.metaKey || event.ctrlKey || event.shiftKey || event.button !== 0)
    return;
  event.preventDefault();
  setQuery(params, { push: true });
  window.scrollTo({ top: 0, behavior: "instant" });
}

export function kernelHref(params) {
  const url = new URLSearchParams(params);
  const text = url.toString();
  return `${import.meta.env.BASE_URL}kernels.html${text ? `?${text}` : ""}`;
}
