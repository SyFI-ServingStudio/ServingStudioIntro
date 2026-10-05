import { useEffect, useState, useSyncExternalStore } from "react";
import { archNames, loadModels } from "../models/modelData";
import { loadCatalog } from "./kernelData";
import { readQuery, subscribeUrl } from "../../url";
import { KernelCatalog } from "./KernelCatalog";
import { KernelDetail } from "./KernelDetail";

const search = () => window.location.search;

export default function Kernels() {
  const query = useSyncExternalStore(subscribeUrl, search);
  const [catalog, setCatalog] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    // /models names each deployment; without it the pages show arch tags.
    Promise.all([loadCatalog(), loadModels().catch(() => null)]).then(
      ([kernels, models]) =>
        setCatalog({
          ...kernels,
          archNames: archNames(models),
        }),
      setError,
    );
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

export function kernelHref(params) {
  const url = new URLSearchParams(params);
  const text = url.toString();
  return `${import.meta.env.BASE_URL}kernels.html${text ? `?${text}` : ""}`;
}
