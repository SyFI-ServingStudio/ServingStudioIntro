import { useEffect, useState } from "react";
import { archNames, loadModels } from "../models/modelData";
import { loadCatalog } from "./kernelData";
import { useQuery } from "../../url";
import { KernelCatalog } from "./KernelCatalog";
import { KernelDetail } from "./KernelDetail";

export default function Kernels() {
  const query = useQuery();
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
  const { kind } = query;
  // Only documented kinds have a detail page.
  const entry =
    kind && catalog.kernels.find((k) => k.kind === kind && k.documented);
  return entry ? (
    <KernelDetail key={kind} catalog={catalog} entry={entry} />
  ) : (
    <KernelCatalog catalog={catalog} unknownKind={kind && !entry ? kind : null} />
  );
}
