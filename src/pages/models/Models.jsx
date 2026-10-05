import { useEffect, useState, useSyncExternalStore } from "react";
import { readQuery, subscribeUrl } from "../../url";
import { loadModels } from "./modelData";
import { ModelCatalog } from "./ModelCatalog";
import { ModelDetail } from "./ModelDetail";

const search = () => window.location.search;

/* The Models page: the list of checkpoints, or one public preset
   (`?preset=<checkpoint>/<arch>`) with one member picked by its axis values,
   its cost tree and Live predict. */
export default function Models() {
  const query = useSyncExternalStore(subscribeUrl, search);
  const [catalog, setCatalog] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    loadModels().then(setCatalog, setError);
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
  const { preset: id } = readQuery(query);
  const checkpoint =
    id && catalog.checkpoints.find((c) => c.presets.some((p) => p.id === id));
  return checkpoint ? (
    <ModelDetail
      key={checkpoint.checkpoint}
      catalog={catalog}
      checkpoint={checkpoint}
      preset={checkpoint.presets.find((p) => p.id === id)}
    />
  ) : (
    <ModelCatalog catalog={catalog} unknownPreset={id && !checkpoint ? id : null} />
  );
}
