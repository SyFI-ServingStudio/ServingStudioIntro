import { BookOpen } from "lucide-react";
import { Suspense, lazy, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import ui from "virtual:servingstudio-ui";
import { API_BASE } from "../kernels/kernelData";
import s from "./Models.module.css";

/* "Read more": a live prediction or a simulation's run in ServingStudio UI's
   own result pages, over the whole page. The pages are the UI's
   (scripts/servingstudio-ui.mjs), rendered here in a shadow root, so this
   site's styles stay out and theirs stay in. The data service analyzed the
   prediction (POST /predict with `analyze`) or the run, and forwards the
   Analyzer's routes for one prediction or run under /analyzer/predictions and
   /analyzer/runs; the viewer's reads go there. */

const ResultViewer = lazy(() =>
  import("@servingstudio/ui/embed").then((module) => ({
    default: module.ResultViewer,
  })),
);

const LOADING = { margin: 0, padding: "48px 16px", textAlign: "center" };

/* Whether this build has the viewer, and why not. */
export const READ_MORE = ui;

/* The Analyzer's routes for one prediction or run, as the viewer asks for
   them, and where the data service forwards them. Nothing else is forwarded:
   any other read fails with a message naming it, and is logged, so a gap
   shows. */
const ANALYZER = /^\/api\/analyzer\/v1\/(predictions|runs)\/(.+)$/;

export async function forwardedFetch(url, { signal } = {}) {
  const address = new URL(url, window.location.origin);
  const route = address.pathname.match(ANALYZER);
  if (!route) {
    const detail = `The public site does not serve ${address.pathname}${address.search}: only the Analyzer's routes for one prediction or run are forwarded.`;
    console.error(`[read more] ${detail}`);
    return new Response(JSON.stringify({ detail }), {
      status: 404,
      headers: { "Content-Type": "application/json" },
    });
  }
  const path = `${API_BASE}/analyzer/${route[1]}/${route[2]}`;
  return fetch(path + address.search, {
    signal,
    headers: { accept: "application/json" },
  });
}

export function ReadMoreButton({ onClick, disabled, busy }) {
  if (!READ_MORE.available) return null;
  return (
    <button
      type="button"
      className={s.addButton}
      onClick={onClick}
      disabled={disabled || busy}
      aria-busy={busy || undefined}
    >
      <BookOpen size={16} aria-hidden="true" />
      {busy ? "Opening the full analysis…" : "Read more"}
    </button>
  );
}

/* The overlay for one analyzed prediction or run: `id` is its
   `prediction_id` or `run_id`; `name` heads it, so the viewer reads no
   catalog. */
export function ResultOverlay({ kind = "prediction", id, name, onClose }) {
  const overlay = useRef(null);
  const [container, setContainer] = useState(null);

  // The page behind is inert and still while the overlay is open; focus
  // returns to what opened it.
  useEffect(() => {
    const root = document.getElementById("root");
    const opener = document.activeElement;
    const overflow = document.body.style.overflow;
    root.inert = true;
    document.body.style.overflow = "hidden";
    overlay.current.focus();
    return () => {
      root.inert = false;
      document.body.style.overflow = overflow;
      opener?.focus?.();
    };
  }, []);

  const attach = (host) => {
    if (!host || host.shadowRoot) return;
    const mount = document.createElement("div");
    mount.style.height = "100%";
    host.attachShadow({ mode: "open" }).append(mount);
    setContainer(mount);
  };

  return createPortal(
    <div
      ref={overlay}
      className={s.readMore}
      role="dialog"
      aria-modal="true"
      aria-label={kind === "run" ? "Simulation" : "Timing prediction"}
      tabIndex={-1}
      // The viewer's own dialogs take Escape first (MUI stops it there).
      onKeyDown={(event) => event.key === "Escape" && onClose()}
    >
      <div ref={attach} className={s.readMoreHost} />
      {container &&
        createPortal(
          // Inside the shadow root only inline styles apply.
          <Suspense fallback={<p style={LOADING}>Loading the analysis…</p>}>
            <ResultViewer
              kind={kind}
              id={id}
              displayName={name}
              transport={forwardedFetch}
              onClose={onClose}
            />
          </Suspense>,
          container,
        )}
    </div>,
    document.body,
  );
}
