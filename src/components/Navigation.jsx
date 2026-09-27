import { useEffect, useRef } from "react";
import { Brand } from "./Brand";
import { preparePage } from "../hooks/useSiteNavigation";
import s from "./Navigation.module.css";

export function Navigation({ page = "overview" }) {
  const base = import.meta.env.BASE_URL;
  const nav = useRef(null);
  // On a phone the row scrolls sideways: keep the current page in view, and
  // mark which edges hide more links so they fade there.
  useEffect(() => {
    const row = nav.current;
    if (!row) return undefined;
    const settle = () => {
      const current = row.querySelector("[aria-current]");
      if (current) {
        const hidden =
          current.getBoundingClientRect().right - row.getBoundingClientRect().right;
        // Clear of the 32px fade too; the browser clamps at the row's end.
        if (hidden > -32) row.scrollLeft += hidden + 32;
      }
      markEdges(row);
    };
    const mark = () => markEdges(row);
    settle();
    // The web font widens the links once it arrives.
    document.fonts?.ready.then(settle);
    window.addEventListener("resize", mark);
    return () => window.removeEventListener("resize", mark);
  }, [page]);
  return (
    <header className={s.navigation}>
      <div className={s.navigationInner}>
        <Brand />
        <nav
          aria-label="Main navigation"
          ref={nav}
          onScroll={(event) => markEdges(event.currentTarget)}
        >
          {[
            ["overview", "Overview", base],
            ["features", "Features", `${base}features.html`],
            ["architecture", "Architecture", `${base}architecture.html`],
            ["models", "Models", `${base}models.html`],
            ["kernels", "Kernels", `${base}kernels.html`],
            ["blog", "Blog", `${base}blog.html`],
          ].map(([id, name, href]) => (
            <a
              key={id}
              href={href}
              aria-current={page === id ? "page" : undefined}
              onPointerEnter={() => {
                if (["blog", "models", "kernels"].includes(id))
                  preparePage(id).catch(() => {});
              }}
              onFocus={() => {
                if (["blog", "models", "kernels"].includes(id))
                  preparePage(id).catch(() => {});
              }}
            >
              {name}
            </a>
          ))}
        </nav>
      </div>
    </header>
  );
}

function markEdges(row) {
  if (!row) return;
  const end = row.scrollWidth - row.clientWidth;
  row.dataset.more =
    end <= 1
      ? ""
      : row.scrollLeft <= 1
        ? "end"
        : row.scrollLeft >= end - 1
          ? "start"
          : "both";
}
