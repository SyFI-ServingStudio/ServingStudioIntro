import { prepareHero } from "../heroImages";
import { useEffect, useRef, useState } from "react";
import { flushSync } from "react-dom";
import { notifyUrl } from "../url";

const base = import.meta.env.BASE_URL;
const pages = new Map([
  [base, "overview"],
  [`${base}index.html`, "overview"],
  [`${base}features.html`, "features"],
  [`${base}architecture.html`, "architecture"],
  [`${base}models.html`, "models"],
  [`${base}simulate.html`, "simulate"],
  [`${base}kernels.html`, "kernels"],
  [`${base}blog.html`, "blog"],
]);
export function pageFromPath(pathname) {
  if (pages.has(pathname)) return pages.get(pathname);
  if (pathname.startsWith(`${base}blog/`)) {
    const slug = pathname
      .slice(`${base}blog/`.length)
      .replace(/\/(?:index\.html)?$/, "");
    if (/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(slug)) return `blog/${slug}`;
  }
  return null;
}
const currentPage = () => pageFromPath(window.location.pathname) || "overview";

// Start route-specific work before replacing the current page. The browser can
// keep showing the current page while a lazily loaded page becomes ready.

export function preparePage(page, search = "") {
  const hero = prepareHero(page, search);
  if (page === "models")
    return Promise.all([
      hero,
      import("../pages/models/Models"),
      import("../pages/models/modelData").then(({ loadModels }) =>
        loadModels().catch(() => {}),
      ),
    ]);
  if (page === "simulate")
    return Promise.all([
      import("../pages/simulate/Simulate"),
      import("../pages/simulate/simulateData").then(({ loadSimPresets }) =>
        loadSimPresets().catch(() => {}),
      ),
    ]);
  if (page === "kernels")
    return Promise.all([
      hero,
      import("../pages/kernels/Kernels"),
      import("../pages/kernels/kernelData").then(({ loadCatalog }) =>
        loadCatalog().catch(() => {}),
      ),
    ]);
  if (page?.startsWith("blog")) return Promise.all([hero, import("../pages/Blog")]);
  return hero;
}

export function useSiteNavigation(mainRef) {
  const [page, setPage] = useState(currentPage);
  const transitionRef = useRef(null);

  useEffect(() => {
    let navigationId = 0;
    async function navigate(url, push) {
      const id = ++navigationId;
      await preparePage(pageFromPath(url.pathname), url.search);
      if (id !== navigationId) return;
      transitionRef.current?.skipTransition();
      const update = () => {
        if (push) window.history.pushState(null, "", url);
        // A link to the page already shown changes only its query: the
        // page's URL subscribers re-read it in the same commit.
        flushSync(() => {
          setPage(currentPage());
          if (push) notifyUrl();
        });
        const hash = url.hash.slice(1);
        const target = hash ? document.getElementById(hash) : null;
        if (target) target.scrollIntoView({ behavior: "instant" });
        else window.scrollTo({ top: 0, behavior: "instant" });
        mainRef.current?.focus({ preventScroll: true });
      };
      const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      if (document.startViewTransition && !reduced) {
        transitionRef.current = document.startViewTransition(update);
        // A navigation that starts before this one's animation ends skips it,
        // and skipTransition() rejects `ready` with an AbortError. Any other
        // rejection is a real failure and stays uncaught.
        transitionRef.current.ready.catch((error) => {
          if (error?.name !== "AbortError") throw error;
        });
      } else {
        update();
        if (!reduced)
          mainRef.current?.animate([{ opacity: 0 }, { opacity: 1 }], {
            duration: 160,
            easing: "ease-out",
          });
      }
    }
    function onClick(event) {
      if (
        event.defaultPrevented ||
        event.button !== 0 ||
        event.metaKey ||
        event.ctrlKey ||
        event.shiftKey ||
        event.altKey
      )
        return;
      const link = event.target.closest?.("a[href]");
      if (
        !link ||
        link.hasAttribute("download") ||
        (link.target && link.target !== "_self")
      )
        return;
      const url = new URL(link.href, window.location.href);
      if (url.origin !== window.location.origin || !pageFromPath(url.pathname))
        return;
      if (
        url.pathname === window.location.pathname &&
        url.search === window.location.search &&
        url.hash
      )
        return;
      event.preventDefault();
      if (url.href === window.location.href) {
        window.scrollTo({ top: 0, behavior: "smooth" });
        return;
      }
      navigate(url, true);
    }
    const onPopState = () => navigate(new URL(window.location.href), false);
    document.addEventListener("click", onClick);
    window.addEventListener("popstate", onPopState);
    return () => {
      navigationId++;
      document.removeEventListener("click", onClick);
      window.removeEventListener("popstate", onPopState);
      transitionRef.current?.skipTransition();
    };
  }, [mainRef]);

  useEffect(() => {
    if (page.startsWith("blog")) return;
    document.title =
      page === "overview"
        ? "ServingStudio | Simulate and improve LLM serving"
        : `${{ features: "Features", models: "Models", simulate: "Simulate", kernels: "Kernels" }[page] || "Architecture"} | ServingStudio`;
  }, [page]);
  return page;
}
