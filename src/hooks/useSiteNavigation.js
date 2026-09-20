import { prepareHero } from "../heroImages";
import { useEffect, useRef, useState } from "react";
import { flushSync } from "react-dom";

const base = import.meta.env.BASE_URL;
const pages = new Map([
  [base, "overview"],
  [`${base}index.html`, "overview"],
  [`${base}features.html`, "features"],
  [`${base}architecture.html`, "architecture"],
]);
const currentPage = () => pages.get(window.location.pathname) || "overview";

export function useSiteNavigation(mainRef) {
  const [page, setPage] = useState(currentPage);
  const transitionRef = useRef(null);

  useEffect(() => {
    let navigationId = 0;
    async function navigate(url, push) {
      const id = ++navigationId;
      await prepareHero(pages.get(url.pathname), url.search);
      if (id !== navigationId) return;
      transitionRef.current?.skipTransition();
      const update = () => {
        if (push) window.history.pushState(null, "", url);
        flushSync(() => setPage(currentPage()));
        const hash = url.hash.slice(1);
        const target = hash ? document.getElementById(hash) : null;
        if (target) target.scrollIntoView({ behavior: "instant" });
        else window.scrollTo({ top: 0, behavior: "instant" });
        mainRef.current?.focus({ preventScroll: true });
      };
      const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      if (document.startViewTransition && !reduced) {
        transitionRef.current = document.startViewTransition(update);
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
      if (url.origin !== window.location.origin || !pages.has(url.pathname)) return;
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
    document.title =
      page === "overview"
        ? "ServingStudio | Simulate and improve LLM serving"
        : `${page === "features" ? "Features" : "Architecture"} | ServingStudio`;
  }, [page]);
  return page;
}
