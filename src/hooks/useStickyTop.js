import { useCallback, useRef } from "react";

/* Where a sticky panel sticks (`--stick-top`, which its CSS reads): under
   the navigation (`--nav-clearance`) when it fits in the window, otherwise
   by its bottom, so its last line stays reachable. Returns a callback ref. */
const BOTTOM_GAP = 24;

export function useStickyTop() {
  const release = useRef(null);
  // A callback ref: the panel remounts with the tree panel around it.
  return useCallback((node) => {
    release.current?.();
    release.current = null;
    if (!node) return;
    const place = () => {
      const clearance = parseFloat(
        getComputedStyle(node).getPropertyValue("--nav-clearance"),
      );
      const top = Math.min(
        clearance,
        window.innerHeight - node.offsetHeight - BOTTOM_GAP,
      );
      node.style.setProperty("--stick-top", `${top}px`);
    };
    const observer = new ResizeObserver(place);
    observer.observe(node);
    window.addEventListener("resize", place);
    release.current = () => {
      observer.disconnect();
      window.removeEventListener("resize", place);
    };
  }, []);
}
