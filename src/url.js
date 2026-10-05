/* The query string as page state. Pages that keep their view in it (the
   kernel library, the Models page) read it with useSyncExternalStore over
   subscribeUrl. pushState and replaceState fire no event, so whatever
   changes the URL without a page load says so: setQuery here, and the site
   router (useSiteNavigation) through notifyUrl. Back and forward arrive as
   popstate. */

const listeners = new Set();

export function subscribeUrl(listener) {
  listeners.add(listener);
  window.addEventListener("popstate", listener);
  return () => {
    listeners.delete(listener);
    window.removeEventListener("popstate", listener);
  };
}

export const notifyUrl = () => listeners.forEach((listener) => listener());

export function setQuery(params, { push = false } = {}) {
  const url = new URL(window.location.href);
  url.search = "";
  for (const [key, value] of Object.entries(params)) {
    if (value != null && value !== "") url.searchParams.set(key, value);
  }
  if (url.href === window.location.href) return;
  window.history[push ? "pushState" : "replaceState"](null, "", url);
  notifyUrl();
}

export const readQuery = () =>
  Object.fromEntries(new URLSearchParams(window.location.search));

/* A link that changes only this page's query: a plain click goes there in
   place, from the top; a modified click (new tab, window) is the browser's. */
export function openInPage(event, params) {
  if (event.metaKey || event.ctrlKey || event.shiftKey || event.button !== 0)
    return;
  event.preventDefault();
  setQuery(params, { push: true });
  window.scrollTo({ top: 0, behavior: "instant" });
}
