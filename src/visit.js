/* Page views for the site's own log (public/visit.php): one beacon on the
   first load and one per in-site navigation. Only the deployed build sends
   them; the dev server and `vite preview` do not run PHP. The referrer is
   sent once, on the first load, where it names the site the visitor came
   from, and only its origin and path. */

let sentReferrer = false;

function referrer() {
  if (sentReferrer || !document.referrer) return null;
  try {
    const url = new URL(document.referrer);
    if (url.origin === window.location.origin) return null;
    return `${url.origin}${url.pathname}`;
  } catch {
    return null;
  }
}

export function recordVisit() {
  if (!import.meta.env.PROD || !navigator.sendBeacon) return;
  const body = {
    page: window.location.pathname + window.location.search,
    referrer: referrer(),
  };
  sentReferrer = true;
  navigator.sendBeacon(
    `${import.meta.env.BASE_URL}visit.php`,
    new Blob([JSON.stringify(body)], { type: "application/json" }),
  );
}
