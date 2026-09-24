const base = `${import.meta.env.BASE_URL}images/`;
const heroes = { overview: "b", features: "a", architecture: "c", blog: "blog" };
const pending = new Map();

// Share the same responsive sources as the rendered hero so prefetches are reused.
export function prepareHero(page, search = "", priority = "high") {
  if (page?.startsWith("blog/")) return Promise.resolve();
  const preview = new URLSearchParams(search).get("hero");
  const variant =
    page === "overview" && /^datacenter-[abc]$/.test(preview)
      ? preview.slice(-1)
      : heroes[page] || "b";
  const original = page === "overview" && preview === "original";
  const name = original
    ? "shoreline-v3"
    : page === "blog"
      ? "hero-blog"
      : `hero-datacenter-${variant}`;
  const key = `${name}:${window.innerWidth}:${window.devicePixelRatio}`;
  if (!pending.has(key)) {
    const image = new Image();
    image.fetchPriority = priority;
    if (!original) {
      image.sizes = "100vw";
      image.srcset = `${base}${name}-960.webp 960w, ${base}${name}.webp 1672w`;
    }
    image.src = `${base}${name}.webp`;
    // A failed image must not prevent access to the page.
    pending.set(
      key,
      image.decode().catch(() => {
        pending.delete(key);
      }),
    );
  }
  return pending.get(key);
}

export function warmHeroImages() {
  for (const page of Object.keys(heroes)) prepareHero(page, "", "low");
}
