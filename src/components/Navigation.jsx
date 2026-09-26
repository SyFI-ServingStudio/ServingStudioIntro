import { Brand } from "./Brand";
import { preparePage } from "../hooks/useSiteNavigation";
import s from "./Navigation.module.css";

export function Navigation({ page = "overview" }) {
  const base = import.meta.env.BASE_URL;
  return (
    <header className={s.navigation}>
      <div className={s.navigationInner}>
        <Brand />
        <nav aria-label="Main navigation">
          {[
            ["overview", "Overview", base],
            ["features", "Features", `${base}features.html`],
            ["architecture", "Architecture", `${base}architecture.html`],
            ["kernels", "Kernels", `${base}kernels.html`],
            ["blog", "Blog", `${base}blog.html`],
          ].map(([id, name, href]) => (
            <a
              key={id}
              href={href}
              aria-current={page === id ? "page" : undefined}
              onPointerEnter={() => {
                if (id === "blog" || id === "kernels")
                  preparePage(id).catch(() => {});
              }}
              onFocus={() => {
                if (id === "blog" || id === "kernels")
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
