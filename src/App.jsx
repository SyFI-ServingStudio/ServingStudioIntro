import { lazy, Suspense, useLayoutEffect, useRef } from "react";
import { useSiteNavigation } from "./hooks/useSiteNavigation";
import { Navigation } from "./components/Navigation";
import { Hero } from "./sections/Hero";
import { ProductIntroduction } from "./sections/ProductIntroduction";
import { UseCases } from "./sections/UseCases";
import { Workflow } from "./sections/Workflow";
import { Features } from "./pages/Features";
import { Closing } from "./sections/Closing";
import { Architecture } from "./pages/Architecture";
import { RealUseCases } from "./sections/RealUseCases";
import { FeatureOverview } from "./sections/FeatureOverview";

const Blog = lazy(() => import("./pages/Blog"));

export default function App() {
  const mainRef = useRef(null);
  const page = useSiteNavigation(mainRef);
  useLayoutEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    /* Reveal targets are marked with data-reveal by the section that owns them.
       Querying by class name meant reaching into another section's internals,
       which broke silently the moment those classes became CSS Modules and
       started being hashed: the six rows in Supported systems simply stopped
       being found, and nothing rendered differently enough to notice. */
    const elements =
      mainRef.current.parentElement.querySelectorAll("[data-reveal]");
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("in-view");
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.12 },
    );
    elements.forEach((element) => {
      element.classList.add("will-reveal");
      observer.observe(element);
    });
    return () => observer.disconnect();
  }, [page]);
  return (
    <div className="site" id="top">
      <a href="#main" className="skip-link">
        Skip to content
      </a>
      <Navigation page={page.startsWith("blog") ? "blog" : page} />
      <main ref={mainRef} id="main" tabIndex={-1}>
        {page === "overview" ? (
          <>
            <Hero />
            <UseCases />
            <ProductIntroduction />
            <Workflow />
            <RealUseCases />
            <FeatureOverview />
          </>
        ) : page === "features" ? (
          <Features />
        ) : page.startsWith("blog") ? (
          <Suspense
            fallback={
              <p className="wrap blog-loading" role="status">
                Loading blog…
              </p>
            }
          >
            <Blog
              key={page}
              slug={page.startsWith("blog/") ? page.slice(5) : undefined}
            />
          </Suspense>
        ) : (
          <Architecture />
        )}
      </main>
      <Closing compact={page.startsWith("blog")} />
    </div>
  );
}
