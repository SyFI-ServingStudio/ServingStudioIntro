import { useState } from "react";
import {
  ArrowRight,
  ArrowUpRight,
  Github,
  FileText,
  BookOpen,
  Twitter,
} from "lucide-react";
import s from "./Hero.module.css";

// Add publication URLs here when they are available; never navigate to placeholders.
const resources = [
  {
    label: "GitHub",
    icon: Github,
    href: "https://github.com/SyFI-ServingStudio/ServingStudio",
  },
  { label: "Paper", icon: FileText },
  { label: "Blog", icon: BookOpen, href: `${import.meta.env.BASE_URL}blog.html` },
  { label: "X / Twitter", icon: Twitter },
];

export function Hero() {
  const [notice, setNotice] = useState(null);
  /* These live in public/, so they are not import-resolved and Vite cannot
     rewrite them: the base prefix has to be applied here. It is "/" in dev and
     "/ServingStudioIntro/" in the published build. */
  const asset = (name) => `${import.meta.env.BASE_URL}images/${name}`;
  const previewImages = {
    original: asset("shoreline-v3.webp"),
    "datacenter-a": asset("hero-datacenter-a.webp"),
    "datacenter-b": asset("hero-datacenter-b.webp"),
    "datacenter-c": asset("hero-datacenter-c.webp"),
  };
  const preview = new URLSearchParams(window.location.search).get("hero");
  const image = previewImages[preview] || asset("hero-datacenter-b.webp");
  return (
    <section
      className={s.hero}
      aria-labelledby="page-title"
      onKeyDown={(event) => {
        if (event.key === "Escape") setNotice(null);
      }}
    >
      <img
        className={s.heroImage}
        src={image}
        srcSet={
          preview === "original"
            ? undefined
            : `${image.replace(".webp", "-960.webp")} 960w, ${image} 1672w`
        }
        sizes="100vw"
        alt=""
        fetchPriority="high"
        width="1672"
        height="941"
      />
      <div className={s.heroShade} />
      <div className={`${s.heroCopy} wrap`}>
        <h1 id="page-title">
          Bridge Simulation and Agents for{" "}
          <span className={s.noBreak}>Speed-of-Light</span> Serving
        </h1>
        <p>
          Compare configurations using performance models grounded in real GPU
          measurements, then let an agent build and validate the strongest
          candidates in real serving frameworks.
        </p>
        <div className={s.heroActions}>
          <a className={`button button-primary`} href="#use-cases">
            See the Agent at work <ArrowRight size={16} />
          </a>
          <a
            className={s.heroSecondary}
            href={`${import.meta.env.BASE_URL}features.html`}
          >
            Explore ServingStudio features <ArrowUpRight size={16} />
          </a>
        </div>
        <ul className={s.resources} aria-label="Project resources">
          {resources.map(({ label, icon: Icon, href }) => {
            const Tag = href ? "a" : "button";
            return (
              <li key={label}>
                <Tag
                  className={s.resource}
                  aria-label={label}
                  {...(href
                    ? {
                        href,
                        ...(href.startsWith("https:")
                          ? { target: "_blank", rel: "noreferrer" }
                          : {}),
                      }
                    : {
                        type: "button",
                        onClick: () => setNotice(label),
                        onMouseLeave: () => setNotice(null),
                        onBlur: () => setNotice(null),
                        "aria-describedby":
                          notice === label ? "resource-tooltip" : undefined,
                      })}
                >
                  <Icon
                    className={s.resourceIcon}
                    size={23}
                    strokeWidth={1.6}
                    aria-hidden="true"
                  />
                  <span className={s.resourceLabel}>
                    {label === "X / Twitter" ? (
                      <>
                        <span className={s.socialFull}>X / Twitter</span>
                        <span className={s.socialShort}>X</span>
                      </>
                    ) : (
                      label
                    )}
                  </span>
                </Tag>
                {notice === label && (
                  <span
                    className={s.resourceTooltip}
                    id="resource-tooltip"
                    role="tooltip"
                  >
                    Coming soon
                  </span>
                )}
              </li>
            );
          })}
        </ul>
      </div>
    </section>
  );
}
