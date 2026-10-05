import { ArrowDown } from "lucide-react";
import s from "./PageHero.module.css";
import { SkillInstall } from "./SkillInstall";

export function PageHero({
  eyebrow,
  title,
  description,
  image,
  href,
  linkLabel,
  compact = false,
  skill,
}) {
  return (
    <header className={`${s.hero} ${compact ? s.compact : ""}`}>
      <img
        className={s.image}
        src={`${import.meta.env.BASE_URL}images/${image}`}
        srcSet={`${import.meta.env.BASE_URL}images/${image.replace(".webp", "-960.webp")} 960w, ${import.meta.env.BASE_URL}images/${image} 1672w`}
        sizes="100vw"
        alt=""
        width="1672"
        height="941"
        fetchPriority="high"
      />
      <div className={s.shade} aria-hidden="true" />
      <div className={`wrap ${s.content}`}>
        {eyebrow && <p className={s.eyebrow}>{eyebrow}</p>}
        <h1>{title}</h1>
        <p className={s.description}>{description}</p>
        {skill && <SkillInstall skill={skill} />}
        {href && linkLabel && (
          <a className={s.link} href={href}>
            {linkLabel}
            <ArrowDown size={20} aria-hidden="true" />
          </a>
        )}
      </div>
    </header>
  );
}
