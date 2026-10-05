import { Check, Copy, ExternalLink } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import s from "./SkillInstall.module.css";

/* The agent skills that read the public API. The `skills` CLI installs all
   of them from GitHub, finding them under public_api/; each page links the
   source of the one that does what the page does. */
const SKILL_REPO =
  "https://github.com/SyFI-ServingStudio/ServingStudioSim/tree/main";
const SKILL_INSTALL = `npx skills add ${SKILL_REPO}/public_api`;

const LABEL = { idle: "Copy", copied: "Copied", selected: "Selected" };
const STATUS = {
  idle: "",
  copied: "Install command copied.",
  selected:
    "Clipboard unavailable. The command is selected; copy it from the keyboard.",
};

/* One line under a hero's description. Without a clipboard (an insecure
   origin, a denied permission) the button selects the command instead, ready
   for a manual copy. */
export function SkillInstall({ skill }) {
  const [state, setState] = useState("idle");
  const command = useRef(null);
  useEffect(() => {
    if (state === "idle") return undefined;
    const timer = setTimeout(() => setState("idle"), 1600);
    return () => clearTimeout(timer);
  }, [state]);

  const selectCommand = () => {
    window.getSelection()?.selectAllChildren(command.current);
    setState("selected");
  };
  const copy = () => {
    if (!navigator.clipboard?.writeText) return selectCommand();
    return navigator.clipboard
      .writeText(SKILL_INSTALL)
      .then(() => setState("copied"), selectCommand);
  };

  return (
    <div className={s.install}>
      <span className={s.label}>Use it from your agent:</span>
      <span className={s.chip}>
        <code ref={command} className={s.command} tabIndex={0}>
          {SKILL_INSTALL}
        </code>
        <button type="button" className={s.copy} onClick={copy}>
          {state === "copied" ? (
            <Check size={14} aria-hidden="true" />
          ) : (
            <Copy size={14} aria-hidden="true" />
          )}
          {LABEL[state]}
        </button>
      </span>
      <a
        className={s.source}
        href={`${SKILL_REPO}/public_api/skills/${skill}`}
        target="_blank"
        rel="noreferrer"
      >
        Skill source
        <ExternalLink size={14} aria-label="opens in a new tab" />
      </a>
      <span className="visually-hidden" role="status">
        {STATUS[state]}
      </span>
    </div>
  );
}
