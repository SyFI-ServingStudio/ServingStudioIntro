import { Cpu } from "lucide-react";
import s from "./Tag.module.css";

/* One colour per entity, the same everywhere on the page.

   Families are hues of their own. Precision runs cool to hot with fewer bits,
   so how quantized a kernel is reads before the label does. GPUs take the
   hues neither of those use: green for Blackwell, orchid for Hopper. A value
   missing here falls back to a neutral tag rather than a generated hue. */
const COLORS = {
  // Categories colour the group bands in the list and the category tag.
  category: {
    GEMM: "#5b9cf5",
    Attention: "#a08cf0",
    MoE: "#3cc49a",
    Communication: "#e0a33a",
    Normalization: "#e0709a",
    Quantization: "#ef8354",
  },
  family: {
    Llama: "#7aa7ff",
    Qwen: "#b89cff",
    GLM: "#4fd1b0",
    DeepSeek: "#ff9a76",
  },
  precision: {
    BF16: "#8fb8e8",
    FP16: "#7fd0e6",
    FP8: "#ffc454",
    MXFP4: "#ff8a8a",
    NVFP4: "#ff6fb8",
  },
  gpu: {
    B200: "#8fd14f",
    H200: "#e58cf0",
  },
};

export const tagColor = (type, value) => COLORS[type]?.[value];

export function Tag({ type, value, title, children }) {
  const color = COLORS[type]?.[value];
  const neutral = !color || value === "Any";
  return (
    <span
      className={`${s.tag} ${s[type]} ${neutral ? s.neutral : ""}`}
      style={color ? { "--tag": color } : undefined}
      title={title}
    >
      {type === "gpu" && <Cpu size={14} aria-hidden="true" />}
      {children ?? value}
    </span>
  );
}

export function TagList({ children, label }) {
  return (
    <span className={s.list} aria-label={label}>
      {children}
    </span>
  );
}

/* A tag that filters. Off, it keeps its identity (the dot, the icon) but not
   its colour; on, it is the same tag the table rows wear, so a choice and the
   rows it keeps look alike. "mixed" is a family with some of its models. */
// A choice with no colour of its own (a backend, a batch size) is marked in
// the site accent once picked, so the current pick always stands out.
const CHOICE = "#99baff";

export function ToggleTag({
  type,
  value,
  pressed,
  count,
  small,
  title,
  disabled,
  faint,
  onClick,
  children,
}) {
  const color = COLORS[type]?.[value] ?? (type === "choice" ? CHOICE : null);
  const empty = disabled ?? (count === 0 && !pressed);
  return (
    <button
      type="button"
      className={[
        s.tag,
        s[type],
        s.toggle,
        small ? s.small : "",
        pressed ? "" : s.off,
        COLORS[type]?.[value] ? s.entity : "",
        value === "Any" ? s.neutral : "",
        faint ? s.faint : "",
      ].join(" ")}
      style={color ? { "--tag": color } : undefined}
      aria-pressed={pressed}
      title={title}
      disabled={empty}
      onClick={onClick}
    >
      {type === "gpu" && <Cpu size={14} aria-hidden="true" />}
      {children ?? value}
      {count != null && <span className={s.count}>{count}</span>}
    </button>
  );
}
