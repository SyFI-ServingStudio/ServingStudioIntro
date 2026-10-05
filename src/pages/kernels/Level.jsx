import picker from "./ConfigPicker.module.css";

/* One row of a picker: its name, then its choices. `label` names the group
   for assistive technology and, unless `name` gives another, on the row. */
export function Level({ label, name = label, className, children }) {
  return (
    <div
      className={className ? `${picker.level} ${className}` : picker.level}
      role="group"
      aria-label={label}
    >
      <span className={picker.levelName}>{name}</span>
      <div className={picker.choices}>{children}</div>
    </div>
  );
}
