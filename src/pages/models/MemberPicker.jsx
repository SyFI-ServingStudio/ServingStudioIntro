import { CircleDashed } from "lucide-react";
import { Tag, ToggleTag } from "../kernels/Tag";
import { closestMember, memberMatches, paramValue, sameValue } from "./modelData";
import picker from "../kernels/ConfigPicker.module.css";
import s from "./Models.module.css";

/* One value per axis of a preset, over the members that exist: a level per
   axis, a chip per value. A value no member pairs with the other current
   values is faint; picking it moves to the member that keeps the most of
   them. `blocked(member)`, when it returns a reason, marks the chip that
   leads to that member. `label(axis)` and `valueTitle(axis, value)` let a
   page name an axis or explain a value. Used by the Models page and the
   Kernels page's chart over several configs. */
export function MemberPicker({
  axes,
  members,
  current,
  onPick,
  blocked = () => null,
  label = (axis) => axis.name,
  valueTitle = () => undefined,
}) {
  const names = axes.map((axis) => axis.name);
  return axes.map((axis) => {
    const others = names.filter((name) => name !== axis.name);
    return (
      <div
        key={axis.name}
        className={picker.level}
        role="group"
        aria-label={axis.name}
      >
        <span className={picker.levelName}>
          <code className={s.axisName}>{label(axis)}</code>
        </span>
        <div className={picker.choices}>
          {axis.values.length === 1 ? (
            <Tag type="choice" value={String(axis.values[0])}>
              {paramValue(axis.values[0])}
            </Tag>
          ) : (
            axis.values.map((value) => {
              const exact = members.find(
                (m) =>
                  sameValue(m.params[axis.name], value) &&
                  memberMatches(m, current, others),
              );
              const target =
                exact ?? closestMember(members, names, current, axis.name, value);
              if (!target) return null;
              const reason = blocked(target);
              const title = [
                valueTitle(axis, value),
                !exact && "No member pairs it with the other values picked",
                reason,
              ]
                .filter(Boolean)
                .join("\n");
              return (
                <ToggleTag
                  key={String(value)}
                  type="choice"
                  value={String(value)}
                  pressed={sameValue(current[axis.name], value)}
                  faint={!exact}
                  title={title || undefined}
                  onClick={() => onPick(target.params)}
                >
                  <span className={s.axisChip}>
                    {paramValue(value)}
                    {reason && <CircleDashed size={13} aria-label={reason} />}
                  </span>
                </ToggleTag>
              );
            })
          )}
        </div>
      </div>
    );
  });
}
