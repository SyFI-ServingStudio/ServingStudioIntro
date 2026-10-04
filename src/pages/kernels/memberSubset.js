/* Which of a preset's members a subset is, in the fewest axes: the axes whose
   values decide membership, each other axis dropped. `members` is every
   member's params, `inSubset` whether each is in. Returns `{axes, values}`
   when the subset is every combination of `values` (one list per axis),
   `{axes, tuples}` when it is some of them, or null when no axis tells it
   apart (the subset is every member, or none). */
export function memberSubset(members, inSubset) {
  const tuple = (params, axes) => JSON.stringify(axes.map((axis) => params[axis]));
  const decides = (axes) => {
    const seen = new Map();
    return members.every((params, i) => {
      const key = tuple(params, axes);
      if (seen.has(key) && seen.get(key) !== inSubset[i]) return false;
      seen.set(key, inSubset[i]);
      return true;
    });
  };
  let axes = Object.keys(members[0] ?? {}).filter(
    (axis) => new Set(members.map((params) => params[axis])).size > 1,
  );
  if (!decides(axes)) return null;
  for (const axis of [...axes]) {
    const fewer = axes.filter((a) => a !== axis);
    if (decides(fewer)) axes = fewer;
  }
  if (!axes.length) return null;
  const chosen = members.filter((_, i) => inSubset[i]);
  const values = axes.map((axis) => [
    ...new Set(chosen.map((params) => params[axis])),
  ]);
  const product = members.every(
    (params, i) =>
      inSubset[i] === axes.every((axis, j) => values[j].includes(params[axis])),
  );
  if (product) return { axes, values };
  const tuples = [...new Set(chosen.map((params) => tuple(params, axes)))].map(
    (t) => JSON.parse(t),
  );
  return { axes, tuples };
}
