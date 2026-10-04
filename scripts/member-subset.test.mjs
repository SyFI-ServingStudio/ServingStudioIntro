import assert from "node:assert/strict";
import { test } from "node:test";
import { memberSubset } from "../src/pages/kernels/memberSubset.js";

const grid = [];
for (const ep_size of [4, 8])
  for (const max_model_len of [8192, 65536])
    for (const workload of ["a", "b"])
      grid.push({ ep_size, max_model_len, workload });

test("a subset one axis decides names only that axis", () => {
  const inSubset = grid.map((p) => p.ep_size === 4);
  assert.deepEqual(memberSubset(grid, inSubset), {
    axes: ["ep_size"],
    values: [[4]],
  });
});

test("a subset that is not a product names the combinations of its axes", () => {
  const inSubset = grid.map(
    (p) => (p.ep_size === 4) === (p.max_model_len === 8192),
  );
  assert.deepEqual(memberSubset(grid, inSubset), {
    axes: ["ep_size", "max_model_len"],
    tuples: [
      [4, 8192],
      [8, 65536],
    ],
  });
});

test("every member needs no description", () => {
  assert.equal(
    memberSubset(
      grid,
      grid.map(() => true),
    ),
    null,
  );
});
