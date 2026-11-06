import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import {
  amplificationText,
  bytesText,
  capacity,
  engineText,
  hasSizes,
  layoutRows,
  layoutSizes,
  lengthStops,
  placementText,
  placementWords,
  spreadLabels,
  requestRows,
  vllmAddsText,
} from "../src/components/kvCache.js";

// Pools of /simulations/presets as the public API served them.
const kv = JSON.parse(
  readFileSync(new URL("./fixtures/kv-cache.json", import.meta.url)),
);

const held = (pool) =>
  requestRows(pool).map((row) => [row.name, row.holds, bytesText(row.bytes)]);

test("bytes print to three significant figures in binary units", () => {
  assert.equal(bytesText(512), "512 B");
  assert.equal(bytesText(45056), "44 KiB");
  assert.equal(bytesText(2105771008), "1.96 GiB");
});

test("a window holds its last W - 1 tokens plus the step in flight", () => {
  assert.deepEqual(held(kv.inkling), [
    ["global_attn", "131,072 tokens", "704 MiB"],
    ["swa_attn", "4,607 tokens (511 + 4,096 in flight)", "247 MiB"],
    ["sconv", "4,099 tokens (3 + 4,096 in flight)", "1.03 GiB"],
  ]);
  const facts = capacity(kv.inkling);
  assert.equal(facts.gpusPerPartition, 8);
  assert.equal(facts.tokens, null); // windows are no token pool
  assert.ok(Math.abs(facts.concurrency - 37.66) < 0.01);
});

test("blocks round a per-token part up and a state keeps the previous step's copy", () => {
  assert.deepEqual(held(kv.glm53_flash_tp), [
    ["mla", "8,704 tokens (4 blocks of 2,176)", "46.8 MiB"],
    ["indexer", "8,704 tokens, 2,176 entries", "3.01 MiB"],
    ["kda", "2 copies (1 + the previous step's)", "99.5 MiB"],
    ["kpool_tail", "1 copy", "12.4 MiB"],
  ]);
  assert.equal(
    engineText(kv.glm53_flash_tp),
    "vLLM allocates in blocks of 2,176 tokens, 12.4 MiB set aside.",
  );
  const rows = layoutRows(kv.glm53_flash_tp);
  assert.deepEqual(
    rows.map((row) => [row.name, row.grows, row.bytes, row.overhead, row.reusable]),
    [
      ["mla", "Every token", "5.5 KiB per token", "", true],
      ["indexer", "Every 4 tokens", "1.42 KiB per 4 tokens", "", true],
      [
        "kda",
        "Once per request",
        "35.2 MiB per request",
        "padded to 49.8 MiB",
        true,
      ],
      [
        "kpool_tail",
        "Once per request",
        "22 KiB per request",
        "padded to 12.4 MiB",
        false,
      ],
    ],
  );
});

test("attention DP partitions each hold their own cache", () => {
  const facts = capacity(kv.glm52_dp);
  assert.equal(facts.partitions, 4);
  assert.equal(facts.tokens, 410876);
});

test("a pool whose cache cannot hold one full-length request reads below one", () => {
  const facts = capacity(kv.glm52_fp8_hp);
  assert.equal(facts.maxModelLen, 1048576);
  assert.ok(facts.concurrency < 1);
  assert.ok(facts.perRequest > facts.budget);
});

test("placement and amplification print as the simulator's report does", () => {
  assert.equal(placementText({ shards: 4, replicas: 1 }), "4 shards");
  assert.equal(placementText({ shards: 1, replicas: 4 }), "4 replicas");
  assert.equal(placementText({ shards: 2, replicas: 2 }), "2 shards × 2 replicas");
  // GLM-5.3 TP4's numbers from the simulator's --kv-report.
  assert.equal(
    amplificationText({ replication: 3.5255512, amplification: 3.9131768 }),
    "3.53× replication × 1.11× engine overhead = 3.91×",
  );
});

test("each size comes from the API: stored once, per GPU and over the partition", () => {
  const pool = kv.glm53_flash_tp;
  assert.ok(hasSizes(pool));
  const sizes = layoutSizes(pool, 8192);
  const three = (size) => [size.once, size.gpu, size.partition];
  // GLM-5.3-Flash on attention TP4: MLA, indexer and the kpool tail are kept
  // whole on each GPU, KDA heads split across them.
  assert.deepEqual(three(sizes.token), [
    5632 + 1452 / 4,
    5632 + 1452 / 4,
    4 * (5632 + 1452 / 4),
  ]);
  assert.deepEqual(three(sizes.request), [
    4 * 36904960 + 22528,
    36904960 + 22528,
    4 * 36904960 + 4 * 22528,
  ]);
  assert.equal(sizes.window, null);
  assert.deepEqual(
    sizes.rows.map((row) => [row.name, placementWords(row.placement, 4)]),
    [
      ["mla", "A whole copy on each of the 4 GPUs"],
      ["indexer", "A whole copy on each of the 4 GPUs"],
      ["kda", "Split across the 4 GPUs"],
      ["kpool_tail", "A whole copy on each of the 4 GPUs"],
    ],
  );
  // The request's per-GPU bytes on the Simulate page are the API's too.
  assert.equal(
    capacity(pool).perRequest,
    pool.result.request_parts.reduce((sum, part) => sum + part.per_gpu_bytes, 0),
  );
});

test("one GPU's cache has one size, and placement says nothing", () => {
  const sizes = layoutSizes(kv.glm52_dp, 8192);
  assert.equal(sizes.total.once, sizes.total.gpu);
  assert.equal(sizes.total.gpu, sizes.total.partition);
  assert.equal(placementWords(sizes.rows[0].placement, 1), null);
  assert.equal(
    placementWords({ shards: 2, replicas: 2 }, 4),
    "Split in 2, each slice on 2 GPUs",
  );
});

test("what vLLM adds is named, not counted, on the Models page", () => {
  assert.equal(
    vllmAddsText(kv.glm53_flash_tp),
    "vLLM pads kda by 41%, pads kpool_tail to 579× its size and allocates in blocks of 2,176 tokens.",
  );
  assert.equal(vllmAddsText(kv.inkling), "vLLM pads sconv by 2%.");
  assert.equal(vllmAddsText(kv.glm52_dp), null);
});

test("the slider steps by powers of two and halves up to the longest context", () => {
  assert.deepEqual(lengthStops(8192), [1024, 1536, 2048, 3072, 4096, 6144, 8192]);
  assert.deepEqual(lengthStops(9216).slice(-3), [6144, 8192, 9216]);
  assert.deepEqual(lengthStops(512), [512]);
});

const at = (pool, tokens) =>
  layoutSizes(pool, tokens).rows.map((row) => [
    row.name,
    row.holds,
    bytesText(row.gpu),
  ]);

test("a decoding request at a picked length holds the model's own bytes", () => {
  // No padding, blocks or extra copies: those are the Simulate page's.
  assert.deepEqual(at(kv.glm53_flash_tp, 8192), [
    ["mla", "8,192 tokens", "44 MiB"],
    ["indexer", "8,192 tokens, 2,048 entries", "2.84 MiB"],
    ["kda", "1 state", "35.2 MiB"],
    ["kpool_tail", "1 state", "22 KiB"],
  ]);
  assert.deepEqual(at(kv.inkling, 131072), [
    ["global_attn", "131,072 tokens", "704 MiB"],
    ["swa_attn", "512 tokens, a full window", "27.5 MiB"],
    ["sconv", "4 tokens, a full window", "1.01 MiB"],
  ]);
  assert.deepEqual(at(kv.inkling, 300), [
    ["global_attn", "300 tokens", "1.61 MiB"],
    ["swa_attn", "300 tokens", "16.1 MiB"],
    ["sconv", "4 tokens, a full window", "1.01 MiB"],
  ]);
  // Every part grows per token: the shares stay put and bytes scale.
  const short = layoutSizes(kv.glm52_dp, 1024);
  const long = layoutSizes(kv.glm52_dp, 8192);
  assert.equal(long.total.gpu, short.total.gpu * 8);
  assert.equal(long.rows[0].share, short.rows[0].share);
  // Without overhead, the longest context matches the server's split.
  const server = kv.glm52_fp8_hp.result.request_parts.reduce(
    (sum, part) => sum + part.bytes,
    0,
  );
  assert.equal(layoutSizes(kv.glm52_fp8_hp, 1048576).total.partition, server);
});

test("labels in one row spread apart and stay inside it", () => {
  // Far apart: each sits centred on its anchor.
  assert.deepEqual(spreadLabels([50, 250], [40, 40], 300), [30, 230]);
  // Crowded: the second moves right past the first, 8 px apart.
  assert.deepEqual(spreadLabels([100, 110], [60, 60], 300), [70, 138]);
  // At the right edge: both pulled back inside.
  assert.deepEqual(spreadLabels([280, 295], [60, 60], 300), [172, 240]);
  // At the left edge: kept off it.
  assert.deepEqual(spreadLabels([5], [60], 300), [0]);
});

test("padding under one percent does not read as none", () => {
  const pool = structuredClone(kv.glm52_dp);
  pool.layout.parts[0].padded_bytes_per_entry = Math.round(
    pool.layout.parts[0].bytes_per_entry * 1.002,
  );
  assert.equal(vllmAddsText(pool), "vLLM pads mla by under 1%.");
});
