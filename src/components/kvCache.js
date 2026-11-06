/* What a deployment's KV cache holds, from the public API.

   The arch declares its cache part by part (`kv_cache.layout`): each part
   grows with every token of context (or every few tokens, when compressed),
   keeps only a sliding window of recent tokens, or holds one state per
   request. A KV partition is the GPUs that share one cache
   (`attention_shards`); a part is either split across them or kept whole on
   each, so one entry has three sizes, and the API gives all three:
   `one_copy_bytes_per_entry` (stored once, the least the model needs),
   `per_gpu_bytes_per_entry` (on one GPU) and `bytes_per_entry` (over the
   partition). The site reads them as given and never divides by the
   partition's GPU count to get them.

   A simulation member also carries what its worker makes of that layout
   (the run's `raw/kv_cache.json`): the memory budget, whether vLLM's own
   overhead is charged, and what one request at the longest context holds,
   part by part, hence how many such requests fit.

   A newer simulator also says how that request compares with the model's
   own bytes stored once: each part's placement on the partition's GPUs
   (sharded or replicated), each part's bytes in the three sizes, and the
   request's replication and amplification. These are the simulator's
   numbers, shown as they come. */

/* Binary units to three significant figures, as the simulator prints them. */
export function bytesText(bytes) {
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${Number(value.toPrecision(3))} ${units[unit]}`;
}

const whole = (value) => Math.round(value).toLocaleString("en-US");

const shards = (kv) => Math.max(1, kv.attention_shards ?? 1);

/* A budget-side figure the API gives over the partition only (the memory
   budget, bytes set aside, a padded entry), on one of its GPUs. What the
   cache holds comes per GPU from the API instead. */
export const perGpu = (kv, bytes) => bytes / shards(kv);

/* How a part sits on the partition's GPUs, in words: kept whole on each,
   split across them, or both. Null on a one-GPU partition. */
export function placementWords(placement, gpus) {
  if (!placement || gpus <= 1) return null;
  const { shards: shardCount, replicas } = placement;
  if (replicas === 1) return `Split across the ${gpus} GPUs`;
  if (shardCount === 1) return `A whole copy on each of the ${gpus} GPUs`;
  return `Split in ${shardCount}, each slice on ${replicas} GPUs`;
}

/* How a part sits on the partition's GPUs, as the simulator's text report
   prints it: "4 shards", "4 replicas" or "2 shards × 2 replicas". Null when
   the API predates placement. */
export function placementText(placement) {
  if (!placement) return null;
  const { shards: shardCount, replicas } = placement;
  const plural = (count, noun) => `${count} ${noun}${count === 1 ? "" : "s"}`;
  if (replicas === 1) return plural(shardCount, "shard");
  if (shardCount === 1) return plural(replicas, "replica");
  return `${shardCount} shards × ${replicas} replicas`;
}

/* A ratio over one copy, two decimals as the simulator prints it. */
export const timesText = (value) => (value == null ? "-" : `${value.toFixed(2)}×`);

/* How a part grows, in words, and what one of its entries covers. */
export function growthText(growth) {
  switch (growth.kind) {
    case "per_token":
      return growth.tokens_per_entry > 1
        ? {
            grows: `Every ${whole(growth.tokens_per_entry)} tokens`,
            per: `per ${whole(growth.tokens_per_entry)} tokens`,
          }
        : { grows: "Every token", per: "per token" };
    case "sliding_window":
      return {
        grows: `Last ${whole(growth.window_tokens)} tokens`,
        per: "per token",
      };
    case "per_request":
      return { grows: "Once per request", per: "per request" };
    default:
      return { grows: growth.kind, per: "per entry" };
  }
}

/* The layout as table rows, per GPU: what each part is, how it grows, its
   own bytes, what vLLM allocates on top of them, and how it sits on the
   partition's GPUs (null without placement). */
export function layoutRows(kv) {
  return kv.layout.parts.map((part) => {
    const { grows, per } = growthText(part.growth);
    const extra = [];
    if (part.padded_bytes_per_entry != null)
      extra.push(`padded to ${bytesText(perGpu(kv, part.padded_bytes_per_entry))}`);
    if (part.copies != null && part.copies > 1) extra.push(`${part.copies} copies`);
    return {
      name: part.name,
      kind: part.growth.kind,
      grows,
      bytes: `${bytesText(part.per_gpu_bytes_per_entry)} ${per}`,
      overhead: extra.join(", "),
      reusable: part.prefix_reusable,
      placement: placementText(part.placement),
    };
  });
}

/* The engine's own terms, when it declares any. */
export function engineText(kv) {
  const engine = kv.layout.engine;
  const terms = [];
  if (engine.block_tokens != null)
    terms.push(`blocks of ${whole(engine.block_tokens)} tokens`);
  if (engine.fixed_device_bytes > 0)
    terms.push(`${bytesText(perGpu(kv, engine.fixed_device_bytes))} set aside`);
  return terms.length ? `vLLM allocates in ${terms.join(", ")}.` : null;
}

/* The three sizes the Models page shows a figure in: stored once, on one
   GPU and over the partition, the API's per-entry sizes times the entries. */
const sized = (part, entries) => ({
  once: entries * part.one_copy_bytes_per_entry,
  gpu: entries * part.per_gpu_bytes_per_entry,
  partition: entries * part.bytes_per_entry,
});

const ZERO = { once: 0, gpu: 0, partition: 0 };
const plus = (a, b) => ({
  once: a.once + b.once,
  gpu: a.gpu + b.gpu,
  partition: a.partition + b.partition,
});

/* Whether the API gives every part's three sizes. */
export const hasSizes = (kv) =>
  kv.layout.parts.every(
    (part) =>
      part.one_copy_bytes_per_entry != null && part.per_gpu_bytes_per_entry != null,
  );

/* The arch's cache, in the model's own bytes (vLLM's padding, blocks and
   extra copies left to the Simulate page), each figure in the three sizes:
   what each token of context adds in the parts that grow with it, the
   windowed parts when full, the state each request keeps, and one decoding
   request of `tokens` context part by part. A kind the layout lacks is null.
   A row's `share` is of the request on one GPU. */
export function layoutSizes(kv, tokens) {
  let token = null;
  let window = null;
  let request = null;
  const rows = kv.layout.parts.map((part) => {
    const growth = part.growth;
    let entries;
    let holds;
    if (growth.kind === "per_token") {
      token = plus(token ?? ZERO, sized(part, 1 / growth.tokens_per_entry));
      entries = Math.ceil(tokens / growth.tokens_per_entry);
      holds =
        growth.tokens_per_entry > 1
          ? `${whole(tokens)} tokens, ${whole(entries)} entries`
          : `${whole(tokens)} tokens`;
    } else if (growth.kind === "sliding_window") {
      window = plus(window ?? ZERO, sized(part, growth.window_tokens));
      entries = Math.min(tokens, growth.window_tokens);
      holds =
        tokens > growth.window_tokens
          ? `${whole(entries)} tokens, a full window`
          : `${whole(entries)} tokens`;
    } else {
      request = plus(request ?? ZERO, sized(part, 1));
      entries = 1;
      holds = "1 state";
    }
    return {
      name: part.name,
      kind: growth.kind,
      holds,
      perEntry: growthText(growth).per,
      entry: sized(part, 1),
      placement: part.placement,
      reusable: part.prefix_reusable,
      ...sized(part, entries),
    };
  });
  const total = rows.reduce(plus, ZERO);
  return {
    token,
    window,
    request,
    total,
    rows: rows.map((row) => ({
      ...row,
      share: total.gpu > 0 ? row.gpu / total.gpu : 0,
    })),
  };
}

/* What vLLM adds on top of the model's own bytes, in one line: padded
   entries as a percentage (as a multiple from twice the size), extra state
   copies and its block size. Null when
   it adds nothing the layout declares. */
export function vllmAddsText(kv) {
  const terms = [];
  for (const part of kv.layout.parts) {
    if (part.padded_bytes_per_entry != null && part.bytes_per_entry > 0) {
      const ratio = part.padded_bytes_per_entry / part.bytes_per_entry;
      terms.push(
        ratio >= 2
          ? `pads ${part.name} to ${whole(ratio)}× its size`
          : `pads ${part.name} by ${Math.round((ratio - 1) * 100)}%`,
      );
    }
    if (part.copies != null && part.copies > 1)
      terms.push(`keeps ${part.copies} copies of ${part.name}`);
  }
  const block = kv.layout.engine.block_tokens;
  if (block != null) terms.push(`allocates in blocks of ${whole(block)} tokens`);
  if (!terms.length) return null;
  const last = terms.pop();
  return `vLLM ${terms.length ? `${terms.join(", ")} and ${last}` : last}.`;
}

const RESUME = {
  prompt_end: "at the end of each prompt",
  response_end: "at the end of each response",
  prompt_and_response_end: "at the end of each prompt and response",
  every_block: "at every block",
};

/* The worker's choices that shape the charge, in one line. */
export function optionsText(kv) {
  const options = kv.options;
  const text = [
    options.kv_overhead === "vllm"
      ? "Charged as vLLM allocates it"
      : "Charged at the model's own bytes, without vLLM's overhead",
  ];
  if (options.prefix_caching) {
    let resume = `prefix caching resumes ${RESUME[options.prefix_resume_points] ?? options.prefix_resume_points}`;
    if (options.prefix_resume_interval_tokens != null)
      resume += ` and every ${whole(options.prefix_resume_interval_tokens)} tokens`;
    text.push(resume);
  } else {
    text.push("no prefix caching");
  }
  return `${text.join("; ")}.`;
}

/* One request at the longest context, part by part, per GPU: what each part
   holds and its share. vLLM keeps a window's last W − 1 tokens plus the
   tokens the step in flight adds; a per-request state may be kept twice, the
   previous step's beside the current one, when prefix caching resumes it.
   `oneCopy` is the model's own bytes of the part, stored once, over the
   whole model rather than per GPU; `amplified` is the partition's bytes
   over it. Both are null without `one_copy_bytes`. */
export function requestRows(kv) {
  const result = kv.result;
  const vllm = kv.options.kv_overhead === "vllm";
  const total = result.request_parts.reduce((sum, part) => sum + part.bytes, 0);
  return result.request_parts.map((held, index) => {
    const part = kv.layout.parts[index];
    const charge = kv.accounting.parts[index];
    const tokens = held.tokens_held;
    let holds;
    if (tokens == null) {
      // The copies a running request holds, plus align mode's previous-step
      // copy that vLLM's concurrency estimate adds.
      const copies = charge.held_copies + (charge.align_copy === true ? 1 : 0);
      holds =
        charge.align_copy === true
          ? `${copies} copies (${copies - 1} + the previous step's)`
          : copies === 1
            ? "1 copy"
            : `${copies} copies`;
    } else if (
      part.growth.kind === "sliding_window" &&
      tokens < result.max_model_len
    ) {
      const window = part.growth.window_tokens;
      holds =
        vllm && tokens > window - 1
          ? `${whole(tokens)} tokens (${whole(window - 1)} + ${whole(tokens - (window - 1))} in flight)`
          : `${whole(tokens)} tokens`;
    } else if (
      part.growth.kind === "per_token" &&
      part.growth.tokens_per_entry > 1
    ) {
      holds = `${whole(tokens)} tokens, ${whole(held.entries)} entries`;
    } else if (part.growth.kind === "per_token" && tokens > result.max_model_len) {
      const block = kv.accounting.charged_block_tokens;
      holds = `${whole(tokens)} tokens (${whole(tokens / block)} ${tokens === block ? "block" : "blocks"} of ${whole(block)})`;
    } else {
      holds = `${whole(tokens)} tokens`;
    }
    return {
      name: held.name,
      kind: part.growth.kind,
      holds,
      bytes: held.per_gpu_bytes,
      share: total > 0 ? held.bytes / total : 0,
      oneCopy: held.one_copy_bytes ?? null,
      amplified: held.one_copy_bytes > 0 ? held.bytes / held.one_copy_bytes : null,
    };
  });
}

/* The headline numbers of a simulation member's pool, or null before its
   build: requests of `max_model_len` that fit at once in one partition, the
   same without vLLM's overhead, the budget, the token pool when every part
   grows per token, and what one such request holds: per GPU, stored once
   (`oneCopy`), and how many times over the partition stores it
   (`replication` by placement, `amplification` with the engine's overhead
   too). The last three are null when the API does not send them. */
export function capacity(kv) {
  const result = kv?.result;
  if (
    !result ||
    result.max_model_len == null ||
    result.max_concurrency_at_max_model_len == null
  )
    return null;
  const budget = perGpu(kv, kv.accounting.budget_bytes);
  const reserved = perGpu(
    kv,
    kv.accounting.reserved_units * kv.accounting.bytes_per_unit,
  );
  return {
    maxModelLen: result.max_model_len,
    concurrency: result.max_concurrency_at_max_model_len,
    other: result.other_max_concurrency_at_max_model_len,
    otherMode: result.other_kv_overhead,
    budget,
    reserved,
    perRequest: result.per_gpu_bytes_at_max_model_len,
    tokens: result.capacity_tokens,
    oneCopy: result.one_copy_bytes_at_max_model_len ?? null,
    replication: result.replication_at_max_model_len ?? null,
    amplification: result.amplification_at_max_model_len ?? null,
    partitions: kv.partitions ?? 1,
    gpusPerPartition: shards(kv),
  };
}

/* The amplification split as the simulator's text report prints it:
   "3.53× replication × 1.11× engine overhead = 3.91×", the overhead being
   amplification over replication. Null unless the API sends both. */
export function amplificationText(facts) {
  const { replication, amplification } = facts;
  if (replication == null || amplification == null || replication <= 0) return null;
  return `${timesText(replication)} replication × ${timesText(amplification / replication)} engine overhead = ${timesText(amplification)}`;
}

/* A concurrency as vLLM logs it: two decimals, below one a request does not fit. */
export const concurrencyText = (value) =>
  value == null ? "-" : `${value >= 100 ? whole(value) : value.toFixed(2)}×`;

/* Context lengths a slider steps through: each power of two from 1,024
   tokens and the step halfway to the next, up to the longest context, which
   is always the last stop. */
export function lengthStops(maxTokens) {
  const stops = [];
  for (let base = 1024; base < maxTokens; base *= 2) {
    stops.push(base);
    if (base * 1.5 < maxTokens) stops.push(base * 1.5);
  }
  stops.push(maxTokens);
  return stops;
}

/* Left edges for labels in one row, in pixels: each centred on its anchor,
   pushed right past the label before it with `gap` between, then pulled
   back inside `span` from the right, then kept off the left edge. Labels
   wider together than the row overlap; nothing else does. */
export function spreadLabels(anchors, widths, span, gap = 8) {
  const lefts = anchors.map((anchor, index) => anchor - widths[index] / 2);
  for (let index = 0; index < lefts.length; index += 1) {
    const floor = index ? lefts[index - 1] + widths[index - 1] + gap : 0;
    lefts[index] = Math.max(lefts[index], floor);
  }
  for (let index = lefts.length - 1; index >= 0; index -= 1) {
    const ceiling =
      index === lefts.length - 1
        ? span - widths[index]
        : lefts[index + 1] - gap - widths[index];
    lefts[index] = Math.max(0, Math.min(lefts[index], ceiling));
  }
  return lefts;
}
