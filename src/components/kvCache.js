/* What a deployment's KV cache holds, from the public API.

   The arch declares its cache part by part (`kv_cache.layout`): each part
   grows with every token of context (or every few tokens, when compressed),
   keeps only a sliding window of recent tokens, or holds one state per
   request. Bytes are summed over the GPUs one KV partition spans
   (`attention_shards`); everything shown here is per GPU.

   A simulation member also carries what its worker makes of that layout
   (the run's `raw/kv_cache.json`): the memory budget, whether vLLM's own
   overhead is charged, and what one request at the longest context holds,
   part by part, hence how many such requests fit. */

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

/* Bytes over one partition, on one of its GPUs. */
export const perGpu = (kv, bytes) => bytes / shards(kv);

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
   own bytes, and what vLLM allocates on top of them. */
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
      bytes: `${bytesText(perGpu(kv, part.bytes_per_entry))} ${per}`,
      overhead: extra.join(", "),
      reusable: part.prefix_reusable,
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

/* The arch's cache at a glance, per GPU: what each token of context costs
   in the parts that grow with it, the most the windowed parts hold, and the
   state each request keeps. A kind the layout lacks is null. */
export function layoutTotals(kv) {
  let token = null;
  let window = null;
  let request = null;
  for (const part of kv.layout.parts) {
    const bytes = perGpu(kv, part.bytes_per_entry);
    if (part.growth.kind === "per_token")
      token = (token ?? 0) + bytes / part.growth.tokens_per_entry;
    else if (part.growth.kind === "sliding_window")
      window = (window ?? 0) + bytes * part.growth.window_tokens;
    else if (part.growth.kind === "per_request") request = (request ?? 0) + bytes;
  }
  return { token, window, request };
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
   previous step's beside the current one, when prefix caching resumes it. */
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
      const copies = charge.charged_copies;
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
      bytes: perGpu(kv, held.bytes),
      share: total > 0 ? held.bytes / total : 0,
    };
  });
}

/* The headline numbers of a simulation member's pool, or null before its
   build: requests of `max_model_len` that fit at once in one partition, the
   same without vLLM's overhead, the budget, and the token pool when every
   part grows per token. */
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
    perRequest: perGpu(
      kv,
      result.request_parts.reduce((sum, part) => sum + part.bytes, 0),
    ),
    tokens: result.capacity_tokens,
    partitions: kv.partitions ?? 1,
    gpusPerPartition: shards(kv),
  };
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

/* One decoding request of `tokens` context, part by part, per GPU, as vLLM
   allocates the layout: padded entries, whole engine blocks, the declared
   state copies, and a window's last W tokens (W − 1 kept plus the one in
   flight). A prefill chunk widens each window by its tokens, and prefix
   caching keeps a reusable state's previous-step copy; both depend on the
   server's settings, so a deployment's figure on the Simulate page shows
   them. */
export function layoutAt(kv, tokens) {
  const block = kv.layout.engine.block_tokens ?? 1;
  const allocated = tokens > 0 ? Math.ceil(tokens / block) * block : 0;
  const rows = kv.layout.parts.map((part) => {
    const entry = part.padded_bytes_per_entry ?? part.bytes_per_entry;
    const growth = part.growth;
    let entries;
    let holds;
    if (growth.kind === "per_token") {
      entries = Math.ceil(allocated / growth.tokens_per_entry);
      holds =
        growth.tokens_per_entry > 1
          ? `${whole(allocated)} tokens, ${whole(entries)} entries`
          : allocated > tokens
            ? `${whole(allocated)} tokens (${whole(allocated / block)} ${allocated === block ? "block" : "blocks"} of ${whole(block)})`
            : `${whole(allocated)} tokens`;
    } else if (growth.kind === "sliding_window") {
      entries = Math.min(tokens, growth.window_tokens);
      holds =
        tokens > growth.window_tokens
          ? `${whole(entries)} tokens, a full window`
          : `${whole(entries)} tokens`;
    } else {
      entries = part.copies ?? 1;
      holds = entries === 1 ? "1 copy" : `${entries} copies`;
    }
    return {
      name: part.name,
      kind: growth.kind,
      holds,
      bytes: perGpu(kv, entries * entry),
    };
  });
  const total = rows.reduce((sum, row) => sum + row.bytes, 0);
  return {
    total,
    rows: rows.map((row) => ({ ...row, share: total > 0 ? row.bytes / total : 0 })),
  };
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
