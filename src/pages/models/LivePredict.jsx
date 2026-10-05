import { LoaderCircle, Plus, RotateCcw, TriangleAlert, X } from "lucide-react";
import { useEffect, useId, useMemo, useRef, useState } from "react";
import {
  archLabel,
  blockedReason,
  formatMs,
  paramValue,
  paramsText,
  predict,
  predictable,
  sameValue,
} from "./modelData";
import { presetCheckpoint } from "../kernels/kernelData";
import { ReadMoreButton, ResultOverlay } from "./ReadMore";
import { Pending } from "./TreeParts";
import cost from "../../components/CostTree.module.css";
import s from "./Models.module.css";

/* Live predict: a batch of requests for the member the cost tree beside it
   shows, timed by the data service (POST /predict, Sim's timing-predict on
   the measured rows) a moment after each edit. The answer is here; every
   node's time goes onto the tree through `onTimes`.

   The editor follows the member's prediction shape (`member.predict`: the
   selector, its data-parallel groups, a speculative member's verify width)
   and the fields /models gives each selector (`caseFields`). */
export function LivePredict({
  preset,
  member,
  tree,
  caseFields,
  batch,
  setBatch,
  onPick,
  onTimes,
  onPending,
}) {
  const key = JSON.stringify([preset.id, member.params]);
  useEffect(() => {
    onTimes(null);
    onPending(false);
  }, [key, onTimes, onPending]);
  const shape = useMemo(
    () => member.predict && editorShape(member.predict, caseFields),
    [member.predict, caseFields],
  );

  const reason = blockedReason(member);
  let blocked = null;
  if (reason)
    blocked = [
      "This configuration cannot be timed yet",
      <p key="m">
        {reason}. Predictions use measurements only; pick a configuration without
        the dashed mark above.
      </p>,
    ];
  else if (shape?.problem)
    blocked = [
      "This page cannot edit this prediction",
      <p key="m">{shape.problem}</p>,
    ];

  return (
    <section className={s.livePredict} aria-labelledby="predict-title">
      <div className={s.predictHead}>
        <h2 id="predict-title">Live predict</h2>
      </div>
      {blocked ? (
        <div className={s.predictStatus} data-tone="warn" role="alert">
          <TriangleAlert size={18} aria-hidden="true" />
          <div>
            <p className={s.predictStatusTitle}>{blocked[0]}</p>
            {blocked[1]}
          </div>
        </div>
      ) : !tree || tree.error ? (
        <p className={s.predictQuiet}>Waiting for the cost tree.</p>
      ) : (
        <Workbench
          key={key}
          preset={preset}
          member={member}
          tree={tree}
          shape={shape}
          batch={batch}
          setBatch={setBatch}
          onPick={onPick}
          onTimes={onTimes}
          onPending={onPending}
        />
      )}
    </section>
  );
}

/* What the editor builds for a selector, from its case fields: a list of
   tokens per group (an FFN side), or per group prefill requests and decode
   requests, the latter as KV lengths or, for a speculative member, as
   [KV length, verify width] pairs. */
function editorShape(info, caseFields) {
  const fields = caseFields?.[info.selector];
  if (!fields)
    return {
      problem: `The data service names no case fields for ${info.selector}.`,
    };
  const byName = (list) => new Map((list ?? []).map((f) => [f.name, f]));
  const top = byName(fields.case);
  const group = byName(fields.groups);
  if (top.has("tokens_per_group"))
    return { mode: "tokens", fields: { tokens: top.get("tokens_per_group") } };
  const prefill = group.get("prefill_chunk_pairs");
  const decode = group.get("decode_requests") ?? group.get("decode_kv_lens");
  if (!prefill || !decode)
    return {
      problem: `A ${info.selector} case takes ${[...top.keys(), ...group.keys()].join(", ")}, which this page has no editor for.`,
    };
  if (decode.name === "decode_requests" && info.query_width == null)
    return {
      problem:
        "This configuration names no verify width, so its decode requests have no shape.",
    };
  return { mode: "requests", fields: { prefill, decode } };
}

// A field's description, its `identifiers` read as plain words in a tooltip.
const plain = (field) => field?.description?.replaceAll("`", "");

/* ---------- The batch ---------- */

// A group's requests; `tokens` is what an FFN side reads of it instead.
const newGroup = () => ({ prefills: [], decodes: [], tokens: "256" });
const clamp = (tokens, limit) => String(limit ? Math.min(tokens, limit) : tokens);

function batches(limit) {
  const len = (tokens) => clamp(tokens, limit);
  return [
    [
      "1 prefill of 4k",
      {
        prefills: [{ prefix: "0", append: len(4096) }],
        decodes: [],
        tokens: "4096",
      },
    ],
    [
      "256 decodes at 4k",
      {
        prefills: [],
        decodes: [{ count: "256", context: len(4096) }],
        tokens: "256",
      },
    ],
    [
      "Mixed: 1 prefill of 2k, 64 decodes at 8k",
      {
        prefills: [{ prefix: "0", append: len(2048) }],
        decodes: [{ count: "64", context: len(8192) }],
        tokens: "2112",
      },
    ],
  ];
}

const defaultBatch = (limit) => ({
  mirror: true,
  current: 0,
  groups: [batches(limit)[2][1]],
});

// Past this many requests in one batch the request body grows past what a
// reader would wait on.
const MAX_REQUESTS = 100_000;
const WHOLE = /^\d+$/;

/* The batch as one case of the member's selector, or why it is not one yet. */
function toCase(groups, shape, queryWidth) {
  if (shape.mode === "tokens") {
    if (groups.some((group) => !WHOLE.test(group.tokens)))
      return { problem: "Every group takes a whole number of tokens." };
    const tokens = groups.map((group) => Number(group.tokens));
    if (!tokens.some(Boolean)) return { problem: "Send the layer some tokens." };
    return { case: { [shape.fields.tokens.name]: tokens } };
  }
  let requests = 0;
  const bad = groups.some((group) =>
    [
      ...group.prefills.flatMap((p) => [p.prefix, p.append]),
      ...group.decodes.flatMap((d) => [d.count, d.context]),
    ].some((value) => !WHOLE.test(value)),
  );
  if (bad)
    return { problem: "Every field takes a whole number of tokens or requests." };
  const verify = shape.fields.decode.name === "decode_requests";
  const shaped = groups.map((group) => {
    const pairs = group.prefills.map((p) => [Number(p.prefix), Number(p.append)]);
    const decodes = group.decodes.flatMap((d) =>
      Array.from({ length: Math.min(Number(d.count), MAX_REQUESTS + 1) }, () =>
        Number(d.context),
      ),
    );
    requests += pairs.length + decodes.length;
    return {
      [shape.fields.prefill.name]: pairs,
      [shape.fields.decode.name]: verify
        ? decodes.map((kv) => [kv, queryWidth])
        : decodes,
    };
  });
  if (requests > MAX_REQUESTS)
    return {
      problem: `${requests.toLocaleString("en-US")} requests is more than this page sends at once (${MAX_REQUESTS.toLocaleString("en-US")}).`,
    };
  if (!requests) return { problem: "Add a request to predict the iteration." };
  return { case: { groups: shaped } };
}

function Workbench({
  preset,
  member,
  tree,
  shape,
  batch,
  setBatch,
  onPick,
  onTimes,
  onPending,
}) {
  const info = member.predict;
  const groupCount = Math.max(1, info.groups ?? 1);
  // The context the simulator checks each case against; a shape of token
  // counts (an FFN side) has none.
  const limit = info.max_model_len ?? null;
  const queryWidth = info.query_width ?? null;
  const plan = batch ?? defaultBatch(limit);
  const groups = Array.from(
    { length: groupCount },
    (_, index) => (plan.mirror ? plan.groups[0] : plan.groups[index]) ?? newGroup(),
  );
  const current = plan.mirror ? 0 : Math.min(plan.current, groupCount - 1);
  const edit = (next) => setBatch({ ...plan, ...next });
  const setGroup = (group) => {
    if (plan.mirror) return edit({ groups: [group] });
    const all = groups.slice();
    all[current] = group;
    edit({ groups: all });
  };

  // `groups` is rebuilt every render; its content is what the case follows.
  const groupsText = JSON.stringify(groups);
  const shaped = useMemo(
    () => toCase(JSON.parse(groupsText), shape, queryWidth),
    [groupsText, shape, queryWidth],
  );
  const body = shaped.case
    ? { preset: preset.id, params: member.params, cases: [shaped.case] }
    : null;
  const { result, failure, busy, retry } = usePrediction(body, onTimes, onPending);
  const tokens = shape.mode === "tokens";

  return (
    <div className={s.predictBody}>
      <div className={s.batch}>
        <div className={s.batchHead}>
          <p className={s.batchFacts}>
            {groupCount > 1
              ? `${groupCount} attention data-parallel groups, each with its own batch`
              : "One batch"}
            {limit != null && !tokens && (
              <>
                {" · "}
                <code>max_model_len</code> {paramValue(limit)}
              </>
            )}
            {queryWidth != null && (
              <>
                {" · "}
                verify width {queryWidth}{" "}
                <span className={s.soft}>(draft tokens + 1)</span>
              </>
            )}
          </p>
          {groupCount > 1 && (
            <label className={s.mirror}>
              <input
                type="checkbox"
                checked={plan.mirror}
                onChange={(event) =>
                  edit(
                    event.target.checked
                      ? { mirror: true, groups: [groups[current]] }
                      : {
                          mirror: false,
                          current: 0,
                          groups: Array.from(
                            { length: groupCount },
                            () => groups[0],
                          ),
                        },
                  )
                }
              />
              Same batch in every group
            </label>
          )}
        </div>
        {groupCount > 1 && !plan.mirror && (
          <div className={s.groupTabs} role="radiogroup" aria-label="DP group">
            {groups.map((group, index) => (
              <button
                key={index}
                type="button"
                role="radio"
                aria-checked={index === current}
                onClick={() => edit({ current: index })}
              >
                Group {index + 1}
                <small>
                  {tokens ? `${group.tokens} tokens` : requestCount(group)}
                </small>
              </button>
            ))}
          </div>
        )}
        {tokens ? (
          <TokensEditor
            group={groups[current]}
            onChange={setGroup}
            field={shape.fields.tokens}
          />
        ) : (
          <GroupEditor
            group={groups[current]}
            onChange={setGroup}
            limit={limit}
            fields={shape.fields}
            label={
              groupCount === 1
                ? null
                : plan.mirror
                  ? "Every group"
                  : `Group ${current + 1}`
            }
          />
        )}
      </div>
      <Result
        selector={info.selector}
        shaped={shaped}
        result={result}
        failure={failure}
        busy={busy}
        retry={retry}
        groups={groups}
        groupCount={groupCount}
        tokens={tokens}
        tree={tree}
        preset={preset}
        member={member}
        body={body}
        onPick={onPick}
      />
    </div>
  );
}

function requestCount(group) {
  const decodes = group.decodes.reduce(
    (sum, d) => sum + (WHOLE.test(d.count) ? Number(d.count) : 0),
    0,
  );
  const n = group.prefills.length + decodes;
  return `${n.toLocaleString("en-US")} ${n === 1 ? "request" : "requests"}`;
}

/* Times `body` whenever it changes, 150 ms after the last edit; a newer
   edit cancels the request still in flight. Each section's tree, as the
   Analyzer gives it, goes to `onTimes`: { section: nodes }, with the
   Analyzer's kernel ranking of the batch (`kernel_time_share`). From the edit
   until the answer the old answer is dropped and `onPending` says the
   tree's times are stale. */
function usePrediction(body, onTimes, onPending) {
  const [state, setState] = useState({});
  const [attempt, setAttempt] = useState(0);
  const text = body ? JSON.stringify(body) : null;
  useEffect(() => {
    if (!text) {
      setState({});
      onTimes(null);
      onPending(false);
      return undefined;
    }
    const controller = new AbortController();
    setState({ busy: true });
    onPending(true);
    const timer = setTimeout(async () => {
      try {
        const answer = await predict(JSON.parse(text), controller.signal);
        const sections = answer.cases[0].sections;
        setState({ result: { sections } });
        onTimes(
          Object.fromEntries(
            sections.map(({ section, nodes }) => [section, nodes]),
          ),
          answer.kernel_time_share,
        );
        onPending(false);
      } catch (error) {
        if (controller.signal.aborted) return;
        setState({ failure: error });
        onTimes(null);
        onPending(false);
      }
    }, 150);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [text, attempt, onTimes, onPending]);
  return { ...state, retry: () => setAttempt((n) => n + 1) };
}

function GroupEditor({ group, onChange, limit, fields, label }) {
  const verify = fields.decode.name === "decode_requests";
  const setRow = (list, index, row) =>
    onChange({
      ...group,
      [list]: group[list].map((r, i) => (i === index ? row : r)),
    });
  const removeRow = (list, index) =>
    onChange({ ...group, [list]: group[list].filter((_, i) => i !== index) });
  const addRow = (list, row) =>
    onChange({ ...group, [list]: [...group[list], row] });
  return (
    <div className={s.groupEditor}>
      <fieldset className={s.requests}>
        <legend title={plain(fields.prefill)}>
          Prefill requests{label && <span className={s.soft}> · {label}</span>}
        </legend>
        {group.prefills.length === 0 && <p className={s.predictQuiet}>None.</p>}
        {group.prefills.map((row, index) => (
          <div className={s.requestRow} key={index}>
            <Field
              label="Cached prefix"
              value={row.prefix}
              onChange={(prefix) => setRow("prefills", index, { ...row, prefix })}
            />
            <Field
              label="New tokens"
              value={row.append}
              onChange={(append) => setRow("prefills", index, { ...row, append })}
            />
            <RemoveButton
              label={`Remove prefill ${index + 1}`}
              onClick={() => removeRow("prefills", index)}
            />
          </div>
        ))}
        <button
          type="button"
          className={s.addButton}
          onClick={() =>
            addRow("prefills", { prefix: "0", append: clamp(2048, limit) })
          }
        >
          <Plus size={16} aria-hidden="true" />
          Add a prefill
        </button>
      </fieldset>
      <fieldset className={s.requests}>
        <legend title={plain(fields.decode)}>
          {verify ? "Decode (verify) requests" : "Decode requests"}
          {label && <span className={s.soft}> · {label}</span>}
        </legend>
        {group.decodes.length === 0 && <p className={s.predictQuiet}>None.</p>}
        {group.decodes.map((row, index) => (
          <div className={s.requestRow} key={index}>
            <Field
              label="Requests"
              value={row.count}
              onChange={(count) => setRow("decodes", index, { ...row, count })}
            />
            <Field
              label={
                verify ? "Context each, verify tokens included" : "Context each"
              }
              value={row.context}
              onChange={(context) => setRow("decodes", index, { ...row, context })}
            />
            <RemoveButton
              label={`Remove decode row ${index + 1}`}
              onClick={() => removeRow("decodes", index)}
            />
          </div>
        ))}
        <button
          type="button"
          className={s.addButton}
          onClick={() =>
            addRow("decodes", { count: "32", context: clamp(4096, limit) })
          }
        >
          <Plus size={16} aria-hidden="true" />
          Add decodes
        </button>
      </fieldset>
      <div className={s.presets} aria-label="Example batches" role="group">
        <span>Start from</span>
        {batches(limit).map(([name, preset]) => (
          <button key={name} type="button" onClick={() => onChange(preset)}>
            {name}
          </button>
        ))}
        <button type="button" onClick={() => onChange(newGroup())}>
          Empty
        </button>
      </div>
    </div>
  );
}

/* An FFN side's batch: the tokens one attention group sends it. */
function TokensEditor({ group, onChange, field }) {
  return (
    <div className={s.groupEditor}>
      <fieldset className={s.requests}>
        <legend title={plain(field)}>Tokens in the batch</legend>
        <div className={s.requestRow}>
          <Field
            label="Tokens from the group"
            value={group.tokens}
            onChange={(tokens) => onChange({ ...group, tokens })}
          />
        </div>
      </fieldset>
    </div>
  );
}

function Field({ label, value, onChange }) {
  const id = useId();
  const invalid = !WHOLE.test(value);
  return (
    <label className={s.field} htmlFor={id}>
      <span>{label}</span>
      <input
        id={id}
        type="text"
        inputMode="numeric"
        autoComplete="off"
        value={value}
        aria-invalid={invalid || undefined}
        onChange={(event) => onChange(event.target.value.replace(/[,_\s]/g, ""))}
      />
    </label>
  );
}

function RemoveButton({ label, onClick }) {
  return (
    <button
      type="button"
      className={s.removeButton}
      aria-label={label}
      title={label}
      onClick={onClick}
    >
      <X size={16} aria-hidden="true" />
    </button>
  );
}

/* ---------- The answer ---------- */

function batchText(groups, groupCount, tokens) {
  const all = groups.slice(0, groupCount);
  if (tokens) {
    const sum = all.reduce((total, g) => total + Number(g.tokens || 0), 0);
    const where = groupCount > 1 ? ` from ${groupCount} groups` : "";
    return `${sum.toLocaleString("en-US")} tokens${where}`;
  }
  const prefills = all.flatMap((g) => g.prefills);
  const appended = prefills.reduce((sum, p) => sum + Number(p.append || 0), 0);
  const decodes = all
    .flatMap((g) => g.decodes)
    .reduce((sum, d) => sum + Number(d.count || 0), 0);
  const parts = [];
  if (prefills.length)
    parts.push(
      `${prefills.length} ${prefills.length === 1 ? "prefill" : "prefills"} (${appended.toLocaleString("en-US")} new tokens)`,
    );
  if (decodes)
    parts.push(
      `${decodes.toLocaleString("en-US")} ${decodes === 1 ? "decode" : "decodes"}`,
    );
  const where = groupCount > 1 ? ` across ${groupCount} groups` : "";
  return `${parts.join(" and ")}${where}`;
}

// What one section's time stands for, by selector.
const HEADLINE = {
  iter: "Predicted iteration time",
  speculative_iter: "Predicted iteration time",
  attn: "Predicted time per attention layer",
};

function Result({
  selector,
  shaped,
  result,
  failure,
  busy,
  retry,
  groups,
  groupCount,
  tokens,
  tree,
  preset,
  member,
  body,
  onPick,
}) {
  const sections = !shaped.problem && !failure ? result?.sections : null;
  // One section is one time. Layer-wise sections (an FFN side's blocks)
  // are each one block's, so they are listed, not added up.
  const single = sections?.length === 1 ? sections[0] : null;
  const extrapolated = sections
    ? sections.reduce((n, item) => n + (item.coverage.extrapolated?.length ?? 0), 0)
    : 0;
  return (
    <div className={s.answer} aria-live="polite">
      <div className={cost.result}>
        <p className={s.answerHead}>
          {HEADLINE[selector] ?? "Predicted time per block"}
          {busy && !shaped.problem && (
            <span className={s.computing} role="status">
              <LoaderCircle size={14} className={s.spin} aria-hidden="true" />
              Computing
            </span>
          )}
        </p>
        {busy && !shaped.problem ? (
          <strong>
            <Pending className={s.pendingHeadline} />
          </strong>
        ) : sections && !single ? (
          <dl className={s.sectionTimes}>
            {sections.map((item) => (
              <div key={`${item.section}:${item.layer}`}>
                <dt>
                  {item.section.replaceAll("_", " ")}
                  {item.layer >= 0 && <small> layer {item.layer}</small>}
                </dt>
                <dd title={`${item.total_ms} ms`} data-ms={item.total_ms}>
                  {formatMs(item.total_ms)} <span>ms</span>
                </dd>
              </div>
            ))}
          </dl>
        ) : (
          <strong
            data-ms={single?.total_ms}
            title={single ? `${single.total_ms} ms` : undefined}
          >
            {single ? (
              <>
                {formatMs(single.total_ms)} <span>ms</span>
              </>
            ) : (
              <span className={s.soft}>–</span>
            )}
          </strong>
        )}
        {shaped.problem ? (
          <p>{shaped.problem}</p>
        ) : (
          <p>
            {batchText(groups, groupCount, tokens)}
            <br />
            {tree.gpus_per_replica ?? "?"} × {tree.gpu.replace(/^NVIDIA /, "")} ·{" "}
            <span title={preset.arch}>{archLabel(preset)}</span>
          </p>
        )}
      </div>
      {failure && !shaped.problem && (
        <Failure
          failure={failure}
          retry={retry}
          preset={preset}
          member={member}
          onPick={onPick}
        />
      )}
      {sections && (!single || extrapolated > 0) && (
        <p className={s.predictNote}>
          {!single &&
            "Each block is timed once: a middle layer stands for every layer but the last."}
          {!single && extrapolated > 0 && " "}
          {extrapolated > 0 &&
            `${extrapolated} kernel ${extrapolated === 1 ? "call reads" : "calls read"} past the measured grid and ${extrapolated === 1 ? "is" : "are"} extrapolated.`}
        </p>
      )}
      {sections && !busy && (
        <PredictReadMore
          body={body}
          name={[
            presetCheckpoint(preset.id),
            archLabel(preset),
            paramsText(member.params),
          ]
            .filter(Boolean)
            .join(" · ")}
        />
      )}
    </div>
  );
}

/* Read more: the batch predicted again with `analyze`, so the data service
   keeps the Analyzer's view of it, and opened in the full result pages. A
   batch already opened is not predicted again. Live edits never analyze. */
function PredictReadMore({ body, name }) {
  const [state, setState] = useState({});
  const opened = useRef(null);
  const key = JSON.stringify(body);
  const open = async () => {
    if (opened.current?.key === key) return setState({ open: opened.current });
    setState({ busy: true });
    try {
      const answer = await predict({ ...body, analyze: true });
      if (!answer.prediction_id)
        throw new Error("The data service kept no analysis of this prediction.");
      opened.current = { key, id: answer.prediction_id };
      setState({ open: opened.current });
    } catch (error) {
      setState({ failure: error.reason ?? error.message });
    }
  };
  return (
    <>
      <div className={s.presets}>
        <ReadMoreButton onClick={open} busy={state.busy} />
      </div>
      {state.failure && (
        <p className={s.predictNote} role="alert">
          {state.failure}
        </p>
      )}
      {state.open && (
        <ResultOverlay
          id={state.open.id}
          name={name}
          onClose={() => setState({})}
        />
      )}
    </>
  );
}

/* A rejected batch, in the service's words. One longer than the member's
   max_model_len offers the members of the preset that go further. */
function Failure({ failure, retry, preset, member, onPick }) {
  const reason = failure.reason ?? failure.message;
  const limit = failure.detail?.too_long?.max_model_len;
  const axis = preset.axes.find((a) => a.name === "max_model_len");
  if (limit != null && axis) {
    const others = preset.axes.filter((a) => a !== axis).map((a) => a.name);
    const larger = preset.members.filter(
      (m) =>
        Number(m.params.max_model_len) > limit &&
        others.every((name) => sameValue(m.params[name], member.params[name])),
    );
    return (
      <div className={s.predictStatus} data-tone="warn" role="alert">
        <TriangleAlert size={18} aria-hidden="true" />
        <div>
          <p className={s.predictStatusTitle}>
            A request is longer than this configuration&apos;s{" "}
            <code>max_model_len</code> {paramValue(limit)}
          </p>
          <p className={s.failureText}>{reason}</p>
          {larger.length ? (
            <>
              <p>Pick a configuration that goes further, or shorten the request:</p>
              <div className={s.suggestions}>
                {larger.map((m) => (
                  <button
                    key={m.params.max_model_len}
                    type="button"
                    onClick={() => onPick(m.params)}
                  >
                    <code>max_model_len</code> {paramValue(m.params.max_model_len)}
                    {!predictable(m) && (
                      <span className={s.soft}> (cannot be timed yet)</span>
                    )}
                  </button>
                ))}
              </div>
            </>
          ) : (
            <p>
              No configuration of this deployment has a larger{" "}
              <code>max_model_len</code>; shorten the request.
            </p>
          )}
        </div>
      </div>
    );
  }
  const title =
    failure.status === 400
      ? "The simulator rejected this batch"
      : failure.status === 409
        ? "This configuration cannot be timed yet"
        : "The prediction failed";
  return (
    <div className={s.predictStatus} data-tone="warn" role="alert">
      <TriangleAlert size={18} aria-hidden="true" />
      <div>
        <p className={s.predictStatusTitle}>{title}</p>
        <p className={s.failureText}>{reason}</p>
        {failure.status !== 400 && failure.status !== 409 && (
          <button type="button" className={s.addButton} onClick={retry}>
            <RotateCcw size={16} aria-hidden="true" />
            Try again
          </button>
        )}
      </div>
    </div>
  );
}
