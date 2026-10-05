import { CircleCheck, CircleDashed, FileUp, LoaderCircle } from "lucide-react";
import { useId, useState } from "react";
import { shortGpu } from "../kernels/kernelData";
import { Tag, ToggleTag } from "../kernels/Tag";
import { MemberPicker } from "../models/MemberPicker";
import { missingText } from "../models/modelData";
import { count, draftTokens, rate, tokens, uploadWorkload } from "./simulateData";
import detail from "../kernels/KernelDetail.module.css";
import picker from "../kernels/ConfigPicker.module.css";
import models from "../models/Models.module.css";
import s from "./Simulate.module.css";

/* What a reader sets before a run, in their terms: which traffic (a
   recording, requests of one shape, or their own file) and how hard it
   comes (a request rate or a concurrency). Every request of the traffic
   runs; its count is the traffic's own. A null rate or concurrency follows
   the traffic: its own arrival times, or all of it in flight. */
export const initialWorkload = {
  source: "capture",
  capture: null,
  custom: { requests: "100", prompt: "1024", output: "256" },
  upload: null,
  load: "rate",
  rate: null,
  concurrency: null,
  acceptRate: "",
};

const SOURCES = [
  ["capture", "Recorded"],
  ["generated", "Your shape"],
  ["upload", "Your file"],
];

// A model routes experts when its captures carry a routing.
export const routes = (preset) => preset.captures.some((c) => c.routing);

/* The capture a run reads: its requests and routing, or only its routing. Until
   the reader picks one, the first the member can run. */
export const captureOf = (preset, workload, member) =>
  preset.captures.find((c) => c.name === workload.capture) ??
  preset.captures.find((c) => !blockedText(member, c)) ??
  preset.captures[0];

/* Why a member cannot run with a capture, in a reader's words; null when it can. */
export function blockedText(member, capture) {
  if (member.error) return `The simulator cannot build it: ${member.error}`;
  const reason = capture && member.unavailable[capture.name];
  if (!reason) return null;
  if (reason.misfit)
    return reason.misfit.requests != null
      ? `Too long here: ${count(reason.misfit.requests)} of ${count(reason.misfit.total)} requests need more than this configuration's ${count(reason.misfit.max_model_len)}-token context`
      : `Its requests do not fit this configuration: ${reason.misfit.reason}`;
  if (reason.missing)
    return `Cannot run yet: measurements missing for ${missingText(reason.missing)}`;
  return `The simulator cannot build it: ${reason.error}`;
}

/* The generator a custom shape draws from: tracegen's `synthetic`, one round
   per session so every request stands alone, at the lengths given. */
export const customGenerator = (custom) => ({
  type: "synthetic",
  sessions: custom.requests,
  rounds: "1",
  input_len: custom.prompt,
  output_len: custom.output,
});

/* What the picked traffic holds, as the service reads it (a capture's or an
   upload's `facts`), or for a custom shape as it was asked: its count and
   lengths, arriving at the generator's own rate. Null until there is one. */
export function trafficFacts(preset, member, workload, workloads) {
  if (workload.source === "capture")
    return captureOf(preset, workload, member)?.facts ?? null;
  if (workload.source === "upload") return workload.upload;
  const own = workloads?.sources?.generated.generators
    .find((g) => g.name === "synthetic")
    ?.arguments.find((a) => a.name === "arrival_rate")?.default;
  return {
    requests: Number(workload.custom.requests),
    sessions: false,
    prompt_tokens: Number(workload.custom.prompt),
    output_tokens: Number(workload.custom.output),
    rate: own == null ? null : Number(own),
  };
}

/* Whether the traffic brings each request's own draft acceptance: a
   recording made on a speculative server, or a file with that column. */
export const carriesAcceptance = (facts) =>
  Boolean(facts?.input_file_tags?.includes("speculative"));

/* Why a number a reader typed cannot be sent; null when it can. */
export function numberProblem(text, { whole = false, max = null } = {}) {
  const value = Number(text);
  if (String(text).trim() === "" || !Number.isFinite(value) || value <= 0)
    return "Give a number above 0.";
  if (whole && !Number.isInteger(value)) return "Give a whole number.";
  if (max != null && value > max) return `At most ${count(max)}.`;
  return null;
}

export function Setup({
  catalog,
  checkpoint,
  preset,
  member,
  onPick,
  workloads,
  workload,
  setWorkload,
}) {
  const capture = captureOf(preset, workload, member);
  const set = (change) => setWorkload((w) => ({ ...w, ...change }));
  return (
    <div className={s.setup}>
      <section
        className={`${detail.controls} ${s.group}`}
        aria-labelledby="deployment-title"
      >
        <h2 id="deployment-title">Deployment</h2>
        <div className={models.picker}>
          <Level name="Model" wrap>
            {catalog.checkpoints.map((c) => (
              <ToggleTag
                key={c.checkpoint}
                type="choice"
                value={c.name}
                pressed={c === checkpoint}
                title={c.checkpoint}
                onClick={() => onPick(c.presets[0].id, null)}
              >
                {c.name}
              </ToggleTag>
            ))}
          </Level>
          <Level name="Configuration" wrap>
            {checkpoint.presets.map((p) => (
              <ToggleTag
                key={p.id}
                type="choice"
                value={p.id}
                pressed={p === preset}
                title={p.id}
                onClick={() => onPick(p.id, member.params)}
              >
                {p.name}
              </ToggleTag>
            ))}
          </Level>
          <Level name="GPU">
            {[...new Set(Object.values(preset.pools).map((p) => p.gpu))].map(
              (gpu) => (
                <Tag key={gpu} type="gpu" value={shortGpu(gpu)} />
              ),
            )}
          </Level>
          {preset.axes.length > 0 && (
            <div className={models.axes}>
              <MemberPicker
                axes={preset.axes}
                members={preset.members}
                current={member.params}
                onPick={(params) => onPick(preset.id, params)}
                blocked={(m) => blockedText(m, capture)}
              />
            </div>
          )}
          <MemberLine preset={preset} member={member} capture={capture} />
        </div>
      </section>

      <Traffic
        preset={preset}
        member={member}
        limits={catalog.limits}
        workloads={workloads}
        workload={workload}
        set={set}
      />
      <Load
        facts={trafficFacts(preset, member, workload, workloads)}
        workload={workload}
        set={set}
      />
    </div>
  );
}

function Level({ name, wrap, children }) {
  return (
    <div
      className={`${picker.level} ${wrap ? models.wrapLevel : ""}`}
      role="group"
      aria-label={name}
    >
      <span className={picker.levelName}>{name}</span>
      <div className={picker.choices}>{children}</div>
    </div>
  );
}

/* The GPUs the picked deployment takes, pool by pool, and whether it runs. */
function MemberLine({ preset, member, capture }) {
  const reason = blockedText(member, capture);
  const pools = Object.entries(member.pools);
  const draft = draftTokens(member);
  return (
    <div className={models.memberStatus} data-state={reason ? "blocked" : "ready"}>
      {reason ? (
        <CircleDashed size={16} aria-hidden="true" />
      ) : (
        <CircleCheck size={16} aria-hidden="true" />
      )}
      <p>
        {member.gpus != null && (
          <>
            {count(member.gpus)} {member.gpus === 1 ? "GPU" : "GPUs"}:{" "}
            {pools
              .map(([role, pool]) =>
                [
                  pools.length > 1 && role,
                  `${pool.replicas} ${pool.replicas === 1 ? "replica" : "replicas"} of ${pool.gpus_per_replica ?? "?"} ${shortGpu(preset.pools[role].gpu)}`,
                ]
                  .filter(Boolean)
                  .join(" "),
              )
              .join(", ")}
            .
          </>
        )}
        {draft != null && <> Drafts {draft} tokens per step.</>}
        {reason && (
          <>
            {" "}
            <b>{reason}</b>
          </>
        )}
      </p>
    </div>
  );
}

/* ---------- Traffic ---------- */

/* What a trace holds, in one line: how many requests, how long. */
export function factsText(facts) {
  const unit = facts.requests === 1 ? "request" : "requests";
  return `${count(facts.requests)} ${unit}${facts.sessions ? " in conversations" : ""}, prompts ${tokens(facts.prompt_tokens)} and outputs ${tokens(facts.output_tokens)} tokens on average`;
}

function Traffic({ preset, member, limits, workloads, workload, set }) {
  const moe = routes(preset);
  const capture = captureOf(preset, workload, member);
  const draft = draftTokens(member);
  const ownAcceptance = carriesAcceptance(
    trafficFacts(preset, member, workload, workloads),
  );
  return (
    <section
      className={`${detail.controls} ${s.group}`}
      aria-labelledby="traffic-title"
    >
      <h2 id="traffic-title">Traffic</h2>
      <div
        className={detail.modeSwitch}
        role="radiogroup"
        aria-label="Where the requests come from"
      >
        {SOURCES.map(([source, label]) => (
          <button
            key={source}
            type="button"
            role="radio"
            aria-checked={workload.source === source}
            onClick={() => set({ source })}
          >
            {label}
          </button>
        ))}
      </div>

      {workload.source === "capture" ? (
        <Recordings
          preset={preset}
          member={member}
          current={capture}
          onPick={(name) => set({ capture: name })}
        />
      ) : workloads?.error ? (
        <p className={s.hint} role="alert">
          The service did not answer ({workloads.error.message}). Recorded traffic
          still runs.
        </p>
      ) : !workloads ? (
        <p className={s.hint} role="status">
          Loading…
        </p>
      ) : workload.source === "generated" ? (
        <Shape limits={limits} custom={workload.custom} set={set} />
      ) : (
        <Upload
          described={workloads.sources.upload}
          upload={workload.upload}
          set={set}
        />
      )}

      {workload.source !== "capture" && moe && (
        <Field
          label="Experts route as in"
          help="Your requests carry no expert routing, so their tokens reach the experts as in this recording."
        >
          {(id, helpId) => (
            <select
              id={id}
              aria-describedby={helpId}
              value={capture.name}
              onChange={(event) => set({ capture: event.target.value })}
            >
              {preset.captures.map((c) => (
                <option key={c.name} value={c.name}>
                  {c.name}
                </option>
              ))}
            </select>
          )}
        </Field>
      )}

      {draft != null &&
        (ownAcceptance ? (
          <p className={s.hint}>
            {workload.source === "capture"
              ? "Each request drafts with the acceptance it had in this recording."
              : "Draft acceptance comes from your file's accept_rate column."}
          </p>
        ) : (
          <Field
            label="Draft acceptance"
            help={`${NO_ACCEPTANCE[workload.source]}, so give the chance a drafted token is accepted: one number for every position, or ${draft} numbers, one per position.`}
            problem={acceptanceProblem(workload.acceptRate, draft)}
          >
            {(id, helpId) => (
              <input
                id={id}
                aria-describedby={helpId}
                inputMode="decimal"
                placeholder="0.7"
                value={workload.acceptRate}
                onChange={(event) => set({ acceptRate: event.target.value })}
              />
            )}
          </Field>
        ))}
    </section>
  );
}

// Why the traffic needs an acceptance typed in, by where it comes from.
const NO_ACCEPTANCE = {
  capture: "This recording kept no draft acceptance",
  generated: "Requests of your shape carry no draft acceptance",
  upload: "Your file has no accept_rate column",
};

/* The acceptance a reader typed as numbers, or why it cannot be sent. */
export function parseAcceptance(text, draft) {
  const parts = text
    .split(/[\s,]+/)
    .filter(Boolean)
    .map(Number);
  if (!parts.length) return { problem: "Give a draft acceptance under Traffic." };
  if (parts.some((value) => !(value >= 0 && value <= 1)))
    return { problem: "Each chance is a number from 0 to 1." };
  if (parts.length !== 1 && parts.length !== draft)
    return { problem: `Give one number, or ${draft}.` };
  return { value: parts.length === 1 ? parts[0] : parts };
}

const acceptanceProblem = (text, draft) =>
  text.trim() === "" ? null : parseAcceptance(text, draft).problem;

/* The recordings the deployment can replay, each with what it holds. */
function Recordings({ preset, member, current, onPick }) {
  const moe = routes(preset);
  return (
    <>
      <div className={s.options} role="radiogroup" aria-label="Recording">
        {preset.captures.map((capture) => {
          const reason = blockedText(member, capture);
          return (
            <button
              key={capture.name}
              type="button"
              role="radio"
              aria-checked={capture === current}
              className={s.option}
              onClick={() => onPick(capture.name)}
            >
              <span className={s.optionName}>
                {capture.name}
                {reason && <CircleDashed size={13} aria-hidden="true" />}
              </span>
              <span className={s.optionFacts}>
                {reason ?? (capture.facts ? factsText(capture.facts) : "")}
              </span>
            </button>
          );
        })}
      </div>
      <p className={s.hint}>
        {moe
          ? "Requests recorded from a real server, with the experts each token reached."
          : "Requests recorded from a real server."}
      </p>
    </>
  );
}

/* Requests of one shape: how many, and how long each prompt and output is. */
function Shape({ limits, custom, set }) {
  const change = (name, text) => set({ custom: { ...custom, [name]: text } });
  const fields = [
    ["requests", "Requests", { whole: true, max: limits.max_requests }],
    ["prompt", "Prompt tokens", { whole: true }],
    ["output", "Output tokens", { whole: true }],
  ];
  return (
    <>
      <div className={s.fields}>
        {fields.map(([name, label, rule]) => (
          <Field
            key={name}
            label={label}
            help={
              name === "requests"
                ? `Up to ${count(limits.max_requests)}.`
                : undefined
            }
            problem={numberProblem(custom[name], rule)}
          >
            {(id, helpId) => (
              <input
                id={id}
                aria-describedby={helpId}
                inputMode="numeric"
                value={custom[name]}
                onChange={(event) => change(name, event.target.value)}
              />
            )}
          </Field>
        ))}
      </div>
      <p className={s.hint}>
        Every request has these lengths and shares no prefix.
      </p>
    </>
  );
}

/* A CSV of the reader's own requests. The service reads its columns as the
   format they fit and keeps it a day. */
function Upload({ described, upload, set }) {
  const [state, setState] = useState({});
  const fileId = useId();
  const [plain, ...others] = described.formats;

  const send = async (file) => {
    if (!file) return;
    if (file.size > described.max_bytes) {
      setState({
        failure: `${file.name} is ${count(file.size / 1024)} KB; the limit is ${count(described.max_bytes / 1024)} KB.`,
      });
      return;
    }
    setState({ busy: file.name });
    set({ upload: null });
    try {
      const record = await uploadWorkload(file);
      set({ upload: { ...record, file: file.name } });
      setState({});
    } catch (error) {
      setState({ failure: error.reason ?? error.message });
    }
  };

  return (
    <>
      <div className={s.uploadRow}>
        <label className={models.addButton} htmlFor={fileId}>
          {state.busy ? (
            <LoaderCircle size={16} className={models.spin} aria-hidden="true" />
          ) : (
            <FileUp size={16} aria-hidden="true" />
          )}
          {state.busy
            ? `Reading ${state.busy}…`
            : upload
              ? "Choose another CSV"
              : "Choose a CSV"}
        </label>
        <input
          id={fileId}
          className={s.fileInput}
          type="file"
          accept=".csv,text/csv"
          onChange={(event) => {
            send(event.target.files?.[0]);
            event.target.value = "";
          }}
        />
        {upload && (
          <p className={s.uploaded} role="status">
            <code>{upload.file}</code>: {factsText(upload)}
          </p>
        )}
      </div>
      {state.failure && (
        <p className={s.failure} role="alert">
          {state.failure}
        </p>
      )}
      {plain && (
        <p className={s.hint}>
          One request per row, with the columns <Columns names={plain.columns} /> (
          <code>arrival_time</code> in milliseconds). Up to{" "}
          {count(described.max_bytes / 1024)} KB.
        </p>
      )}
      {others.length > 0 && (
        <details className={s.more}>
          <summary>Other formats</summary>
          {others.map((format) => (
            <p key={format.name} className={s.hint}>
              Conversations, one round per row: <Columns names={format.columns} />.
            </p>
          ))}
          <p className={s.hint}>
            Optional columns:{" "}
            {described.tags.map((tag, index) => (
              <span key={tag.name}>
                {index > 0 && "; "}
                <Columns names={tag.columns} />
              </span>
            ))}
            .
          </p>
        </details>
      )}
    </>
  );
}

function Columns({ names }) {
  return names.map((name, index) => (
    <span key={name}>
      {index > 0 && ", "}
      <code>{name}</code>
    </span>
  ));
}

/* ---------- Load ---------- */

/* How hard the traffic comes: one number. A rate keeps the traffic's own
   spacing, scaled to that average; a concurrency has every request ready at
   once and at most that many running. */
function Load({ facts, workload, set }) {
  const canRate = facts?.rate != null;
  const mode = canRate ? workload.load : "concurrency";
  const per = facts?.sessions ? "conversations/s" : "requests/s";
  return (
    <section
      className={`${detail.controls} ${s.group}`}
      aria-labelledby="load-title"
    >
      <h2 id="load-title">Load</h2>
      <div className={detail.modeSwitch} role="radiogroup" aria-label="Load">
        {[
          ["rate", "Request rate"],
          ["concurrency", "Concurrency"],
        ].map(([value, label]) => (
          <button
            key={value}
            type="button"
            role="radio"
            aria-checked={mode === value}
            disabled={value === "rate" && !canRate}
            onClick={() => set({ load: value })}
          >
            {label}
          </button>
        ))}
      </div>
      {!facts ? (
        <p className={s.hint}>Pick the traffic first.</p>
      ) : mode === "rate" ? (
        <Field
          label={`Arrivals (${per})`}
          help={`Arrivals keep the traffic's own spacing, scaled to this average. Its own: ${rate(facts.rate)} ${per}.`}
          problem={workload.rate == null ? null : numberProblem(workload.rate)}
        >
          {(id, helpId) => (
            <input
              id={id}
              aria-describedby={helpId}
              inputMode="decimal"
              value={workload.rate ?? typed(facts.rate)}
              onChange={(event) => set({ rate: event.target.value })}
            />
          )}
        </Field>
      ) : (
        <Field
          label={facts.sessions ? "Conversations in flight" : "Requests in flight"}
          help={
            canRate
              ? "Every request is ready at the start; at most this many run at once."
              : "These requests all arrive at once, so only a concurrency applies: at most this many run at once."
          }
          problem={
            workload.concurrency == null
              ? null
              : numberProblem(workload.concurrency, { whole: true })
          }
        >
          {(id, helpId) => (
            <input
              id={id}
              aria-describedby={helpId}
              inputMode="numeric"
              value={workload.concurrency ?? String(facts.requests)}
              onChange={(event) => set({ concurrency: event.target.value })}
            />
          )}
        </Field>
      )}
    </section>
  );
}

// A number as a reader would type it: three significant figures, no grouping.
const typed = (value) => String(Number(value.toPrecision(3)));

/* A label above its control, a helper under it, and what is wrong with the
   value as soon as it is typed. */
function Field({ label, help, problem, children }) {
  const id = useId();
  const helpId = `${id}-help`;
  return (
    <div className={s.field}>
      <label htmlFor={id}>{label}</label>
      {children(id, help || problem ? helpId : undefined)}
      {problem ? (
        <small id={helpId} className={s.problem}>
          {problem}
        </small>
      ) : (
        help && <small id={helpId}>{help}</small>
      )}
    </div>
  );
}
