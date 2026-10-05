import { useEffect, useMemo, useState, useSyncExternalStore } from "react";
import { readQuery, setQuery, subscribeUrl } from "../../url";
import { archNames, loadModels, memberFromQuery } from "../models/modelData";
import { RunList } from "./RunList";
import { RunPanel } from "./RunPanel";
import {
  blockedText,
  captureOf,
  carriesAcceptance,
  customGenerator,
  initialWorkload,
  numberProblem,
  parseAcceptance,
  replays,
  routes,
  Setup,
  trafficFacts,
} from "./Setup";
import { useRuns } from "./runs";
import {
  draftTokens,
  loadSimPresets,
  loadWorkloads,
  presetName,
  startSimulation,
} from "./simulateData";
import detail from "../kernels/KernelDetail.module.css";
import models from "../models/Models.module.css";
import s from "./Simulate.module.css";

const search = () => window.location.search;

/* The Simulate page: pick a deployment (`?preset=<id>` and a value per axis),
   the traffic it serves and how hard it comes, run it on the service, and
   compare the runs this browser started (`?run=<id>` shows one). */
export default function Simulate() {
  const query = useSyncExternalStore(subscribeUrl, search);
  const [catalog, setCatalog] = useState(null);
  const [error, setError] = useState(null);
  const [workloads, setWorkloads] = useState(null);
  useEffect(() => {
    Promise.all([loadSimPresets(), loadModels().catch(() => null)]).then(
      ([sims, models]) => setCatalog(join(sims, models)),
      setError,
    );
    loadWorkloads().then(setWorkloads, (failure) =>
      setWorkloads({ error: failure }),
    );
  }, []);

  if (error)
    return (
      <p className="wrap blog-loading" role="alert">
        The deployments did not load ({error.reason ?? error.message}). The
        simulation service may be down; reload the page to try again.
      </p>
    );
  if (!catalog)
    return (
      <p className="wrap blog-loading" role="status">
        Loading deployments…
      </p>
    );
  return <Workbench catalog={catalog} workloads={workloads} />;
}

/* Sim presets by checkpoint, in /models order, each named as /models names
   its checkpoint and the arch presets its pools run. */
function join(sims, models) {
  const checkpoints = models?.checkpoints ?? [];
  const names = archNames(models);
  const order = (checkpoint) => {
    const index = checkpoints.findIndex((c) => c.checkpoint === checkpoint);
    return index < 0 ? checkpoints.length : index;
  };
  const groups = new Map();
  for (const preset of [...sims.presets].sort(
    (a, b) => order(a.checkpoint) - order(b.checkpoint),
  )) {
    const entry = checkpoints.find((c) => c.checkpoint === preset.checkpoint);
    if (!groups.has(preset.checkpoint))
      groups.set(preset.checkpoint, {
        checkpoint: preset.checkpoint,
        name: entry?.name ?? preset.checkpoint,
        presets: [],
      });
    const group = groups.get(preset.checkpoint);
    group.presets.push({
      ...preset,
      name: presetName(preset, names),
      checkpointName: group.name,
    });
  }
  const list = [...groups.values()];
  return {
    limits: sims.limits,
    checkpoints: list,
    presets: new Map(list.flatMap((g) => g.presets.map((p) => [p.id, p]))),
  };
}

/* A member that builds and can use one of the preset's captures: measured on
   it, its requests replayed or (a misfit) its routing alone. */
const runnable = (preset) => (member) =>
  !member.error &&
  preset.captures.some((c) => {
    const reason = member.unavailable[c.name];
    return !reason || reason.misfit;
  });

function Workbench({ catalog, workloads }) {
  const url = readQuery();
  const preset =
    catalog.presets.get(url.preset) ?? catalog.checkpoints[0].presets[0];
  const checkpoint = catalog.checkpoints.find((c) => c.presets.includes(preset));
  const { member, unmatched } = memberFromQuery(preset, url, runnable(preset));
  const [workload, setWorkload] = useWorkload();
  const runs = useRuns(url.run);
  const selected = url.run ?? runs.ids[0] ?? null;
  const [start, setStart] = useState({});

  const pick = (id, params) => {
    const next = catalog.presets.get(id);
    const fitted = params
      ? memberFromQuery(next, params, runnable(next)).member.params
      : {};
    setQuery({ preset: id, ...fitted, run: url.run });
  };
  const select = (id) => setQuery({ ...url, run: id });
  const remove = (id) => {
    runs.remove(id);
    if (id === url.run) setQuery({ ...url, run: undefined });
  };

  const request = useMemo(
    () => requestFor(preset, member, workload, workloads, catalog.limits),
    [preset, member, workload, workloads, catalog.limits],
  );

  const run = async () => {
    setStart({ busy: true });
    try {
      const answer = await startSimulation(request.body);
      runs.add(answer.simulation_id, null);
      setStart({});
      setQuery({ ...url, run: answer.simulation_id });
    } catch (failure) {
      setStart({
        failure:
          failure.status === 429 && failure.retryAfter
            ? `${failure.reason}. Try again in ${failure.retryAfter} s.`
            : (failure.reason ?? failure.message),
      });
    }
  };

  return (
    <div className={detail.page}>
      <header className={`wrap ${detail.header}`}>
        <h1>Simulate</h1>
        <p className={detail.summary}>
          Send traffic to a deployment and read the latency and throughput its
          clients would see.
        </p>
      </header>
      <div className={`wrap ${s.body}`}>
        {unmatched && (
          <p className={models.notice} role="status">
            The link named values no configuration of this deployment has, so the
            closest one is shown.
          </p>
        )}
        <div className={s.layout}>
          <Setup
            catalog={catalog}
            checkpoint={checkpoint}
            preset={preset}
            member={member}
            onPick={pick}
            workloads={workloads}
            workload={workload}
            setWorkload={(change) => {
              setStart({});
              setWorkload(change);
            }}
          />
          <aside className={s.aside}>
            <RunPanel
              request={request}
              starting={start.busy}
              startError={start.failure}
              onRun={run}
              id={selected}
              record={selected ? runs.records[selected] : null}
              presets={catalog.presets}
              onStop={remove}
            />
          </aside>
        </div>
      </div>
      <RunList
        ids={runs.ids}
        records={runs.records}
        selected={selected}
        presets={catalog.presets}
        onSelect={select}
        onRemove={remove}
      />
    </div>
  );
}

/* The workload outlives a change of deployment: a reader who built a trace
   keeps it while comparing deployments. Kept for the visit only. */
let kept = initialWorkload;
function useWorkload() {
  const [workload, setState] = useState(kept);
  const set = (change) =>
    setState((current) => {
      kept = typeof change === "function" ? change(current) : change;
      return kept;
    });
  return [workload, set];
}

/* POST /simulate's body for the setup, or why there is none yet. Each field
   says what is wrong with its own value as it is typed; this only repeats it
   beside the run button. */
function requestFor(preset, member, workload, workloads, limits) {
  const capture = captureOf(preset, workload, member);
  const reason = blockedText(member, capture, replays(workload));
  if (reason)
    return { blocked: `${reason}. Pick another configuration or recording.` };
  const body = { source: workload.source };
  if (workload.source === "capture" || routes(preset)) body.capture = capture?.name;
  if (workload.source === "generated") {
    const c = workload.custom;
    const problem =
      numberProblem(c.requests, { whole: true, max: limits.max_requests }) ??
      numberProblem(c.prompt, { whole: true }) ??
      numberProblem(c.output, { whole: true });
    if (problem) return { blocked: `Requests of your shape: ${problem}` };
    body.generator = customGenerator(c);
  }
  if (workload.source === "upload") {
    if (!workload.upload) return { blocked: "Choose a CSV first." };
    body.upload = workload.upload.workload_id;
  }

  const facts = trafficFacts(preset, member, workload, workloads);
  if (!facts) return { blocked: "Loading what the traffic holds…" };
  if (workload.load === "rate" && facts.rate != null) {
    if (workload.rate != null) {
      const problem = numberProblem(workload.rate);
      if (problem) return { blocked: `Request rate: ${problem}` };
      body.load = { rate: Number(workload.rate) };
    }
  } else {
    const text = workload.concurrency ?? String(facts.requests);
    const problem = numberProblem(text, { whole: true });
    if (problem) return { blocked: `Concurrency: ${problem}` };
    body.load = { concurrency: Number(text) };
  }

  const draft = draftTokens(member);
  if (draft != null && !carriesAcceptance(facts)) {
    const acceptance = parseAcceptance(workload.acceptRate, draft);
    if (acceptance.problem) return { blocked: acceptance.problem };
    body.accept_rate = acceptance.value;
  }
  return { body: { preset: preset.id, params: member.params, workload: body } };
}
