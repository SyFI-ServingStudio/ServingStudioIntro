import { useEffect, useState } from "react";
import {
  ArrowRight,
  Check,
  ChevronDown,
  SquareTerminal,
  UserRound,
} from "lucide-react";
import logo from "../servingstudio-symbol.svg";
import s from "./UseCases.module.css";
import { useStudyReplay } from "../hooks/useStudyReplay";
import specDepth from "../data/specDepth.json";
import study from "../data/glmSpec3Study.json";

const bestDepth = specDepth.rows.reduce((best, row) =>
  row.throughput > best.throughput ? row : best,
);
const baseline = specDepth.rows.find((row) => row.depth === 0);
const depthThree = specDepth.rows.find((row) => row.depth === 3);
const depthFour = specDepth.rows.find((row) => row.depth === 4);
const acceptedCounts = Object.values(
  depthThree.acceptance.accepted_tokens_per_position,
);
const cumulativeAcceptance = acceptedCounts.map(
  (count) => count / depthThree.acceptance.num_drafts,
);
const conditionalAcceptance = acceptedCounts.map((count, index) =>
  index === 0
    ? count / depthThree.acceptance.num_drafts
    : count / acceptedCounts[index - 1],
);
const formatNumber = (value, digits = 1) =>
  value.toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });

function DownloadLink({ filename, data }) {
  return (
    <a
      className="text-link"
      download={filename}
      href={`data:application/json;charset=utf-8,${encodeURIComponent(JSON.stringify(data, null, 2))}`}
    >
      Download experiment data <ArrowRight size={17} />
    </a>
  );
}

function AgentTurn({ children, messageClassName = "", ...props }) {
  return (
    <article className={s.agentTurn} {...props}>
      <div className={s.agentAvatar}>
        <img src={logo} alt="" />
      </div>
      <div className={s.agentTurnContent}>
        <h3>ServingStudio Agent</h3>
        <div className={`${s.agentMessage} ${messageClassName}`}>{children}</div>
      </div>
    </article>
  );
}

function ContinueStudy({ children, controls, onContinue }) {
  return (
    <div className={s.roundGate}>
      <button
        type="button"
        className={s.continueStudy}
        aria-controls={controls}
        onClick={onContinue}
      >
        <span>
          <small>Continue the study</small>
          <strong>{children}</strong>
        </span>
        <i aria-hidden="true">
          <ArrowRight size={22} />
        </i>
      </button>
    </div>
  );
}

function ReplayText({ text, chars = text.length }) {
  return (
    <>
      <span>{text.slice(0, chars)}</span>
      <span style={{ visibility: "hidden" }} aria-hidden="true">
        {text.slice(chars)}
      </span>
    </>
  );
}

function reveal(visible) {
  return {
    style: { visibility: visible ? undefined : "hidden" },
    "aria-hidden": !visible,
    inert: !visible,
  };
}

function ReplayStudyRound({
  id,
  question,
  intro,
  steps = [],
  execution,
  executionComplete,
  executionDetail,
  answer = "",
  chart,
  conclusion,
  current,
  continueLabel,
  continueControls,
  onContinue,
}) {
  const replay = useStudyReplay(
    id,
    intro,
    answer,
    steps.length,
    conclusion,
    question,
  );
  const words = question.trim().split(/\s+/);

  return (
    <div
      ref={replay.ref}
      id={id}
      className={s.studyRound}
      data-replay-stage={replay.stage}
      tabIndex={-1}
    >
      <div className={s.chatUserTurn}>
        <div className={s.userTurnContent}>
          <h3>You</h3>
          <div className={s.userMessage}>
            <p aria-label={question} role="group">
              {words.map((word, index) => (
                <span
                  key={`${word}-${index}`}
                  aria-hidden="true"
                  style={{
                    visibility:
                      replay.stage === -1 && index >= replay.chars
                        ? "hidden"
                        : "visible",
                  }}
                >
                  {index > 0 ? " " : ""}
                  {word}
                </span>
              ))}
            </p>
          </div>
        </div>
        <div className={s.userAvatar} aria-hidden="true">
          <UserRound size={23} strokeWidth={1.7} />
        </div>
      </div>

      <div className={s.replyStack}>
        {replay.stage === 0 && (
          <AgentTurn messageClassName={s.thinking}>
            <span>Thinking</span>
            <span className={s.thinkingDots} aria-hidden="true">
              <i />
              <i />
              <i />
            </span>
          </AgentTurn>
        )}
        <AgentTurn {...reveal(replay.stage >= 1)}>
          <p className={s.agentIntro}>
            <ReplayText
              text={intro}
              chars={replay.stage === 1 ? replay.chars : undefined}
            />
          </p>
          {steps.length > 0 && (
            <details
              className={s.agentProgress}
              open
              {...reveal(replay.stage >= 2)}
            >
              <summary>
                <Check size={19} />
                <span>Experiment plan</span>
                <ChevronDown size={18} />
              </summary>
              <ol>
                {steps.map((step, index) => (
                  <li
                    key={step}
                    {...reveal(replay.stage >= 2 && index < replay.plans)}
                  >
                    <Check size={17} />
                    <span>{step}</span>
                  </li>
                ))}
              </ol>
            </details>
          )}
          <div className={s.agentExecution} {...reveal(replay.stage >= 3)}>
            <SquareTerminal size={22} strokeWidth={1.6} aria-hidden="true" />
            <div>
              <p className={s.agentExecutionTitle}>
                <span {...reveal(replay.stage < 4)}>{execution}</span>
                <span {...reveal(replay.stage >= 4)}>{executionComplete}</span>
              </p>
              <p className={s.agentExecutionDetail}>{executionDetail}</p>
            </div>
          </div>
          <div className={s.agentAnswer} {...reveal(replay.stage >= 4)}>
            {answer && (
              <p>
                <ReplayText
                  text={answer}
                  chars={replay.stage === 4 ? replay.chars : undefined}
                />
              </p>
            )}
            <div className={s.chartReveal} {...reveal(replay.stage >= 5)}>
              <div className={s.chartClip}>{chart}</div>
            </div>
            <p className={s.agentConclusion} {...reveal(replay.stage >= 6)}>
              <ReplayText
                text={conclusion}
                chars={replay.stage === 6 ? replay.chars : undefined}
              />
            </p>
          </div>
        </AgentTurn>
      </div>
      {replay.stage >= 7 && current && continueLabel && (
        <ContinueStudy controls={continueControls} onContinue={onContinue}>
          {continueLabel}
        </ContinueStudy>
      )}
    </div>
  );
}

function SpeculationSearch() {
  const [depth, setDepth] = useState(bestDepth.depth);
  const row =
    specDepth.rows.find((item) => item.depth === depth) ?? specDepth.rows[0];
  const maximum = Math.max(...specDepth.rows.map((item) => item.throughput));
  return (
    <div className={s.specSweep}>
      <div className={s.chartMeta}>
        <span>Single-request decode throughput · output tok/s</span>
        <span>Higher is better</span>
      </div>
      <div
        className={s.specSweepChart}
        style={{ "--depth-count": specDepth.rows.length }}
        aria-label="Single-request decode throughput by speculative depth"
      >
        {specDepth.rows.map((item) => (
          <button
            key={item.depth}
            type="button"
            aria-pressed={depth === item.depth}
            aria-label={`${
              item.depth === 0 ? "No speculation" : `${item.depth} draft tokens`
            }: ${item.throughput.toFixed(1)} output tokens per second`}
            onClick={() => setDepth(item.depth)}
          >
            <strong>{item.throughput.toFixed(1)}</strong>
            <div className={s.specColumn}>
              <i
                style={{
                  "--column-height": `${(item.throughput / maximum) * 100}%`,
                }}
              />
            </div>
            <span>{item.depth === 0 ? "Off" : item.depth}</span>
          </button>
        ))}
      </div>
      <div className={s.chartAxis}>Draft tokens per round</div>
      <dl className={s.specSweepMetrics}>
        <div>
          <dt>Decode throughput</dt>
          <dd>
            {row.throughput.toFixed(1)} <span>tok/s</span>
          </dd>
        </div>
        <div>
          <dt>Time per output token</dt>
          <dd>
            {row.tpotMs.toFixed(2)} <span>ms</span>
          </dd>
        </div>
        <div>
          <dt>Accepted draft tokens</dt>
          <dd>
            {row.acceptance
              ? (row.acceptance.mean_acceptance_length - 1).toFixed(2)
              : "0"}
            <span> / round</span>
          </dd>
        </div>
      </dl>
      <p className="caption">
        GLM-5.2 NVFP4 · 4 × B200 · TP4 + EP4 (DP1) · enwik9 · one request at a time.
      </p>
      <details className={s.inlineEvidence}>
        <summary>
          View the depth sweep and method <ChevronDown size={17} />
        </summary>
        <div className={s.operationTableWrap} tabIndex={0}>
          <table>
            <thead>
              <tr>
                <th>Draft tokens</th>
                <th>Time per output token (ms)</th>
                <th>Accepted draft tokens / round</th>
                <th>Throughput (tok/s)</th>
              </tr>
            </thead>
            <tbody>
              {specDepth.rows.map((item) => (
                <tr key={item.depth}>
                  <td>{item.depth === 0 ? "Off" : item.depth}</td>
                  <td>{item.tpotMs.toFixed(3)}</td>
                  <td>
                    {item.acceptance
                      ? (item.acceptance.mean_acceptance_length - 1).toFixed(3)
                      : "0"}
                  </td>
                  <td>{item.throughput.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className={s.specMethod}>
          Each request has 16,384 input and 2,048 output tokens. The first request
          warms up the engine; the median of the next seven determines throughput.
          Prefill and the first output token are excluded. Acceptance is measured
          independently at each depth across all eight requests, including the
          warm-up.
        </p>
        <DownloadLink filename="glm52-spec-depth-evidence.json" data={specDepth} />
      </details>
    </div>
  );
}

function AcceptanceEvidence() {
  const percentages = (rates) =>
    rates.map((rate) => `${(rate * 100).toFixed(1)}%`).join(" · ");
  return (
    <div className={s.acceptanceEvidence}>
      <div>
        <span>Cumulative reach rate</span>
        <strong>{percentages(cumulativeAcceptance)}</strong>
        <small>Accepted at positions 1, 2, and 3 ÷ all draft rounds</small>
      </div>
      <div>
        <span>Conditional acceptance</span>
        <strong>{percentages(conditionalAcceptance)}</strong>
        <small>Accepted at each position ÷ rounds that reached that position</small>
      </div>
    </div>
  );
}

function OperationBreakdown() {
  const maximum = Math.max(...study.timing.operations.map((item) => item.timeMs));
  return (
    <div className={s.operationChart}>
      {study.timing.operations.map((item) => {
        const shortName = item.name.split(".").at(-1);
        return (
          <div className={s.operation} key={item.name}>
            <div className={s.operationLabel}>
              <span title={item.name} aria-label={item.name}>
                {shortName}
              </span>
              <strong>
                {item.timeMs.toFixed(2)} <small>ms</small>
              </strong>
            </div>
            <div
              className={s.operationBar}
              style={{ "--bar-width": `${(item.timeMs / maximum) * 100}%` }}
            >
              <i />
              <span>{item.share.toFixed(1)}%</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function OptimalityBreakdown() {
  return (
    <div className={s.optimalityBreakdown}>
      <div
        className={s.optimalityBar}
        role="img"
        aria-label="GPU time divided among the hardware limit, kernel efficiency gap, communication, and rank imbalance"
      >
        {study.optimality.buckets.map((item) => (
          <i
            key={item.key}
            data-bucket={item.key}
            style={{ "--bucket-width": `${item.share * 100}%` }}
          />
        ))}
      </div>
      <dl className={s.optimalityLegend}>
        {study.optimality.buckets.map((item) => (
          <div key={item.key}>
            <dt>
              <i data-bucket={item.key} />
              {item.label}
            </dt>
            <dd>{(item.share * 100).toFixed(1)}%</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

function KernelOptimalityRow({ item }) {
  const shortName = item.name.split(".").at(-1);
  return (
    <div className={s.kernelOptimalityRow}>
      <div className={s.kernelOptimalityLabel}>
        <code title={item.name} aria-label={item.name}>
          {shortName}
        </code>
      </div>
      <div
        className={s.kernelOptimalityBar}
        role="img"
        aria-label={`${item.name}: ${(item.hardwareFloor * 100).toFixed(
          1,
        )}% hardware limit, ${(item.hardwareGap * 100).toFixed(
          1,
        )}% kernel efficiency gap, ${(item.communication * 100).toFixed(
          1,
        )}% communication`}
      >
        {item.hardwareFloor > 0 && (
          <i
            data-bucket="hardware_optimal"
            style={{ "--bucket-width": `${item.hardwareFloor * 100}%` }}
          />
        )}
        {item.hardwareGap > 0 && (
          <i
            data-bucket="hardware_gap"
            style={{ "--bucket-width": `${item.hardwareGap * 100}%` }}
          />
        )}
        {item.communication > 0 && (
          <i
            data-bucket="communication"
            style={{ "--bucket-width": `${item.communication * 100}%` }}
          />
        )}
      </div>
      <strong>{(item.hardwareFloor * 100).toFixed(1)}%</strong>
    </div>
  );
}

function KernelOptimalityChart() {
  return (
    <div className={s.kernelOptimality}>
      <div className={s.kernelOptimalityHeading}>
        <h4>Hardware-limit share by kernel call site</h4>
        <span>Hardware limit as a share of load-balanced time</span>
      </div>
      <div className={s.kernelOptimalityRows}>
        {study.optimality.kernels.map((item) => (
          <KernelOptimalityRow item={item} key={item.name} />
        ))}
      </div>
      <p className="caption">
        The value at right divides each call site’s hardware limit by its
        load-balanced time—its time with rank imbalance removed. These{" "}
        {study.optimality.kernels.length} call sites cover{" "}
        {(study.optimality.kernelCoverage * 100).toFixed(1)}% of the iteration’s
        load-balanced kernel time.
      </p>
    </div>
  );
}

function ConcurrencySweep() {
  const [selected, setSelected] = useState(study.concurrency.bestConcurrency);
  const row =
    study.concurrency.rows.find((item) => item.concurrency === selected) ??
    study.concurrency.rows[0];
  const maximum = Math.max(
    ...study.concurrency.rows.map((item) => item.outputThroughput),
  );
  return (
    <div className={s.specSweep}>
      <div className={s.chartMeta}>
        <span>Total output throughput · tok/s</span>
        <span>Per-request target: &gt;80 tok/s</span>
      </div>
      <div
        className={s.specSweepChart}
        style={{ "--depth-count": study.concurrency.rows.length }}
        aria-label="Output throughput and per-request decode speed by concurrency"
      >
        {study.concurrency.rows.map((item) => (
          <button
            key={item.concurrency}
            type="button"
            aria-pressed={selected === item.concurrency}
            data-eligible={item.perRequestDecode > 80}
            aria-label={`Concurrency ${item.concurrency}: ${formatNumber(item.outputThroughput)} total output tokens per second and ${formatNumber(item.perRequestDecode)} tokens per second per request`}
            onClick={() => setSelected(item.concurrency)}
          >
            <strong>{formatNumber(item.outputThroughput, 0)}</strong>
            <div className={s.specColumn}>
              <i
                style={{
                  "--column-height": `${(item.outputThroughput / maximum) * 100}%`,
                }}
              />
            </div>
            <span>{item.concurrency}</span>
          </button>
        ))}
      </div>
      <div className={s.chartAxis}>Concurrent requests</div>
      <dl className={s.specSweepMetrics}>
        <div>
          <dt>Total output throughput</dt>
          <dd>
            {formatNumber(row.outputThroughput)} <span>tok/s</span>
          </dd>
        </div>
        <div>
          <dt>Per-request decode speed</dt>
          <dd>
            {row.perRequestDecode.toFixed(1)} <span>tok/s</span>
          </dd>
        </div>
        <div>
          <dt>Mean time per output token</dt>
          <dd>
            {row.tpotMeanMs.toFixed(2)} <span>ms/token</span>
          </dd>
        </div>
      </dl>
    </div>
  );
}

function StudyFigure({ title, total, children }) {
  return (
    <figure className={s.exampleResult}>
      <div className={s.exampleContext}>
        <span>GLM-5.2 NVFP4</span>
        <span>4 × B200 · TP4 + EP4 (DP1)</span>
      </div>
      <div className={s.resultHeading}>
        <h3>{title}</h3>
        {total && (
          <div className={s.iterationTotal}>
            <strong>{total.value}</strong>
            <span>{total.label}</span>
          </div>
        )}
      </div>
      {children}
    </figure>
  );
}

export function UseCases() {
  const bestConcurrency = study.concurrency.rows.find(
    (row) => row.concurrency === study.concurrency.bestConcurrency,
  );
  const fasterPointBelowTarget = study.concurrency.rows
    .filter(
      (row) =>
        row.outputThroughput > bestConcurrency.outputThroughput &&
        row.perRequestDecode <= 80,
    )
    .reduce(
      (best, row) =>
        !best || row.outputThroughput > best.outputThroughput ? row : best,
      null,
    );
  const routedMoeShare = study.timing.operations
    .filter((item) => item.name.includes(".moe.routed_experts.fused_moe"))
    .slice(0, 2)
    .reduce((sum, item) => sum + item.share, 0);
  const optimalityShares = Object.fromEntries(
    study.optimality.buckets.map((item) => [item.key, item.share * 100]),
  );
  const timingConclusion = `The ${study.timing.operations.length} largest kernel call sites account for ${(study.timing.locationCoverage * 100).toFixed(1)}% of the modeled iteration. The two routed-expert MoE call sites are the largest contributors, together accounting for ${routedMoeShare.toFixed(1)}%.`;
  const optimalityConclusion = `At this shape, the model-derived hardware limit accounts for ${(
    study.optimality.hardwareOptimalShare * 100
  ).toFixed(
    1,
  )}% of GPU time. The largest opportunity is kernel efficiency: the gap between the modeled kernels and their hardware limit accounts for ${optimalityShares.hardware_gap.toFixed(1)}%, compared with ${optimalityShares.communication.toFixed(1)}% for communication and ${optimalityShares.other.toFixed(1)}% for uneven work across GPU ranks.`;
  const concurrencyConclusion = `Among the tested points, concurrency ${bestConcurrency.concurrency} is the highest-throughput choice that stays above the 80 tok/s per-request target: ${formatNumber(bestConcurrency.outputThroughput)} tok/s total at ${formatNumber(bestConcurrency.perRequestDecode)} tok/s per request.${
    fasterPointBelowTarget
      ? ` Concurrency ${fasterPointBelowTarget.concurrency} reaches ${formatNumber(fasterPointBelowTarget.outputThroughput)} tok/s, but per-request speed drops to ${formatNumber(fasterPointBelowTarget.perRequestDecode)} tok/s—below the target.`
      : ""
  }`;
  const [visibleRound, setVisibleRound] = useState(0);

  useEffect(() => {
    if (visibleRound === 0) return;
    const nextRound = document.getElementById(`study-round-${visibleRound}`);
    if (!nextRound) return;
    const reducedMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;
    nextRound.focus({ preventScroll: true });
    nextRound.scrollIntoView({
      behavior: reducedMotion ? "auto" : "smooth",
      block: "start",
    });
  }, [visibleRound]);

  const continueStudy = () => {
    setVisibleRound((current) => Math.min(current + 1, 3));
  };

  return (
    <section id="use-cases" className={`section ${s.examplesSection}`}>
      <div className="wrap">
        <div className="section-intro" data-reveal>
          <h2>Run the study with the Agent.</h2>
          <p>
            Begin with one serving question. The first answer comes from real B200
            measurements; the Agent then uses the Simulator and Analyzer to explain
            the kernel costs, compare them with the hardware limit, and choose an
            operating point.
          </p>
        </div>
        <div className={s.conversationDemo}>
          <div className={`${s.chatTranscript} ${s.mergedStudy}`}>
            <ReplayStudyRound
              id="study-round-0"
              question="For GLM-5.2 on four B200 GPUs with TP4 + EP4, which speculative depth gives the best single-request decode throughput on enwik9 input?"
              intro="I have a completed depth sweep from real vLLM serving on four B200 GPUs with TP4 + EP4 (DP1). It uses enwik9, derived from an English Wikipedia dump, and covers nine settings, from speculation off to eight draft tokens."
              steps={[
                "Replay the same eight 16K-input, 2K-output requests at each depth",
                "Measure accepted draft tokens at each depth",
                "Compare median decode throughput after one warm-up request",
              ]}
              execution="Running depth sweep…"
              executionComplete="Depth sweep complete"
              executionDetail="Nine depth settings measured on the same model, hardware, and request set."
              answer={`Depth 3 is the best tested point at ${bestDepth.throughput.toFixed(1)} tok/s, or ${(bestDepth.throughput / baseline.throughput).toFixed(2)}× the throughput without speculation. It accepts ${(bestDepth.acceptance.mean_acceptance_length - 1).toFixed(2)} draft tokens per round on average.`}
              chart={
                <>
                  <StudyFigure title="Throughput versus speculative depth">
                    <SpeculationSearch />
                  </StudyFigure>
                  <AcceptanceEvidence />
                </>
              }
              conclusion={`In the later concurrency sweep, I reuse the conditional acceptance rates measured here. More accepted drafts do not guarantee higher throughput: depth 4 accepts ${(
                depthFour.acceptance.mean_acceptance_length - 1
              ).toFixed(2)} draft tokens per round, compared with ${(
                depthThree.acceptance.mean_acceptance_length - 1
              ).toFixed(
                2,
              )} at depth 3, but is slower at ${depthFour.throughput.toFixed(
                1,
              )} versus ${depthThree.throughput.toFixed(1)} tok/s.`}
              current={visibleRound === 0}
              continueLabel="Inspect the 128K kernel-time breakdown"
              continueControls="study-round-1"
              onContinue={continueStudy}
            />

            {visibleRound >= 1 && (
              <ReplayStudyRound
                id="study-round-1"
                question="With three draft tokens and a 128K-token input, where does one decode iteration spend its time?"
                intro={
                  study.status === "complete"
                    ? "Now I’ll switch from measurement to simulation. The Simulator predicts one target-model verification pass and three draft passes at 128K context; I rank the kernel call sites by time."
                    : "Now I’ll switch from measurement to simulation and have the Simulator predict one target-model verification pass and three draft passes at 128K context."
                }
                execution="Running time prediction…"
                executionComplete={
                  study.status === "complete"
                    ? "Time prediction complete"
                    : "Running time prediction…"
                }
                executionDetail={
                  study.status === "complete"
                    ? "One speculative iteration: three drafted positions followed by the final verification step."
                    : "Measuring missing B200 kernel shapes before the Analyzer builds the breakdown."
                }
                chart={
                  study.status === "complete" ? (
                    <StudyFigure
                      title="Kernel-time breakdown at 128K context"
                      total={{
                        value: study.timing.totalMs.toFixed(2),
                        label: "ms per speculative iteration",
                      }}
                    >
                      <OperationBreakdown />
                      <p className="caption">
                        The final verification step sees 131,075 KV positions:
                        131,072 input tokens plus three drafted positions.
                      </p>
                    </StudyFigure>
                  ) : null
                }
                conclusion={study.status === "complete" ? timingConclusion : ""}
                current={visibleRound === 1}
                continueLabel={
                  study.status === "complete"
                    ? "Compare the kernels with the B200 hardware limit"
                    : undefined
                }
                continueControls="study-round-2"
                onContinue={continueStudy}
              />
            )}

            {visibleRound >= 2 && study.status === "complete" && (
              <ReplayStudyRound
                id="study-round-2"
                question="How close are these kernels to the B200 hardware limit?"
                intro="I’ll hold the iteration shape fixed and compare the modeled cost of each kernel with its B200 hardware limit."
                execution="Running optimality analysis…"
                executionComplete="Optimality analysis complete"
                executionDetail="The Analyzer split the 128K iteration’s GPU time into the hardware limit and the three sources of overhead above it."
                chart={
                  <StudyFigure
                    title="Distance from the hardware limit at this batch shape"
                    total={{
                      value: `${(
                        study.optimality.hardwareOptimalShare * 100
                      ).toFixed(1)}%`,
                      label: "hardware limit",
                    }}
                  >
                    <OptimalityBreakdown />
                    <KernelOptimalityChart />
                  </StudyFigure>
                }
                conclusion={optimalityConclusion}
                current={visibleRound === 2}
                continueLabel="Find the best concurrency above 80 tok/s"
                continueControls="study-round-3"
                onContinue={continueStudy}
              />
            )}

            {visibleRound >= 3 && study.status === "complete" && (
              <ReplayStudyRound
                id="study-round-3"
                question="With three draft tokens, 32K input tokens, and 16K output tokens, which concurrency maximizes total throughput while keeping each request above 80 tok/s?"
                intro="I’ll sweep concurrency with the request shape and acceptance profile fixed, discard points at or below 80 tok/s per request, and choose the remaining point with the highest total output throughput."
                execution="Running concurrency sweep…"
                executionComplete="Concurrency sweep complete"
                executionDetail={`The Simulator ran ${study.concurrency.rows.length} operating points, each with 96 fixed-length requests.`}
                chart={
                  <StudyFigure title="Throughput with an 80 tok/s per-request target">
                    <ConcurrencySweep />
                    <details className={s.inlineEvidence}>
                      <summary>
                        View every tested point <ChevronDown size={17} />
                      </summary>
                      <div className={s.operationTableWrap} tabIndex={0}>
                        <table>
                          <thead>
                            <tr>
                              <th>Concurrency</th>
                              <th>Total output throughput (tok/s)</th>
                              <th>Per-request decode speed (tok/s)</th>
                              <th>Mean TPOT (ms/token)</th>
                            </tr>
                          </thead>
                          <tbody>
                            {study.concurrency.rows.map((item) => (
                              <tr key={item.concurrency}>
                                <td>{item.concurrency}</td>
                                <td>{formatNumber(item.outputThroughput, 2)}</td>
                                <td>{item.perRequestDecode.toFixed(2)}</td>
                                <td>{item.tpotMeanMs.toFixed(3)}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                      <p className={s.specMethod}>{study.concurrency.method}</p>
                      <DownloadLink
                        filename="glm52-spec3-study.json"
                        data={study}
                      />
                    </details>
                  </StudyFigure>
                }
                conclusion={concurrencyConclusion}
                current={visibleRound === 3}
              />
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
