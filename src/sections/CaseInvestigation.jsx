import {
  UserRound,
  Activity,
  ScanSearch,
  Code2,
  BadgeCheck,
  ArrowUpRight,
} from "lucide-react";
import logo from "../servingstudio-symbol.svg";
import s from "./CaseInvestigation.module.css";
import { qwenRounds } from "./qwenInvestigation";

const sglangRounds = [
  {
    request: "Align ServingStudio Sim with SGLang for GLM-5.2 on four B200 GPUs.",
    steps: [
      {
        icon: Activity,
        title: "Compare kernel timings",
        body: "I profiled SGLang and compared its kernel timings with ServingStudio Sim's predictions for the same inputs. Decode timings were close, but the MoE kernels were slower during prefill. I focused on those kernels to find the source of the difference.",
      },
      {
        icon: ScanSearch,
        title: "The same inputs select different kernels",
        body: "SGLang selected a different MoE kernel configuration from our standalone benchmark. Disabling autotuning in the benchmark reproduced SGLang's slower timings. Because SGLang had autotuning enabled, I traced why its prefill path was not using those results.",
      },
    ],
    finding:
      "The slower prefill timings appear to come from the choice of MoE kernel. Autotuning is enabled in SGLang, but this execution path seems to be missing its results.",
  },
  {
    request:
      "Determine whether this is an SGLang issue. If so, fix it and profile the result.",
    steps: [
      {
        icon: ScanSearch,
        title: "Startup tuning misses the CUDA graph path",
        body: "Startup autotuning and prefill CUDA graphs do not call the MoE kernel in the same way. During graph execution, SGLang defers the final output step, which changes the output tensor shape. FlashInfer includes that shape in its tuning key, so the results collected at startup do not apply to this call.",
      },
      {
        icon: Code2,
        title: "Tune the path that prefill actually uses",
        body: "I added a tuning pass inside the graph capture context, before recording the graphs. Autotuning could then observe the same MoE call used during prefill. I profiled the modified implementation and ran a serving benchmark to measure the effect.",
      },
      {
        icon: BadgeCheck,
        title: "Measure the serving improvement",
        body: "I tested 240 requests, each with 4,096 input tokens and 8 output tokens, at concurrency 24. Across three warmed runs, median completion time fell from 44.873 s to 42.503 s, yielding 5.6% higher input throughput. Both versions also answered 60 of 64 questions correctly in a small GSM8K check.",
        showOutcome: true,
      },
    ],
    finding:
      "The prefill graph path was not covered by startup autotuning. Tuning that path improved serving performance in this benchmark. The PR includes the fix and instructions for reproducing the result.",
  },
];

const specRounds = [
  {
    request:
      "Align ServingStudio Sim with vLLM for GLM-5.2 with five MTP draft tokens on four B200 GPUs.",
    steps: [
      {
        icon: Activity,
        title: "Align the kernel timings first",
        body: "I first checked whether ServingStudio Sim could predict vLLM's kernel timings for the same inputs. The comparison exposed two modeling errors: the draft model processed a different batch than we assumed, and its MoE kernels used BF16 rather than NVFP4. After I corrected both, the predictions followed the measurements closely across most of the run, although some decode differences remained.",
        figure: {
          file: "spec5-kernel-alignment.png",
          width: 1600,
          height: 720,
          alt: "Measured kernel-path timings and ServingStudio Sim predictions across all 943 captured iterations, showing recurring workload changes and remaining timing differences.",
          caption:
            "Kernel-path comparison after the draft-model corrections, covering all 943 captured iterations (0–942). The wider view also shows the remaining decode-stage differences. These timings exclude time spent waiting for ranks to reach collectives, so they do not represent total GPU busy time.",
        },
      },
      {
        icon: Activity,
        title: "Total GPU time still does not align",
        body: "Kernel-level agreement improved, but it did not explain the full iteration. Measured GPU busy time remained longer than predicted. Over a wider interval, the prediction rose and fell with the workload while the measured timeline stayed much flatter. I checked the timing definitions and the original trace before attributing the difference to the model.",
        figure: {
          file: "spec5-sawtooth.png",
          width: 1971,
          height: 1280,
          alt: "The ServingStudio Sim prediction repeatedly rises and falls while the measured vLLM GPU timeline remains flatter.",
          caption:
            "The wider comparison that prompted a closer look at the trace. The lower panel expands the interval from 12 to 95 seconds.",
        },
      },
      {
        icon: ScanSearch,
        title: "The GPUs reach collectives at different times",
        body: "The trace shows that the GPUs do not reach corresponding collective operations at the same time. In iteration 200, their first kernels start within 0.772 ms of one another, but waiting at collectives totals 26.001 ms. The delay therefore accumulates during the iteration. I would inspect this part of the trace next.",
      },
    ],
    finding:
      "The kernel timings alone do not explain the extra GPU time. The GPUs spend additional time waiting for one another at communication operations; the next question is what makes them arrive at different times.",
  },
  {
    request:
      "The trace shows many individual kernel launches, which suggests CPU launch overhead. Is this path running eagerly while the non-speculative path uses CUDA graphs? Find and fix the cause.",
    steps: [
      {
        icon: Activity,
        title: "Yes, this path is running eagerly",
        body: "I checked the CUDA API calls. In iterations 376–400, the speculative path makes 835 individual kernel launches per GPU per forward pass and no graph launches. The equivalent non-speculative path uses piecewise CUDA graphs. This confirms that eager execution contributes CPU launch overhead.",
      },
      {
        icon: ScanSearch,
        title: "The 2,048-token batch falls outside the captured graph sizes",
        body: "With five MTP draft tokens, vLLM processes six positions per decode request. Its graph-size filter therefore retains sizes divisible by six. Under the 2,048-token limit, the largest retained graph is only 2,034 tokens, so common 2,048-token batches fall back to eager execution.",
      },
      {
        icon: Code2,
        title: "Adjust the boundary and profile again",
        body: "I raised both the scheduler budget and the graph limit to 2,052, which is divisible by six. Raising only the graph limit did not work because vLLM capped it at the scheduler budget. In the new profile, the same diagnostic window contains 79 graph launches and 261 individual kernel launches per GPU per forward pass, confirming that graph replay is restored.",
      },
      {
        icon: BadgeCheck,
        title: "Check whether the change improves performance",
        body: "Average iteration time in the diagnostic window fell from 141.048 ms to 114.903 ms. I then extended the analysis beyond that window and ran a separate benchmark without profiling. Both benchmark runs completed all 100 requests, and the 2,052-token configuration achieved 10.77% higher output throughput. Speculative acceptance also changed between the runs, so this throughput result is supporting evidence rather than a controlled measurement of graph replay alone.",
        showOutcome: true,
      },
    ],
    finding:
      "The investigation led from a timing discrepancy to a graph-size boundary in vLLM. Adjusting that boundary restored graph replay and reduced the observed delay. Some decode kernel and scheduling differences remained, so this did not establish complete simulator alignment.",
  },
];

export function CaseInvestigation({ caseId, outcome, charts = {} }) {
  const rounds =
    caseId === "qwen-fusion"
      ? qwenRounds
      : caseId === "spec5-graphs"
        ? specRounds
        : sglangRounds;
  return (
    <section className={s.investigation} aria-labelledby="investigation-title">
      <header className={s.heading}>
        <h3 id="investigation-title">How the investigation unfolded</h3>
        <p>A condensed record of our team's work with the Agent.</p>
      </header>
      <ol className={s.rounds}>
        {rounds.map((round) => (
          <li className={s.round} key={round.request}>
            <div className={s.human}>
              <div className={s.role}>
                <UserRound size={20} aria-hidden="true" /> Our team
              </div>
              <p>{round.request}</p>
            </div>
            <div className={s.agent}>
              <div className={s.role}>
                <img src={logo} alt="" width="24" height="24" /> Agent
              </div>
              <ol className={s.steps}>
                {round.steps.map(
                  ({ icon: Icon, title, body, figure, chart, showOutcome }) => (
                    <li className={s.step} key={title}>
                      <Icon size={20} aria-hidden="true" />
                      <div>
                        <h4>{title}</h4>
                        <p>{body}</p>
                        {chart && charts[chart]}
                        {figure && (
                          <figure className={s.figure}>
                            <a
                              href={`${import.meta.env.BASE_URL}case-studies/${figure.file}`}
                              target="_blank"
                              rel="noreferrer"
                              aria-label={`Open full-size graph: ${title}`}
                            >
                              <img
                                src={`${import.meta.env.BASE_URL}case-studies/${figure.file}`}
                                width={figure.width}
                                height={figure.height}
                                alt={figure.alt}
                              />
                            </a>
                            <figcaption>{figure.caption}</figcaption>
                            <a
                              className={s.figureLink}
                              href={`${import.meta.env.BASE_URL}case-studies/${figure.file}`}
                              target="_blank"
                              rel="noreferrer"
                            >
                              Open full-size graph{" "}
                              <ArrowUpRight size={16} aria-hidden="true" />
                            </a>
                          </figure>
                        )}
                        {showOutcome && outcome}
                      </div>
                    </li>
                  ),
                )}
              </ol>
              <div className={s.finding}>
                <strong>Conclusion</strong>
                <p>{round.finding}</p>
              </div>
            </div>
          </li>
        ))}
      </ol>
    </section>
  );
}
