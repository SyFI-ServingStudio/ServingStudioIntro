import { ArrowRight, ChevronDown, UserRound } from "lucide-react";
import evidence from "../data/glmKernelBandwidth.json";
import logo from "../servingstudio-symbol.svg";
import s from "./UseCases.module.css";

const gbps = (value) => value.toLocaleString("en-US", { maximumFractionDigits: 1 });

export function KernelBandwidthFollowUp() {
  return (
    <div className={s.bandwidthFollowUp}>
      <div className={s.chatUserTurn}>
        <div className={s.userTurnContent}>
          <h3>You</h3>
          <div className={s.userMessage}>
            <p>How far is each kernel from optimal?</p>
          </div>
        </div>
        <div className={s.userAvatar} aria-hidden="true">
          <UserRound size={23} strokeWidth={1.7} />
        </div>
      </div>
      <article className={s.agentTurn}>
        <div className={s.agentAvatar} aria-hidden="true">
          <img src={logo} alt="" />
        </div>
        <div className={s.agentTurnContent}>
          <h3>ServingStudio Agent</h3>
          <div className={s.agentMessage}>
            <div className={s.agentAnswer}>
              <p>
                For these five kernels, I can compare effective bandwidth with the
                hardware ceiling. Each B200 provides 8,000 GB/s of HBM bandwidth;
                the all-reduce is compared with the 900 GB/s one-way NVLink limit.
              </p>
              <figure
                className={`${s.exampleResult} ${s.bandwidthTableWrap}`}
                tabIndex={0}
                role="region"
                aria-label="Kernel bandwidth comparison"
              >
                <table className={s.bandwidthTable}>
                  <caption>Effective bandwidth versus hardware peak</caption>
                  <colgroup>
                    <col className={s.bandwidthKernelColumn} />
                    <col span={3} className={s.bandwidthMetricColumn} />
                  </colgroup>
                  <thead>
                    <tr>
                      <th scope="col">Kernel</th>
                      <th scope="col">Effective GB/s</th>
                      <th scope="col">Peak GB/s</th>
                      <th scope="col">Of peak</th>
                    </tr>
                  </thead>
                  <tbody>
                    {evidence.rows.map((row) => {
                      const rate = row.logicalBytes / row.timeMs / 1e6;
                      const peak =
                        row.limit === "HBM"
                          ? evidence.hbmGbps
                          : evidence.nvlinkOneWayGbps;
                      const percentage = (100 * rate) / peak;
                      return (
                        <tr key={row.name}>
                          <th scope="row">
                            {row.name}
                            <small>
                              {row.limit === "HBM" ? "HBM" : "NVLink bus bandwidth"}
                            </small>
                          </th>
                          <td data-label="Rate GB/s">{gbps(rate)}</td>
                          <td data-label="Peak GB/s">{gbps(peak)}</td>
                          <td data-label="Of peak">
                            <span>{percentage.toFixed(1)}%</span>
                            <div className={s.bandwidthTrack} aria-hidden="true">
                              <i style={{ width: `${percentage}%` }} />
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </figure>
              <p>
                Fused MoE reaches about 77% of the HBM ceiling on the slowest rank.
                Attention and the projections reach about 34–48%, showing a larger
                gap from the bandwidth ceiling. The small all-reduce reaches only
                about 6% of NVLink peak, but at this message size, launch and
                synchronization latency can matter more than link bandwidth.
              </p>
              <p className={s.agentConclusion}>
                These percentages are not an optimality score or a promised speedup.
                Compute throughput, routing balance, cache reuse and launch overhead
                also constrain each kernel.
              </p>
              <details className={s.inlineEvidence}>
                <summary>
                  Bandwidth calculation <ChevronDown size={17} />
                </summary>
                <p className={s.specMethod}>
                  Effective bandwidth is logical bytes divided by the profiled time
                  used in this prediction, not measured HBM transactions. MoE bytes
                  include active-expert weights, scales, routed inputs and finalized
                  outputs, using the corrected byte calculation. The comparison uses
                  one call per rank rather than the cross-layer totals above.
                  All-reduce uses bus-normalized traffic.
                </p>
                <a
                  className="text-link"
                  download="glm52-kernel-bandwidth.json"
                  href={`data:application/json;charset=utf-8,${encodeURIComponent(JSON.stringify(evidence, null, 2))}`}
                >
                  Download bandwidth data <ArrowRight size={17} />
                </a>
              </details>
            </div>
          </div>
        </div>
      </article>
    </div>
  );
}
