import {
  Activity,
  FlaskConical,
  Code2,
  GitCompareArrows,
  BadgeCheck,
  ChartNoAxesCombined,
} from "lucide-react";
import s from "./Workflow.module.css";

const stages = [
  {
    title: "Set the target and baseline",
    icon: Activity,
    description:
      "Define the model, request mix, hardware, and performance target, then measure the current deployment to establish a baseline.",
  },
  {
    title: "Explore in simulation",
    icon: FlaskConical,
    description:
      "Sweep serving configurations and inspect predicted kernel costs to select one change worth testing.",
  },
  {
    title: "Implement one change",
    icon: Code2,
    description:
      "Apply the selected change to a real serving system, whether it is a kernel optimization or support for a new model. Pass correctness checks before measuring performance.",
  },
  {
    title: "Profile the change",
    icon: ChartNoAxesCombined,
    description:
      "Capture a GPU trace of the modified implementation. Break down execution time across kernels, communication, and idle gaps.",
  },
  {
    title: "Compare measurement with prediction",
    icon: GitCompareArrows,
    description:
      "Line up the measured breakdown against the Simulator's prediction, then trace each gap to the performance model, implementation, or measurement setup.",
  },
  {
    title: "Validate end-to-end performance",
    icon: BadgeCheck,
    description:
      "Measure throughput and latency on the full serving workload and compare with the baseline. Keep, reject, or refine the change, then use the evidence to guide the next simulation pass.",
  },
];

export function Workflow() {
  return (
    <section
      id="workflow"
      className={`section ${s.workflow}`}
      aria-labelledby="workflow-title"
    >
      <div className={`wrap ${s.layout}`}>
        <div className={s.intro} data-reveal>
          <h2 id="workflow-title">
            Close the loop between simulation and serving.
          </h2>
          <p>
            Each pass starts with a measured baseline, tests one simulation-guided
            change in a real serving system, and returns the result to the
            Simulator. The Agent uses that evidence to keep, reject, or refine the
            change and decide what to test next.
          </p>
        </div>
        <ol className={s.stages}>
          {stages.map(({ title, description, icon: Icon }, index) => (
            <li className={s.stage} key={title} data-reveal>
              <div className={s.marker} aria-hidden="true">
                <Icon size={28} />
                <span className={s.number}>0{index + 1}</span>
              </div>
              <div className={s.content}>
                <h3>{title}</h3>
                <p>{description}</p>
              </div>
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}
