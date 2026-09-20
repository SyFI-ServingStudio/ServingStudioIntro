import {
  ArrowRight,
  ArrowUpRight,
  Blocks,
  Gauge,
  ScanLine,
  Search,
  ChartNoAxesCombined,
  MessagesSquare,
} from "lucide-react";
import s from "./FeatureOverview.module.css";

const features = [
  {
    id: "support",
    icon: Blocks,
    title: "Flexible configuration",
    description:
      "Explore dense and MoE models across precisions, GPU layouts, and serving strategies.",
  },
  {
    id: "speed",
    icon: Gauge,
    title: "Fast simulation",
    description:
      "Reach steady state and compare more configurations with a simulator built for speed.",
  },
  {
    id: "accuracy",
    icon: ScanLine,
    title: "Accurate predictions",
    description:
      "Predictions draw on real GPU measurements and are calibrated against vLLM and SGLang.",
  },
  {
    id: "observability",
    icon: Search,
    title: "Full observability",
    description:
      "Follow performance from the whole run down to individual requests, steps, and kernels.",
  },
  {
    id: "optimization",
    icon: ChartNoAxesCombined,
    title: "Optimization insights",
    description:
      "Compare simulated GPU time with model-required work to identify where time could be saved.",
  },
  {
    id: "agent",
    icon: MessagesSquare,
    title: "Zero-code exploration",
    description:
      "Give the Agent a goal and receive experiments, analysis, tradeoffs, and supporting evidence.",
  },
];

export function FeatureOverview() {
  const featuresUrl = `${import.meta.env.BASE_URL}features.html`;
  return (
    <section
      id="features-overview"
      className={s.section}
      aria-labelledby="features-overview-title"
    >
      <div className="wrap">
        <header className={s.header} data-reveal>
          <h2 id="features-overview-title">ServingStudio’s key features.</h2>
          <p>From a serving question to the evidence behind the answer.</p>
          <a className={`button button-primary ${s.cta}`} href={featuresUrl}>
            Explore all features <ArrowRight size={18} aria-hidden="true" />
          </a>
        </header>
        <ul className={s.directory}>
          {features.map(({ id, icon: Icon, title, description }) => (
            <li key={id} data-reveal>
              <a className={s.feature} href={`${featuresUrl}#${id}`}>
                <Icon className={s.icon} size={24} aria-hidden="true" />
                <div>
                  <h3>{title}</h3>
                  <p>{description}</p>
                </div>
                <ArrowUpRight className={s.arrow} size={18} aria-hidden="true" />
              </a>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}
