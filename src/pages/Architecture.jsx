import {
  ArrowRight,
  CircuitBoard,
  Database,
  ChartSpline,
  Gauge,
} from "lucide-react";
import { SevenLayers } from "../components/SevenLayers";
import { LauncherDemo } from "../components/LauncherDemo";
import { ServingHierarchy } from "../components/ServingHierarchy";
import { CostTree } from "../components/CostTree";
import { PageHero } from "../components/PageHero";
import s from "./Architecture.module.css";

const stages = [
  {
    icon: CircuitBoard,
    title: "A real kernel, on a real GPU",
    body: "Run a real kernel with concrete inputs, time it, and record the result on the target GPU.",
  },
  {
    icon: Database,
    title: "One row in the database",
    body: "The row stores the measured timing for reuse across experiments.",
  },
  {
    icon: ChartSpline,
    title: "An interpolating cache",
    body: "The cache interpolates measured samples for fast cost queries.",
  },
  {
    icon: Gauge,
    title: "The cost at this shape",
    body: "The simulator evaluates the cache for each input shape as the run progresses.",
  },
];

export function Architecture() {
  return (
    <div className={s.page}>
      <PageHero
        eyebrow="Architecture"
        title={
          <>
            How kernel measurements
            <br />
            become serving predictions.
          </>
        }
        description="A kernel is measured once on the target GPU. Higher layers compose that measurement into a model, then into the workers and pools that serve requests. The top layer predicts a whole serving run."
        image="hero-datacenter-c.webp"
        href="#measurement"
        linkLabel="Explore the architecture"
      />
      <section
        className={`wrap ${s.foundation}`}
        id="measurement"
        aria-labelledby="measurement-title"
      >
        <div className={s.heading} data-reveal>
          <h2 id="measurement-title">
            Every prediction starts with a measurement.
          </h2>
          <p>
            Kernel timings come from measurements on real hardware. During
            simulation, cached costs are evaluated for the current shapes and
            composed into a serving prediction. That prediction can be traced back
            to the kernels that support it.
          </p>
        </div>
        <ol className={s.measureChain}>
          {stages.map(({ icon: Icon, title, body }, index) => (
            <li key={title}>
              <Icon
                className={s.stageIcon}
                size={36}
                strokeWidth={1.5}
                aria-hidden="true"
              />
              <h3>{title}</h3>
              <p>{body}</p>
              {index < stages.length - 1 && (
                <ArrowRight className={s.arrow} size={24} aria-hidden="true" />
              )}
            </li>
          ))}
        </ol>
      </section>
      <section
        className={`wrap ${s.foundation}`}
        id="cost-tree"
        aria-labelledby="cost-tree-title"
      >
        <div className={s.heading} data-reveal>
          <h2 id="cost-tree-title">Assemble kernels into a model.</h2>
          <p>
            A cost tree captures how kernels run in sequence, across parallel ranks,
            and through repeated layers. Build it once, then evaluate its leaves for
            each new batch.
          </p>
        </div>
        <CostTree />
      </section>
      <section
        className={`wrap ${s.foundation}`}
        id="serving-hierarchy"
        aria-labelledby="serving-hierarchy-title"
      >
        <div className={s.heading} data-reveal>
          <h2 id="serving-hierarchy-title">
            Organize workers into a serving system.
          </h2>
          <p>
            Workers manage local requests. Pools organize workers. The orchestrator
            coordinates work across pools.
          </p>
        </div>
        <ServingHierarchy />
      </section>
      <section
        className={`wrap ${s.foundation}`}
        id="launcher-dsl"
        aria-labelledby="launcher-title"
      >
        <div className={s.heading} data-reveal>
          <h2 id="launcher-title">Sweep parameters systematically.</h2>
          <p>
            Define a serving setup in YAML, then vary request rates, GPU allocation,
            and kernel backends. The launcher expands the combinations and applies
            your constraints.
          </p>
        </div>
        <LauncherDemo />
      </section>
      <section
        className={`wrap ${s.foundation}`}
        id="layer-stack"
        aria-labelledby="stack-title"
      >
        <div className={s.heading} data-reveal>
          <h2 id="stack-title">
            Seven layers.
            <br />
            Explicit responsibilities.
          </h2>
          <p>
            Timing flows from layer 1, the kernel, to layer 7, the run. State starts
            at layer 5, the worker. Select a layer to inspect its contract.
          </p>
        </div>
        <SevenLayers />
      </section>
    </div>
  );
}
