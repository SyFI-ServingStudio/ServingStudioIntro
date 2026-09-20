import { PageHero } from "../components/PageHero";
import { Advantages } from "../sections/Advantages";

export function Features() {
  return (
    <>
      <PageHero
        eyebrow="Features"
        title={
          <>
            Powerful simulation.
            <br />
            Zero manual coding.
          </>
        }
        description="Sweep dense and MoE deployments in minutes, check the predictions against real serving frameworks, and trace any result down to the kernel and its lower bound—all in the browser, with the Agent."
        image="hero-datacenter-a.webp"
        href="#advantages"
        linkLabel="Explore the features"
      />
      <Advantages standalone />
    </>
  );
}
