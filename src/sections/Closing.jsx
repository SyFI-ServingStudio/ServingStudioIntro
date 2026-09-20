import { ArrowRight, ArrowUpRight } from "lucide-react";
import { Brand } from "../components/Brand";
import s from "./Closing.module.css";

export function Closing() {
  return (
    <>
      <section className={s.closing} id="start" aria-labelledby="closing-title">
        <div className="wrap" data-reveal>
          <h2 id="closing-title">
            Start exploring your serving setup with ServingStudio.
          </h2>
          <a
            className={`button button-primary`}
            href="https://github.com/SyFI-ServingStudio/ServingStudio"
            target="_blank"
            rel="noreferrer"
          >
            Get started on GitHub <ArrowRight size={16} />
          </a>
        </div>
      </section>
      <footer className={`wrap ${s.footer}`}>
        <Brand className={s.footerBrand} />
        <span>Simulate, understand, and optimize LLM serving.</span>
        <a
          href="https://github.com/SyFI-ServingStudio/ServingStudioIntro"
          target="_blank"
          rel="noreferrer"
        >
          Page source <ArrowUpRight size={14} />
        </a>
      </footer>
    </>
  );
}
