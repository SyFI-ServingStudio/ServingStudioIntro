"""Export the Kernel Library prototype data from a ServingStudio Sim checkout.

Reads profile.db read-only (no GPU, no torch) and writes the JSON the
prototype page fetches from public/kernel-library/v1/. The file layout mirrors
the planned read-only API, so the page can later switch to the API by changing
one base URL.

    python3 scripts/kernel-library/export.py --sim ../ServingStudioSim
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from docs import DOCS, TITLES

HERE = Path(__file__).resolve().parent
OUT = HERE.parent.parent / "public" / "kernel-library" / "v1"

# Columns that are not kernel arguments. Everything between ``backend`` and
# ``profiler_git_hash`` is an argument.
PROVENANCE = ("profiler_git_hash", "profiler_run_at", "cuda_version", "driver_version", "backend_version")
COMPUTE_METRICS = ("time_ms", "tflops", "memory_bandwidth_gbps", "energy_j")
COMM_METRICS = ("time_ms", "algbw_gbps", "busbw_gbps", "energy_j")

# Models, keyed by the L4 arch file stem prefix, with the family the catalog
# groups them under. The catalog shows families only, so its "Used by" column
# stays short as variants are added; the detail page lists every model.
ARCH_FAMILIES = (
    ("llama3_", "Llama 3", "Llama"),
    ("qwen36_", "Qwen3.6", "Qwen"),
    ("qwen3_attn_", "Qwen3", "Qwen"),
    ("qwen3_", "Qwen3-MoE", "Qwen"),
    ("glm52_", "GLM-5.2", "GLM"),
    ("glm53_", "GLM-5.3", "GLM"),
    ("deepseek_v4_", "DeepSeek-V4", "DeepSeek"),
)
NOT_ARCH = {"build", "config", "contract", "mod", "model_cfg", "moe_model_cfg", "glm52_model_cfg"}

# (category, subcategory, pattern), first match wins. Attention is split by
# the attention a model runs, so the kernels one layer needs sit together:
# DSA holds the indexer, top-k and sparse MLA of DeepSeek-V3.2-style layers
# (GLM-5.2, DeepSeek-V4, whose compressor is part of it), MLA the dense
# latent-attention pieces, Gated DeltaNet the linear-attention layers.
CATEGORY_RULES = (
    ("Communication", None, r"^(all_reduce|moe_alltoall|moe_ep_|p2p_)"),
    ("Attention", "DSA", r"^(dsa_|deepseek_v4_(indexer|sparse|packed_cache|qnorm_rope_kv))"),
    ("Attention", "Gated DeltaNet", r"^gdn_"),
    ("Attention", "MLA", r"^(mla_|vllm_mla_)"),
    ("Attention", "MHA / GQA", r"^(flashinfer_attn_|kv_cache_append$)"),
    ("MoE", None, r"(moe|grouped_gemm|swiglu)"),
    ("GEMM", None, r"gemm"),
    ("Quantization", None, r"quant"),
    ("Normalization", None, r"norm"),
    ("Other", None, r""),
)
# Display order: the kernels most models run first.
CATEGORY_ORDER = ["GEMM", "Attention", "MoE", "Communication", "Normalization", "Quantization", "Other"]
SUBCATEGORIES = {
    "Attention": [
        {"name": "MHA / GQA", "summary": "Dense attention over a per-head KV cache, with query heads sharing KV heads under GQA."},
        {"name": "MLA", "summary": "Multi-head latent attention: every head reads one compressed KV latent per token."},
        {"name": "DSA", "summary": "Sparse MLA: an indexer scores the cache and each query attends only to its top-k tokens."},
        {"name": "Gated DeltaNet", "summary": "Linear attention: a gated delta-rule state stands in for the KV cache."},
    ]
}


def classify(kind: str) -> tuple[str, str | None]:
    return next((name, sub) for name, sub, pattern in CATEGORY_RULES if re.search(pattern, kind))


# Precision is what a reader filters by: the serving precision a row belongs to.
# A row takes the most quantized type among its element-type arguments, so an
# NVFP4 MoE with bf16 activations is NVFP4. fp32 and integer columns are
# accumulators and indices and do not count; a kernel with nothing else is
# "Any" and applies at every precision. Scale and layout formats are ignored.
PRECISION_ORDER = ["BF16", "FP16", "FP8", "MXFP4", "NVFP4", "Any"]
_QUANTIZED = ["NVFP4", "MXFP4", "FP8"]


def _precision_of(value: str) -> str | None:
    value = str(value).lower()
    for prefix, label in (("nvfp4", "NVFP4"), ("mxfp4", "MXFP4"), ("fp8", "FP8"), ("bf16", "BF16"), ("fp16", "FP16")):
        if value.startswith(prefix):
            return label
    return None


def precision_columns(args: list[str]) -> list[str]:
    return [a for a in args if a.endswith("dtype") or a in ("weight_format", "activation_format")]


def row_precision(kind: str, backend: str, values: list) -> str:
    found = {p for p in map(_precision_of, values) if p}
    for label in _QUANTIZED:
        if label in found:
            return label
    # The quantized operand is sometimes only named by the kernel or backend,
    # e.g. fp8_block_quant takes bf16 in, or mxfp4_marlin_moe_gemm's weights.
    for label in _QUANTIZED:
        if label.lower() in kind or label.lower() in backend:
            return label
    for label in ("BF16", "FP16"):
        if label in found:
            return label
    return "Any"


def camel(stem: str) -> str:
    return "".join(part.title() for part in stem.split("_"))


def used_by(sim: Path, kinds: list[str]) -> dict[str, list[str]]:
    """Model families whose L4 arch reaches each kernel through worklets and ops.

    A plain reference walk over the Rust sources: file A reaches file B when A
    names B's module stem or its CamelCase type. Good enough for a prototype;
    the real exporter should ask the arch builder instead.
    """
    src = sim / "simulator" / "src"
    files = {}
    for layer in ("arch", "worklet", "op"):
        for path in (src / layer).rglob("*.rs"):
            if path.stem != "mod":
                files[(layer, path.stem)] = path.read_text()

    # Worklet and op types are spelled <Camel>Worklet..., <Camel>Op... or <Camel>.
    module_patterns = {
        key: re.compile(rf"\b({re.escape(key[1])}\b|{re.escape(camel(key[1]))}(Worklet|Op|\b))")
        for key in files
        if key[0] != "arch"
    }
    # Kernel types are spelled <Camel>Kernel...; atomic ops alias them as <Camel>Op.
    kernel_patterns = {
        kind: re.compile(rf"\b({re.escape(kind)}\b|{re.escape(camel(kind))}(Kernel|Op|Spec\b|Config\b|Input\b))")
        for kind in kinds
    }
    direct = {key: {k for k, p in kernel_patterns.items() if p.search(text)} for key, text in files.items()}
    edges = {
        key: {other for other, p in module_patterns.items() if other != key and p.search(text)}
        for key, text in files.items()
    }

    result = defaultdict(set)
    for (layer, stem), _ in files.items():
        if layer != "arch" or stem in NOT_ARCH:
            continue
        family = next((m for prefix, m, _ in ARCH_FAMILIES if stem.startswith(prefix)), None)
        if family is None:
            continue
        seen, stack = set(), [(layer, stem)]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            for kind in direct[node]:
                result[kind].add(family)
            stack.extend(edges[node])
    order = [m for _, m, _ in ARCH_FAMILIES]
    return {kind: sorted(result.get(kind, ()), key=order.index) for kind in kinds}


def module_title(sim: Path, kind: str) -> str:
    path = sim / "profiling" / "kernels" / f"{kind}.py"
    text = path.read_text() if path.exists() else ""
    match = re.match(r'\s*"""(.+?)(?:\n|""")', text)
    title = match.group(1).strip() if match else kind
    return re.sub(r"\s*kernel kind\.?$", "", title).rstrip(".")


def export_kernel(conn, kind, meta):
    cols = [r[1] for r in conn.execute(f"pragma table_info('{kind}')")]
    args = cols[cols.index("backend") + 1 : cols.index("profiler_git_hash")]
    family = "comm" if "busbw_gbps" in cols else "compute"
    metrics = COMM_METRICS if family == "comm" else COMPUTE_METRICS
    select = ["gpu_name", "backend", *args, *metrics, *PROVENANCE]
    quoted = ", ".join(f'"{c}"' for c in select)
    order = ", ".join(f'"{c}"' for c in ["gpu_name", "backend", *args])
    provenance, prov_index, rows = [], {}, []
    for rec in conn.execute(f'select {quoted} from "{kind}" order by {order}'):
        rec = dict(zip(select, rec))
        prov = tuple(rec[c] for c in PROVENANCE)
        if prov not in prov_index:
            prov_index[prov] = len(provenance)
            provenance.append(dict(zip(PROVENANCE, prov)))
        values = []
        for m in metrics:
            v = rec[m]
            # energy_j is written as 0.0 when the runner did not measure it.
            values.append(None if (m == "energy_j" and not v) else round(v, 6))
        rows.append([rec["gpu_name"], rec["backend"], *[rec[a] for a in args], *values, prov_index[prov]])
    return {
        "kind": kind,
        "metric_family": family,
        "columns": ["gpu", "backend", *args, *metrics, "provenance"],
        "args": args,
        "metrics": list(metrics),
        "provenance": provenance,
        "rows": rows,
        **meta,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", type=Path, default=HERE.parents[2] / "ServingStudioSim")
    ns = parser.parse_args()
    sim = ns.sim.resolve()
    db = sim / "profiling" / "profile.db"
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.execute("pragma query_only = on")

    kinds = [
        r[0]
        for r in conn.execute(
            "select name from sqlite_master where type='table' "
            "and name not in ('_db_metadata','sqlite_sequence') order by name"
        )
    ]
    models = used_by(sim, kinds)
    spec = json.loads((sim / "gpu" / "spec.json").read_text())["gpus"]

    catalog, gpus_seen = [], Counter()
    for kind in kinds:
        cols = [r[1] for r in conn.execute(f"pragma table_info('{kind}')")]
        args = cols[cols.index("backend") + 1 : cols.index("profiler_git_hash")]
        pcols = precision_columns(args)
        pick = "".join(f', "{c}"' for c in pcols)
        counts = Counter()
        for g, b, *values in conn.execute(f'select gpu_name, backend{pick} from "{kind}"'):
            counts[(g, b, row_precision(kind, b, values))] += 1
        coverage = [
            {"gpu": g, "backend": b, "precision": p, "rows": n}
            for (g, b, p), n in sorted(counts.items())
        ]
        total = sum(c["rows"] for c in coverage)
        for c in coverage:
            gpus_seen[c["gpu"]] += c["rows"]
        doc = DOCS.get(kind, {})
        catalog.append(
            {
                "kind": kind,
                "title": doc.get("title") or TITLES.get(kind) or module_title(sim, kind),
                "summary": doc.get("summary"),
                "category": classify(kind)[0],
                "subcategory": classify(kind)[1],
                "metric_family": "comm" if "busbw_gbps" in cols else "compute",
                "args": args,
                "rows": total,
                "coverage": coverage,
                "used_by": models[kind],
                "has_reference": (
                    any((sim / "profiling" / "runners").rglob(f"{kind}_reference.py"))
                ),
                "detail": kind in DOCS,
            }
        )

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "kernels").mkdir(exist_ok=True)
    for kind, doc in DOCS.items():
        ref = doc.get("reference")
        if ref and ref.get("path"):
            ref = {**ref, "source": (sim / ref["path"]).read_text()}
        meta = {
            **doc,
            "arg_docs": {
                name: {"role": role, "unit": unit, "meaning": meaning}
                for name, (role, unit, meaning) in doc["arg_docs"].items()
            },
            "backends": {
                name: {"summary": summary, "link": link}
                for name, (summary, link) in doc["backends"].items()
            },
            "reference": ref,
            "category": classify(kind)[0],
            "subcategory": classify(kind)[1],
            "used_by": models[kind],
        }
        data = export_kernel(conn, kind, meta)
        (OUT / "kernels" / f"{kind}.json").write_text(json.dumps(data, separators=(",", ":")))

    peaks = {}
    for name in gpus_seen:
        entry = next((g for g in spec if name in g.get("aliases", []) or name == g["name"]), None)
        if entry:
            peaks[name] = {
                k: entry.get(k)
                for k in (
                    "name",
                    "mem_bandwidth_gbps",
                    "bf16_tflops",
                    "fp16_tflops",
                    "fp8_tflops",
                    "interconnect",
                    "interconnect_bandwidth_gbps",
                )
            }

    metadata = dict(conn.execute("select key, value from _db_metadata"))
    digest = hashlib.sha256(db.read_bytes()).hexdigest()[:12]
    last_run = max(
        conn.execute(f'select max(profiler_run_at) from "{k}"').fetchone()[0] or "" for k in kinds
    )
    (OUT / "catalog.json").write_text(
        json.dumps(
            {
                "snapshot": {
                    "id": f"{last_run[:10]}-{digest}",
                    "last_measured_at": last_run,
                    "schema": metadata.get("schema_hash"),
                    "license": "Apache-2.0",
                },
                "categories": CATEGORY_ORDER,
                "subcategories": SUBCATEGORIES,
                "precisions": PRECISION_ORDER,
                "gpus": [{"name": g, "rows": n, "peak": peaks.get(g)} for g, n in gpus_seen.most_common()],
                "models": [{"name": m, "family": f} for _, m, f in ARCH_FAMILIES],
                "kernels": catalog,
            },
            indent=1,
        )
    )
    print(f"{len(kinds)} kernels, {sum(gpus_seen.values())} rows, {len(DOCS)} detail pages -> {OUT}")


if __name__ == "__main__":
    main()
