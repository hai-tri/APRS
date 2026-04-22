"""Pivot final_results.csv into paper-ready LaTeX tables.

Expects a CSV with one row per (model, defense_config) pair — the schema
produced by run_obfuscation_pipeline.py.

Outputs:
  results/tables/asr_main.tex    — ASR across attacks, rows = defense configs
  results/tables/utility.tex     — gsm8k/math500/mmlu/xstest, same row order
  results/tables/asr_main.md     — same content in markdown (eyeball check)
  results/tables/utility.md
"""

import argparse
import csv
import os
from collections import defaultdict


ASR_COLS = [
    ("harmbench_asr",       "HarmBench"),
    ("llamaguard_asr",      "LlamaGuard"),
    ("gcg_asr",             "GCG"),
    ("autodan_asr",         "AutoDAN"),
    ("pair_asr",            "PAIR"),
    ("cipherchat_best_asr", "CipherChat"),
    ("renellm_asr",         "ReNeLLM"),
    ("softopt_asr",         "SoftOpt"),
]

UTILITY_COLS = [
    ("gsm8k_exact_match",     "GSM8K"),
    ("math500_exact_match",   "MATH500"),
    ("mmlu_acc",              "MMLU"),
    ("xstest_over_refusal_rate", "XSTest-OR"),
]

# Row order (pretty name → matcher on defense_type + projection_mode)
ROW_ORDER = [
    ("Undefended",           lambda r: r["defense_type"] == "none"),
    ("APRS full ε=0.025",    lambda r: r["defense_type"] == "obfuscation" and r["projection_mode"] == "full" and _float(r.get("epsilon")) == 0.025),
    ("APRS hadamard ε=0.3",  lambda r: r["defense_type"] == "obfuscation" and r["projection_mode"] == "hadamard" and _float(r.get("epsilon")) == 0.3),
    ("APRS scalar ε=0.3",    lambda r: r["defense_type"] == "obfuscation" and r["projection_mode"] == "scalar_projection" and _float(r.get("epsilon")) == 0.3),
    ("Surgical",             lambda r: r["defense_type"] == "surgical"),
    ("CAST",                 lambda r: r["defense_type"] == "cast"),
    ("Circuit Breakers",     lambda r: r["defense_type"] == "circuit_breakers"),
    ("AlphaSteer",           lambda r: r["defense_type"] == "alphasteer"),
]


def _float(x):
    try: return float(x)
    except: return None


def _fmt(x):
    v = _float(x)
    return "—" if v is None else f"{v:.3f}"


def build_table(rows_by_model, cols, caption, label):
    """Render LaTeX + markdown tables per model."""
    tex_chunks, md_chunks = [], []
    for model, rows in rows_by_model.items():
        pretty_model = model.split("/")[-1]
        # Latex header
        col_spec = "l" + "c" * len(cols)
        header = " & ".join(["Defense"] + [c[1] for c in cols]) + r" \\"
        lines = [
            r"\begin{table}[t]",
            rf"\centering",
            rf"\caption{{{caption} --- {pretty_model}}}",
            rf"\label{{{label}-{pretty_model}}}",
            rf"\begin{{tabular}}{{{col_spec}}}",
            r"\toprule",
            header,
            r"\midrule",
        ]
        md = [f"### {caption} — {pretty_model}", "",
              "| Defense | " + " | ".join(c[1] for c in cols) + " |",
              "| --- | " + " | ".join("---" for _ in cols) + " |"]

        for name, matcher in ROW_ORDER:
            match = [r for r in rows if matcher(r)]
            if not match:
                continue
            r = match[-1]  # last row for that config (latest sweep)
            vals = [_fmt(r.get(c[0])) for c in cols]
            lines.append(" & ".join([name] + vals) + r" \\")
            md.append(f"| {name} | " + " | ".join(vals) + " |")

        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
        tex_chunks.append("\n".join(lines))
        md_chunks.append("\n".join(md))

    return "\n".join(tex_chunks), "\n\n".join(md_chunks)


def _unused_build_combined_dsr(rows_by_model, cols, caption, label):
    """Kept for reference in case we want the AlphaSteer-style combined layout later."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{l{'c' * len(cols)}|c}}",
        r"\toprule",
        " & ".join(["Model"] + [c[1] for c in cols] + [r"Avg DSR \% $\uparrow$"]) + r" \\",
        r"\midrule",
    ]
    md = ["| Model | " + " | ".join(c[1] for c in cols) + " | Avg DSR |",
          "| --- | " + " | ".join("---" for _ in cols) + " | --- |"]

    for model, rows in rows_by_model.items():
        pretty_model = model.split("/")[-1]
        block = []
        for name, matcher in ROW_ORDER:
            match = [r for r in rows if matcher(r)]
            if not match:
                continue
            r = match[-1]
            dsr = []
            for col, _ in cols:
                v = _float(r.get(col))
                dsr.append(None if v is None else (1.0 - v) * 100.0)
            avg = [v for v in dsr if v is not None]
            avg = sum(avg) / len(avg) if avg else None
            label_txt = pretty_model if name == "Undefended" else f"+ {name}"
            block.append((label_txt, dsr, avg))

        # Determine best per column (ignoring Undefended row)
        bests = []
        for j in range(len(cols)):
            vals = [row[1][j] for row in block[1:] if row[1][j] is not None]
            bests.append(max(vals) if vals else None)

        for idx, (label_txt, dsr, avg) in enumerate(block):
            cells = []
            for j, v in enumerate(dsr):
                if v is None:
                    cells.append("--")
                elif idx > 0 and bests[j] is not None and abs(v - bests[j]) < 1e-9:
                    cells.append(rf"\textbf{{{v:.0f}}}")
                else:
                    cells.append(f"{v:.0f}")
            avg_cell = "--" if avg is None else f"{avg:.2f}"
            lines.append(" & ".join([label_txt] + cells + [avg_cell]) + r" \\")
            md.append("| " + " | ".join([label_txt] + [c.replace(r"\textbf{","**").replace("}","**") for c in cells] + [avg_cell]) + " |")
        lines.append(r"\midrule")
        md.append("| --- |" + " --- |" * (len(cols) + 1))

    # Replace final midrule with bottomrule
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    lines.extend([r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines), "\n".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="all_results.csv")
    ap.add_argument("--out_dir", default="results/tables")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.csv) as f:
        rows = list(csv.DictReader(f))

    rows_by_model = defaultdict(list)
    for r in rows:
        rows_by_model[r["model"]].append(r)

    asr_tex, asr_md = build_table(
        rows_by_model, ASR_COLS,
        "Attack Success Rate (lower is better)", "tab:asr")
    util_tex, util_md = build_table(
        rows_by_model, UTILITY_COLS,
        "Utility (higher is better; XSTest-OR lower is better)", "tab:utility")

    for name, content in [
        ("asr_main.tex", asr_tex), ("asr_main.md", asr_md),
        ("utility.tex",  util_tex), ("utility.md",  util_md),
    ]:
        path = os.path.join(args.out_dir, name)
        with open(path, "w") as f:
            f.write(content)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
