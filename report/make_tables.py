#!/usr/bin/env python3
"""Generate every LaTeX table in the report directly from the result files.

Keeping the tables programmatic avoids transcription errors between the CSV/JSON
artefacts and the numbers printed in the PDF.  Run from the report/ directory.
"""
import json
import os
import re

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "tables")
os.makedirs(OUT, exist_ok=True)


def esc(s):
    """Escape a string for LaTeX text mode."""
    s = str(s)
    s = s.replace("\\", r"\textbackslash{}")
    for a, b in [("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                 ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}"), ("<", r"\textless{}"), (">", r"\textgreater{}"),
                 ("|", r"\textbar{}")]:
        s = s.replace(a, b)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def write(name, text):
    with open(os.path.join(OUT, name), "w") as f:
        f.write(text)


SHORT = {
    "Regional representation (Global South)": "Regional (Global South)",
    "Gender and professional roles": "Gender & professions",
    "Socioeconomic status": "Socioeconomic status",
    "Language and code-switching": "Language / code-switching",
    "Cultural assumptions": "Cultural assumptions",
    "Disability representation": "Disability",
    "Caste and religion signalling (India)": "Caste / religion (India)",
    "Age": "Age",
    "Nationality and migration": "Nationality / migration",
    "Religious accommodation": "Religious accommodation",
}
MODELS = ["qwen2.5-7b", "mistral-7b", "phi3.5-mini", "qwen2.5-1.5b"]
MLABEL = {"qwen2.5-7b": "Qwen2.5-7B", "mistral-7b": "Mistral-7B",
          "phi3.5-mini": "Phi-3.5-mini", "qwen2.5-1.5b": "Qwen2.5-1.5B"}


def q1_tables():
    stats = json.load(open(os.path.join(ROOT, "q1/results/stats.json")))
    dims = pd.read_csv(os.path.join(ROOT, "q1/results/dimension_summary.csv"))

    # ---- model overview -------------------------------------------------------------
    scores = pd.read_csv(os.path.join(ROOT, "q1/results/scores.csv"))
    ref = scores[scores.stage == "baseline"].groupby("model")["refusal"].mean()
    rows = []
    for m in MODELS:
        s = stats["models"][m]
        d = dims[dims.model == m]
        top = d.loc[d.disparity_z.idxmax()]
        rows.append(f"{MLABEL[m]} & {s['mean_disparity_z']:.3f} & "
                    f"{s['mean_abs_sentiment_delta']:.3f} & "
                    f"{s['repeat_sign_consistency']:.2f} & "
                    f"{ref.get(m, 0.0):.3f} & "
                    f"{esc(SHORT[top.dimension])} ({top.disparity_z:.2f}) \\\\")
    body = "\n".join(rows)
    write("tab_q1_models.tex", r"""\begin{tabular}{lcccc l}
\toprule
Model & Disparity $z$ $\downarrow$ & Mean $|\Delta s|$ $\downarrow$ & Repeat & Refusal & Largest dimension (mean $z$) \\
 & & & consistency & rate & \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
""")

    # ---- dimension x model ----------------------------------------------------------
    piv = dims.pivot(index="dimension", columns="model", values="disparity_z")
    dim_order = ["Regional representation (Global South)", "Gender and professional roles",
                 "Socioeconomic status", "Language and code-switching", "Cultural assumptions",
                 "Disability representation", "Caste and religion signalling (India)", "Age",
                 "Nationality and migration", "Religious accommodation"]
    lines = []
    for dim in dim_order:
        vals = " & ".join(f"{piv.loc[dim, m]:.3f}" for m in MODELS)
        best = max(MODELS, key=lambda m: piv.loc[dim, m])
        mean_all = piv.loc[dim, MODELS].mean()
        lines.append(f"{esc(SHORT[dim])} & {vals} & {mean_all:.3f} \\\\")
    means = " & ".join(f"{piv[m].mean():.3f}" for m in MODELS)
    write("tab_q1_dims.tex", r"""\begin{tabular}{lcccc c}
\toprule
 & \multicolumn{4}{c}{Composite disparity $z$} & \\
\cmidrule(lr){2-5}
Dimension & Qwen2.5-7B & Mistral-7B & Phi-3.5 & Qwen2.5-1.5B & mean \\
\midrule
""" + "\n".join(lines) + "\n\\midrule\nAll items & " + means + r""" & \\
\bottomrule
\end{tabular}
""")

    # ---- hypothesis support ---------------------------------------------------------
    hs = pd.read_csv(os.path.join(ROOT, "q1/results/hypothesis_support.csv"))
    lines = []
    for m in MODELS:
        sub = hs[hs.model == m]
        lines.append(f"{MLABEL[m]} & {int(sub.supported.sum())}/{len(sub)} & "
                     f"{int(sub.consistent_seeds.sum())}/{len(sub)} \\\\")
    write("tab_q1_hyp.tex", r"""\begin{tabular}{lcc}
\toprule
Model & Direction supported & Supported in all 3 seeds \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
""")

    # ---- mitigation ----------------------------------------------------------------
    ms = pd.read_csv(os.path.join(ROOT, "q1/results/mitigation_summary.csv"))
    mt = pd.read_csv(os.path.join(ROOT, "q1/results/mitigation_tests.csv"))
    mg = mt.merge(ms, on=["intervention", "model"], suffixes=("", "_s"))
    lines = []
    for _, r in mg.sort_values(["intervention", "model"]).iterrows():
        lines.append(f"{esc(r.intervention)} & {MLABEL[r.model]} & {r.baseline_disparity:.3f} & "
                     f"{r.intervention_disparity:.3f} & {r['p_vs_baseline']:.3f} & "
                     f"{r.n_words:.0f} & {r.refusal:.3f} \\\\")
    write("tab_q1_mit.tex", r"""\begin{tabular}{llcccccc}
\toprule
Intervention & Model & Baseline $z$ & After $z$ & $p$ & Words & Refusal \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
""")
    return stats


def q2_table():
    m = json.load(open(os.path.join(ROOT, "q2/results/fedavg_metrics.json")))
    names = {"iid": "IID", "dirichlet_a0.5": r"Dirichlet $\alpha=0.5$",
             "dirichlet_a0.1": r"Dirichlet $\alpha=0.1$",
             "label_skew_2class": "Label skew (2 classes/client)",
             "centralized": "Centralized (no federation)"}
    lines = []
    for k, r in m["runs"].items():
        gap = m["gaps_vs_centralized"].get(k, 0.0)
        if "client_acc_std" in r:
            cstat = f"{r['client_acc_mean']:.3f} / {r['client_acc_std']:.3f} / {r['client_acc_min']:.3f}"
        else:
            cstat = "--"
        rounds = r.get("rounds_to_90pct_of_final", "--")
        lines.append(f"{names[k]} & {r['final_acc']:.3f} & {r.get('best_acc', r['final_acc']):.3f} & "
                     f"{gap:+.3f} & {rounds} & {cstat} & "
                     f"{r['comm_mb_total']:.1f} \\\\")
    write("tab_q2_runs.tex", r"""\setlength{\tabcolsep}{4pt}\begin{tabular}{lcccccc}
\toprule
Method / partition & Final & Best & Gap vs C. & Rounds & Client acc. & Comm. \\
 & acc. & acc. & & to 90\% & mean/std/min & (MB) \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
""")


def q3_tables():
    runs = pd.read_csv(os.path.join(ROOT, "q3/results/dp_runs.csv"))
    mia = pd.read_csv(os.path.join(ROOT, "q3/results/mia.csv"))
    # curve-derived accuracy is identical to runs['test_acc'] for all trained configs
    lines = []
    for _, r in runs.iterrows():
        if r.config == "baseline_nonprivate":
            C = "--"
        elif pd.isna(r["clip"]):
            C = "--"
        else:
            C = f"{r['clip']:.1f}"
        sig = "--" if r.sigma == 0 else f"{r.sigma:.1f}"
        eps = "$\\infty$" if r.epsilon_rdp == float("inf") else f"{r.epsilon_rdp:.2f}"
        epsp = "$\\infty$" if r.epsilon_prv == float("inf") else f"{r.epsilon_prv:.2f}"
        clipf = "--" if pd.isna(r.mean_clip_fraction) else f"{r.mean_clip_fraction:.2f}"
        t = "--" if pd.isna(r.train_time_s) else f"{r.train_time_s:.0f}"
        lines.append(f"{esc(r.config)} & {C} & {sig} & {eps} & {epsp} & {r.test_acc:.3f} & "
                     f"{r.test_loss:.3f} & {t} & {clipf} \\\\")
    write("tab_q3_runs.tex", r"""\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcccccccc}
\toprule
Configuration & $C$ & $\sigma$ & $\varepsilon_{\mathrm{RDP}}$ & $\varepsilon_{\mathrm{PRV}}$ & Test acc. & Test loss & Time (s) & Clipped frac. \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
""")

    lines = []
    for _, r in mia.iterrows():
        e = "$\\infty$" if r.epsilon == float("inf") else f"{r.epsilon:.2f}"
        lines.append(f"{esc(r.config)} & {e} & {r.test_acc:.3f} & {r.loss_auc:.3f} & "
                     f"{r.shadow_auc:.3f} & {r['shadow_tpr@10%fpr']:.3f} & "
                     f"{r['shadow_advantage@10%']:+.3f} \\\\")
    write("tab_q3_mia.tex", r"""\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcccccc}
\toprule
Configuration & $\varepsilon$ & Test acc. & Loss AUC & Shadow AUC & TPR@10\% FPR & Advantage@10\% \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
""")


def prompt_table():
    d = json.load(open(os.path.join(ROOT, "q1/prompts.json")))
    lines = []
    for it in d["items"]:
        tmpl = it["template"]
        a = it["variants"]["A"]
        b = it["variants"]["B"]
        # when the whole prompt lives in the variant, the template is just {attr}
        tmpl_tex = esc(tmpl) if tmpl.strip() != "{attr}" else "-- (full text in A/B)"
        lines.append(
            "\\midrule\n"
            f"\\textbf{{{it['id']}}} & {esc(SHORT[it['dimension']])} & {esc(it['attribute'])} & "
            f"{tmpl_tex} & \\textbf{{A:}} {esc(a)} & \\textbf{{B:}} {esc(b)} \\\\"
        )
    write("tab_prompts.tex", r"""\begin{longtable}{@{}p{0.8cm}p{3.3cm}p{2.1cm}p{7.4cm}p{4.2cm}p{4.2cm}@{}}
\caption{All 20 counterfactual audit items. \texttt{\{attr\}} is replaced by the
attribute value in columns A and B; every other token is identical across the pair.
L1/L2 place the entire prompt in the variant because the attribute \emph{is} the language;
D1 variant B intentionally substitutes an empty string (no disability disclosure).}
\label{tab:prompts}\\
\toprule
ID & Dimension & Attribute & Template & Variant A & Variant B \\
\midrule
\endfirsthead
\toprule
ID & Dimension & Attribute & Template & Variant A & Variant B \\
\midrule
\endhead
\bottomrule
\endfoot
""" + "\n".join(lines) + "\n\\end{longtable}\n")


if __name__ == "__main__":
    q1_tables()
    q2_table()
    q3_tables()
    prompt_table()
    print("tables written to", OUT)
