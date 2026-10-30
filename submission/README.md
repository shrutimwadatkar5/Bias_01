# RAI-FM (DSE428/DSE628) — Assignment 3

**Submission contents:** report (PDF), all Python code, all prompts, raw model
responses, annotations and result files for Questions 1–3.

> **Before submitting:** fill in your **Name** and **Roll number** on the title page of
> the report (placeholders are left blank on purpose). The 7-minute video is submitted
> separately and is not part of this archive.

## 1. Report

| File | Description |
|---|---|
| `RAI-FM_Assignment3_Report.pdf` | 28-page report: Q1 bias audit, Q2 FedAvg, Q3 DP-SGD + MIA, with full prompt appendix, AI-usage log and reproduction commands. |

The report source is in `code/report/` (`report.tex`, `refs.bib`, generated `tables/`).
It compiles with `pdflatex → bibtex → pdflatex → pdflatex` from `code/report/`; the figure
paths expect `code/q1/figures`, `code/q2/figures`, `code/q3/figures`, which is exactly the
layout used here.

## 2. Code

| File | What it does |
|---|---|
| `code/q1/q1_run.py` | Generates 20 items × 2 variants × 3 seeds × 4 open-weight LLMs (480 responses) with Hugging Face `transformers`; resumable JSONL. |
| `code/q1/q1_score.py` | Deterministic annotation: refusal, sentiment, item-specific favourability, specificity, curated lexicons. |
| `code/q1/q1_analyze.py` | Paired deltas, pooled-SD composite disparity, bootstrap CIs, Wilcoxon/Holm, Friedman, hypothesis support, figures. |
| `code/q1/q1_mitigate.py` | Intervention I1 (fairness system prompt) and I2 (counterfactual self-audit rewrite), 192 responses. |
| `code/q2/q2_fedavg.py` | FedAvg from scratch (5 clients, 20 rounds) vs centralized CNN on a balanced 12k/2k Fashion-MNIST subset; IID, Dirichlet 0.5/0.1, two-class skew. |
| `code/q3/q3_dpsgd.py` | Manual DP-SGD with `torch.func.vmap` per-example gradients, clipping, Gaussian noise, RDP/PRV accounting, membership-inference attacks. |
| `code/q3/q3_extras.py` | Deliberately overfit non-private baseline for the MIA experiment. |
| `code/q3/q3_finalize.py` | Parses training logs into curves/JSON and produces the Q3 figures. |
| `code/report/make_tables.py` | Regenerates every LaTeX table in the report from the result CSVs/JSON. |

All scripts are pure Python and heavily commented. They were run on one NVIDIA A100-40GB
GPU (host `bionlp`, conda env `a3`); seeds are fixed in the source.

## 3. Prompts and model names

- `code/q1/prompts.json` — all 20 counterfactual items, their pre-specified directional
  hypotheses and both attribute values. Appendix A of the report prints every prompt.
- Models audited: `Qwen/Qwen2.5-7B-Instruct`, `mistralai/Mistral-7B-Instruct-v0.3`,
  `microsoft/Phi-3.5-mini-instruct`, `Qwen/Qwen2.5-1.5B-Instruct`.
- Appendix B of the report lists the AI-coding-assistant prompts and the model used
  (`opencode/big-pickle`).

## 4. Results and raw data

| Directory | Contents |
|---|---|
| `results/q1/` | `responses_baseline.jsonl` (480), `responses_mitigation.jsonl` (192), `scores.csv`, `scores_agg.csv`, item/seed deltas, `dimension_summary.csv`, `hypothesis_support.csv`, `model_tests.csv`, `model_pairwise.csv`, `mitigation_*`, `stats.json`. |
| `results/q2/` | `fedavg_metrics.json`, `curves.csv`, `client_acc.csv`, `partitions.npz`. |
| `results/q3/` | `dp_runs.csv`, `mia.csv`, `dp_curves.csv`, `dp_results.json`, `overfit_curve.json`. |

Figures used in the report are in `code/q{1,2,3}/figures/`.

## 5. Headline numbers

- **Q1:** mean composite disparity 0.283 (Qwen2.5-7B), 0.348 (Mistral-7B), 0.405
  (Phi-3.5-mini), 0.517 (Qwen2.5-1.5B) pooled SD; signed sentiment deltas null after Holm
  correction; stereotype and refusal rates ≈ 0; neither mitigation reduces disparity
  reliably.
- **Q2:** centralized 0.889; FedAvg IID 0.885 (gap 0.004); Dirichlet 0.5 → 0.855;
  Dirichlet 0.1 → 0.831; two-class skew → 0.612 final (best 0.707), per-client min 0.035.
- **Q3:** non-private 0.879; DP at σ=0.8 → 0.812 (ε_RDP 6.52) with C=1 and 0.748 with C=5;
  σ=1.2 → 0.795 / 0.636 (ε 2.73). MIA AUC ≈ 0.50–0.52 for all DP runs vs 0.560 non-private
  and 0.579 for the overfit baseline.

## 6. Reproduce

```bash
cd code/q1 && python q1_run.py && python q1_score.py && python q1_analyze.py
python q1_mitigate.py && python q1_score.py && python q1_analyze.py
cd ../q2 && python q2_fedavg.py
cd ../q3 && python q3_dpsgd.py && python q3_extras.py && python q3_finalize.py
cd ../report && python make_tables.py
pdflatex report.tex && bibtex report && pdflatex report.tex && pdflatex report.tex
```
