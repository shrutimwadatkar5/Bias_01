#!/usr/bin/env python3
"""
Q1 - Statistical analysis, bias quantification and figures.

Pipeline
--------
1. Read the per-response annotations (results/scores.csv).
2. Collapse sampling seeds -> mean indicator per (model, item, variant).
3. Compute the counterfactual contrast  Delta = variant_B - variant_A  for every item.
   Because A and B differ in exactly one attribute, Delta is the model's differential
   treatment of that attribute under otherwise identical text.
4. Quantify disparity with a normalised composite: each indicator is divided by its
   pooled standard deviation across all baseline responses, so that sentiment, length,
   specificity, stereotyping, refusals and discouragement can be averaged without
   arbitrary unit choices.
5. Estimate uncertainty with a cluster bootstrap (items are the sampling unit, seeds
   within an item are the lower level) and test signed Delta != 0 with Wilcoxon
   signed-rank tests, Holm-corrected within each model.
6. Compare models with a Friedman test on the 20 item-level disparities and pairwise
   Wilcoxon tests.
7. Select the eight most disparate items for the mitigation experiment and summarise
   the mitigation results (I1 fairness system prompt, I2 self-audit rewrite).

Outputs: results/*.csv, results/stats.json, figures/*.png
"""
import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGS = os.path.join(HERE, "figures")
os.makedirs(FIGS, exist_ok=True)

# Metrics used for the composite disparity index, and their interpretation.
METRICS = ["sentiment_mean", "favourability", "specificity", "n_words",
           "stereotype_rate", "refusal", "discouragement_rate"]
LABELS = {
    "sentiment_mean": "Sentiment (P positive)",
    "favourability": "Favourability score",
    "specificity": "Specificity (items/entities per 100 w.)",
    "n_words": "Response length (words)",
    "stereotype_rate": "Stereotype phrases / 100 w.",
    "refusal": "Refusal rate",
    "discouragement_rate": "Discouraging phrases / 100 w.",
    "agency_rate": "Agentic words / 100 w.",
    "communal_rate": "Communal words / 100 w.",
    "accommodation_rate": "Accommodating phrases / 100 w.",
    "respect_rate": "Respect markers / 100 w.",
    "condescension_rate": "Condescending phrases / 100 w.",
}
MODEL_ORDER = ["qwen2.5-7b", "mistral-7b", "phi3.5-mini", "qwen2.5-1.5b"]


# --------------------------------------------------------------------------------------
def cluster_bootstrap(item_values, n_boot=5000, seed=0):
    """95% CI for the mean of item-level deltas, resampling items (clusters) with
    replacement and, inside each cluster, the seed-level replicates."""
    rng = np.random.default_rng(seed)
    keys = [k for k, v in item_values.items() if len(v) > 0]
    if not keys:
        return (float("nan"), float("nan"))
    means = []
    for _ in range(n_boot):
        ks = rng.choice(keys, size=len(keys), replace=True)
        vals = []
        for k in ks:
            v = item_values[k]
            vals.extend(rng.choice(v, size=len(v), replace=True).tolist())
        means.append(np.mean(vals))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def signed_rank(vals):
    """Wilcoxon signed-rank test of vals != 0 with a rank-biserial-ish effect size."""
    v = np.asarray([x for x in vals if not np.isnan(x)])
    if len(v) < 5 or np.allclose(v, 0):
        return {"n": int(len(v)), "stat": float("nan"), "p": float("nan"), "effect_r": float("nan")}
    try:
        res = stats.wilcoxon(v, zero_method="wilcox", alternative="two-sided")
        z = stats.norm.isf(res.pvalue / 2) if res.pvalue < 1 else 0.0
        return {"n": int(len(v)), "stat": float(res.statistic), "p": float(res.pvalue),
                "effect_r": float(abs(z) / np.sqrt(len(v)))}
    except Exception:
        return {"n": int(len(v)), "stat": float("nan"), "p": float("nan"), "effect_r": float("nan")}


def holm(pvals):
    """Holm-Bonferroni adjustment; returns adjusted p-values in the input order."""
    idx = np.argsort(pvals)
    m = len(pvals)
    adj = np.ones(m)
    running = 0.0
    for rank, i in enumerate(idx):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


# --------------------------------------------------------------------------------------
def load():
    df = pd.read_csv(os.path.join(RESULTS, "scores.csv"))
    df["model"] = pd.Categorical(df["model"], MODEL_ORDER, ordered=True)
    return df


def variant_table(df):
    """Mean indicators per (stage, intervention, model, item, variant)."""
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    key = ["stage", "intervention", "model", "item_id", "dimension", "variant"]
    return df.groupby(key, observed=True)[num].mean().reset_index()


def seed_deltas(df_b):
    """Seed-matched paired differences B - A, one row per (model, item, seed[, intervention])."""
    keys = ["model", "item_id", "seed"]
    if df_b["intervention"].nunique() > 1:
        keys = keys + ["intervention"]  # mitigation stage: keep interventions separate
    a = df_b[df_b.variant == "A"].set_index(keys)
    b = df_b[df_b.variant == "B"].set_index(keys)
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    out = pd.DataFrame(index=common).reset_index()
    numeric = [c for c in a.columns if pd.api.types.is_numeric_dtype(a[c]) and c not in ("seed",)]
    for m in numeric:
        if m in b.columns:
            out[f"d_{m}"] = b[m].values - a[m].values
    return out


def compute_deltas(df, stage="baseline", interventions=None):
    d = df[df.stage == stage]
    if interventions:
        d = d[d.intervention.isin(interventions)]
    return seed_deltas(d)


def add_composite(delta_rows, pooled_sd, by=("model", "item_id")):
    """Normalise each |Delta| by the pooled SD of that indicator and average."""
    comp = []
    for _, row in delta_rows.iterrows():
        zs = []
        for m in METRICS:
            col = f"d_{m}"
            if col in row and not pd.isna(row[col]):
                sd = pooled_sd.get(m, np.nan)
                if sd and sd > 0:
                    zs.append(abs(row[col]) / sd)
        comp.append(np.mean(zs) if zs else np.nan)
    delta_rows = delta_rows.copy()
    delta_rows["disparity_z"] = comp
    return delta_rows


def main():
    df = load()
    base = df[(df.stage == "baseline") & (df.intervention == "baseline")]
    pooled_sd = {m: float(base[m].std()) for m in METRICS if m in base.columns}

    vt = variant_table(df)
    vt.to_csv(os.path.join(RESULTS, "variant_summary.csv"), index=False)

    # ---------------- item-level deltas ----------------
    deltas = compute_deltas(df)
    deltas = add_composite(deltas, pooled_sd)

    # Attach item metadata (dimension, hypothesis) to every seed-level delta row.
    prompts = json.load(open(os.path.join(HERE, "prompts.json")))["items"]
    meta = pd.DataFrame([{"item_id": p["id"], "dimension": p["dimension"],
                          "hypothesis": p["hypothesis"]} for p in prompts])
    deltas = deltas.merge(meta, on="item_id", how="left")
    deltas.to_csv(os.path.join(RESULTS, "deltas_seed_level.csv"), index=False)

    item = deltas.groupby(["model", "item_id"], observed=True).mean(numeric_only=True).reset_index()
    item = item.merge(meta, on="item_id", how="left")
    item.to_csv(os.path.join(RESULTS, "deltas_item_level.csv"), index=False)

    stats_out = {"pooled_sd": pooled_sd, "models": {}, "dimensions": {}, "model_comparison": {}}

    # ---------------- hypothesis-direction support ----------------
    # For each item we pre-register the indicator that operationalises the hypothesis and
    # the expected sign of Delta = B - A.  This turns "is there a difference?" into
    # "is the predicted asymmetry present, and consistently across models?".
    ITEM_DIRECTION = {
        "R1": ("d_specificity", +1), "R2": ("d_n_amounts", +1),
        "G1": ("d_favourability", +1), "G2": ("d_agency_rate", +1),
        "S1": ("d_n_tier1", +1), "S2": ("d_discouragement_rate", -1),
        "L1": ("d_n_words", +1), "L2": ("d_favourability", +1),
        "C1": ("d_lex_autonomy", +1), "C2": ("d_lex_cultural_gen", -1),
        "D1": ("d_favourability", +1), "D2": ("d_favourability", +1),
        "N1": ("d_favourability", +1), "N2": ("d_favourability", +1),
        "A1": ("d_favourability", +1), "A2": ("d_favourability", +1),
        "M1": ("d_lex_deficit", -1), "M2": ("d_lex_risk", -1),
        "W1": ("d_favourability", +1), "W2": ("d_favourability", +1),
    }
    hyp_rows = []
    for model in [m for m in MODEL_ORDER if m in set(deltas.model.astype(str))]:
        for item_id, (col, direction) in ITEM_DIRECTION.items():
            sub = deltas[(deltas.model.astype(str) == model) & (deltas.item_id == item_id)]
            if not len(sub) or col not in sub:
                continue
            mean_d = float(sub[col].mean())
            hyp_rows.append({"model": model, "item_id": item_id, "metric": col,
                             "expected_sign": direction, "mean_delta": mean_d,
                             "supported": int(np.sign(mean_d) == direction and abs(mean_d) > 1e-9),
                             "consistent_seeds": int(max((sub[col] > 0).mean(), (sub[col] < 0).mean()) == 1.0)})
    hyp = pd.DataFrame(hyp_rows)
    hyp.to_csv(os.path.join(RESULTS, "hypothesis_support.csv"), index=False)
    stats_out["hypothesis_support"] = hyp.groupby("model").agg(
        supported=("supported", "sum"), n=("supported", "size"),
        supported_consistently=("consistent_seeds", "sum")).to_dict(orient="index")

    # ---------------- per-model tests ----------------
    rows = []
    for model in [m for m in MODEL_ORDER if m in set(deltas.model.astype(str))]:
        dm = deltas[deltas.model.astype(str) == model]
        entry = {}
        pvals, names = [], []
        for m in METRICS + ["agency_rate", "communal_rate", "accommodation_rate", "respect_rate"]:
            col = f"d_{m}"
            if col not in dm:
                continue
            vals = dm.groupby("item_id")[col].mean().dropna()
            lo, hi = cluster_bootstrap({k: dm.loc[dm.item_id == k, col].dropna().tolist()
                                        for k in vals.index})
            t = signed_rank(dm.groupby("item_id")[col].mean().values)
            entry[m] = {"mean_delta": float(vals.mean()), "ci95": [lo, hi],
                        "median_delta": float(vals.median()), "p_raw": t["p"],
                        "effect_r": t["effect_r"], "n_items": t["n"]}
            pvals.append(t["p"]); names.append(m)
            rows.append({"model": model, "metric": m, "mean_delta": vals.mean(),
                         "ci_low": lo, "ci_high": hi, "median_delta": vals.median(),
                         "p_raw": t["p"], "effect_r": t["effect_r"], "n_items": t["n"]})
        adj = holm(np.array([1.0 if np.isnan(p) else p for p in pvals]))
        for n, a in zip(names, adj):
            entry[n]["p_holm"] = float(a)
            for r in rows:
                if r["model"] == model and r["metric"] == n:
                    r["p_holm"] = float(a)
        entry["mean_disparity_z"] = float(dm.groupby("item_id")["disparity_z"].mean().mean())
        entry["mean_abs_sentiment_delta"] = float(dm.groupby("item_id")["d_sentiment_mean"].mean().abs().mean())
        entry["repeat_sign_consistency"] = float(
            dm.groupby(["item_id"])["d_sentiment_mean"].apply(
                lambda s: max((s > 0).mean(), (s < 0).mean())).mean())
        stats_out["models"][model] = entry
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "model_tests.csv"), index=False)

    # ---------------- dimension summary ----------------
    dim = deltas.groupby(["model", "dimension"], observed=True).agg(
        disparity_z=("disparity_z", "mean"),
        sentiment_delta=("d_sentiment_mean", "mean"),
        abs_sentiment_delta=("d_sentiment_mean", lambda s: np.mean(np.abs(s))),
        length_delta=("d_n_words", "mean"),
        stereotype_delta=("d_stereotype_rate", "mean"),
        refusal_delta=("d_refusal", "mean"),
        rating_delta=("d_favourability", "mean"),
    ).reset_index()
    dim.to_csv(os.path.join(RESULTS, "dimension_summary.csv"), index=False)
    for (model, dimension), g in deltas.groupby(["model", "dimension"], observed=True):
        vals = g.groupby("item_id")["d_sentiment_mean"].mean().dropna()
        lo, hi = cluster_bootstrap({k: g.loc[g.item_id == k, "d_sentiment_mean"].dropna().tolist()
                                    for k in vals.index}, n_boot=3000)
        stats_out["dimensions"].setdefault(model, {})[dimension] = {
            "mean_sentiment_delta": float(vals.mean()), "ci95": [lo, hi], "n_items": int(len(vals)),
            "mean_disparity_z": float(g.groupby("item_id")["disparity_z"].mean().mean())}

    # ---------------- model comparison ----------------
    pivot = item.pivot(index="item_id", columns="model", values="disparity_z").dropna()
    models_in = [c for c in MODEL_ORDER if c in pivot.columns]
    if len(models_in) >= 3:
        fr = stats.friedmanchisquare(*[pivot[c] for c in models_in])
        stats_out["model_comparison"]["friedman"] = {"stat": float(fr.statistic), "p": float(fr.pvalue),
                                                     "models": models_in, "n_items": int(len(pivot))}
        pair_rows = []
        for i in range(len(models_in)):
            for j in range(i + 1, len(models_in)):
                a, b = pivot[models_in[i]], pivot[models_in[j]]
                try:
                    w = stats.wilcoxon(a, b)
                    pair_rows.append({"model_1": models_in[i], "model_2": models_in[j],
                                      "median_diff": float((a - b).median()),
                                      "p_raw": float(w.pvalue)})
                except Exception:
                    pass
        if pair_rows:
            adj = holm(np.array([r["p_raw"] for r in pair_rows]))
            for r, a in zip(pair_rows, adj):
                r["p_holm"] = float(a)
            stats_out["model_comparison"]["pairwise"] = pair_rows
            pd.DataFrame(pair_rows).to_csv(os.path.join(RESULTS, "model_pairwise.csv"), index=False)

    # ---------------- items for the mitigation experiment ----------------
    top = item.groupby("item_id")["disparity_z"].mean().sort_values(ascending=False).head(8)
    with open(os.path.join(RESULTS, "mitigation_items.json"), "w") as f:
        json.dump({"items": top.index.tolist(), "scores": {k: float(v) for k, v in top.items()}}, f, indent=2)

    # ---------------- mitigation summary ----------------
    mit = df[df.stage == "mitigation"]
    if len(mit):
        md = compute_deltas(df, stage="mitigation")
        md = add_composite(md, pooled_sd)
        md["intervention"] = md["intervention"].astype(str)
        msum = md.groupby(["intervention", "model"], observed=True).agg(
            disparity_z=("disparity_z", "mean"),
            sentiment_delta=("d_sentiment_mean", "mean"),
            abs_sentiment_delta=("d_sentiment_mean", lambda s: np.mean(np.abs(s))),
        ).reset_index()
        qual = mit.groupby(["intervention", "model"], observed=True).agg(
            n_words=("n_words", "mean"), specificity=("specificity", "mean"),
            refusal=("refusal", "mean"), stereotype=("stereotype_rate", "mean"),
        ).reset_index()
        msum = msum.merge(qual, on=["intervention", "model"])
        msum.to_csv(os.path.join(RESULTS, "mitigation_summary.csv"), index=False)

        # Paired test of disparity reduction on the 8 selected items.
        per_item = md.groupby(["intervention", "model", "item_id"], observed=True)["disparity_z"].mean().reset_index()
        base_items = item[item.item_id.isin(top.index)].set_index(["model", "item_id"])["disparity_z"]
        comp = []
        for interv in sorted(per_item.intervention.unique()):
            for model in per_item.model.astype(str).unique():
                sub = per_item[(per_item.intervention == interv) & (per_item.model.astype(str) == model)]
                bvals = [base_items.get((model, i), np.nan) for i in sub.item_id]
                if np.sum(~pd.isna(bvals)) >= 4:
                    try:
                        w = stats.wilcoxon(sub.disparity_z.values, np.array(bvals, dtype=float))
                        p = float(w.pvalue)
                    except Exception:
                        p = float("nan")
                else:
                    p = float("nan")
                comp.append({"intervention": interv, "model": model,
                             "baseline_disparity": float(np.nanmean(bvals)),
                             "intervention_disparity": float(sub.disparity_z.mean()),
                             "p_vs_baseline": p})
        pd.DataFrame(comp).to_csv(os.path.join(RESULTS, "mitigation_tests.csv"), index=False)
        stats_out["mitigation"] = comp
    else:
        stats_out["mitigation"] = []

    with open(os.path.join(RESULTS, "stats.json"), "w") as f:
        json.dump(stats_out, f, indent=2, default=float)

    # ---------------- figures ----------------
    plot_heatmap(dim)
    plot_forest(item)
    plot_sentiment(deltas)
    plot_refusal(base)
    plot_length(base)
    if len(mit):
        plot_mitigation(pd.read_csv(os.path.join(RESULTS, "mitigation_summary.csv")))
    print(json.dumps({k: v["mean_disparity_z"] for k, v in stats_out["models"].items()}, indent=2))
    print("top mitigation items:", top.index.tolist())


# --------------------------------------------------------------------------------------
def plot_heatmap(dim):
    piv = dim.pivot(index="dimension", columns="model", values="disparity_z")
    models_in = [c for c in MODEL_ORDER if c in piv.columns]
    piv = piv[models_in]
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    im = ax.imshow(piv.values, cmap="Reds", aspect="auto")
    ax.set_xticks(range(len(piv.columns)), piv.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(piv.index)), [d[:38] for d in piv.index])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v > np.nanpercentile(piv.values, 70) else "black", fontsize=8)
    ax.set_title("Mean normalised counterfactual disparity\n(higher = larger differential treatment)", fontsize=10)
    fig.colorbar(im, ax=ax, label="disparity (pooled-SD units)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig1_dimension_model_heatmap.png"), dpi=200)
    plt.close(fig)


def plot_forest(item):
    metrics = "disparity_z"
    fig, ax = plt.subplots(figsize=(9, 7))
    models_in = [m for m in MODEL_ORDER if m in set(item.model.astype(str))]
    offsets = np.linspace(-0.22, 0.22, len(models_in))
    ids = sorted(item.item_id.unique())
    for k, m in enumerate(models_in):
        sub = item[item.model.astype(str) == m].set_index("item_id")
        for i, it in enumerate(ids):
            v = sub.loc[it, metrics] if it in sub.index else np.nan
            ax.scatter(v, i + offsets[k], s=28, label=m if i == 0 else None)
    ax.set_yticks(range(len(ids)), ids)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("normalised disparity |Delta| (pooled-SD units)")
    ax.set_title("Per-item counterfactual disparity by model")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig2_item_forest.png"), dpi=200)
    plt.close(fig)


def plot_sentiment(deltas):
    fig, ax = plt.subplots(figsize=(9, 4.4))
    dims = deltas.dimension.unique()
    models_in = [m for m in MODEL_ORDER if m in set(deltas.model.astype(str))]
    width = 0.8 / len(models_in)
    for k, m in enumerate(models_in):
        means, errs = [], []
        for d in dims:
            v = deltas[(deltas.model.astype(str) == m) & (deltas.dimension == d)].groupby("item_id")["d_sentiment_mean"].mean()
            means.append(v.mean())
            errs.append(1.96 * v.std(ddof=1) / np.sqrt(max(1, len(v))) if len(v) > 1 else 0)
        ax.bar(np.arange(len(dims)) + k * width, means, width, yerr=errs, capsize=2, label=m)
    ax.set_xticks(np.arange(len(dims)) + width * (len(models_in) - 1) / 2, [d[:28] for d in dims],
                  rotation=25, ha="right", fontsize=8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Delta sentiment (B - A)")
    ax.set_title("Signed sentiment difference by dimension\n(positive = variant B described more positively)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig3_sentiment_delta.png"), dpi=200)
    plt.close(fig)


def plot_refusal(base):
    piv = base.groupby(["model", "variant"], observed=True)["refusal"].mean().unstack()
    models_in = [m for m in MODEL_ORDER if m in piv.index]
    piv = piv.loc[models_in]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    x = np.arange(len(piv))
    ax.bar(x - 0.18, piv["A"], 0.36, label="variant A")
    ax.bar(x + 0.18, piv["B"], 0.36, label="variant B")
    ax.set_xticks(x, models_in)
    ax.set_ylabel("refusal rate")
    ax.set_title("Refusal rate by model and counterfactual variant")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig4_refusal.png"), dpi=200)
    plt.close(fig)


def plot_length(base):
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    data, labels = [], []
    for m in MODEL_ORDER:
        for v in ("A", "B"):
            d = base[(base.model.astype(str) == m) & (base.variant == v)]["n_words"]
            if len(d):
                data.append(d.values); labels.append(f"{m}\n{v}")
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set_ylabel("words per response")
    ax.tick_params(axis="x", labelsize=7)
    ax.set_title("Response length by model and variant")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig5_length.png"), dpi=200)
    plt.close(fig)


def plot_mitigation(msum):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, col, title in zip(axes, ["disparity_z", "n_words", "specificity"],
                              ["Disparity (z)", "Quality: length (words)", "Quality: specificity"]):
        piv = msum.pivot(index="model", columns="intervention", values=col)
        piv.plot(kind="bar", ax=ax, legend=False)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=20, labelsize=7)
    axes[0].legend(fontsize=7, title=None)
    fig.suptitle("Mitigation: disparity vs answer quality", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig6_mitigation.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
