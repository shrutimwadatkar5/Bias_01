#!/usr/bin/env python3
"""
Q3 - finalise results: reconstruct per-epoch curves from the training log, write the
results JSON, and regenerate the figures from the saved tables.  Does not retrain.
"""
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIGS = os.path.join(HERE, "figures")
os.makedirs(FIGS, exist_ok=True)

LOG = os.path.join(HERE, "run_q3.log")
LINE = re.compile(r"\[(?P<cfg>[^\]]+)\] epoch\s+(?P<ep>\d+) acc=(?P<acc>[\d.]+) loss=(?P<loss>[\d.]+)")


def parse_log():
    rows = []
    with open(LOG) as f:
        for line in f:
            m = LINE.search(line)
            if m:
                rows.append({"config": m.group("cfg"), "epoch": int(m.group("ep")),
                             "test_acc": float(m.group("acc")), "test_loss": float(m.group("loss"))})
    return pd.DataFrame(rows)


def main():
    curves = parse_log()
    curves.to_csv(os.path.join(RES, "dp_curves.csv"), index=False)
    runs = pd.read_csv(os.path.join(RES, "dp_runs.csv"))
    mia = pd.read_csv(os.path.join(RES, "mia.csv"))

    # final accuracy from the curves is authoritative (same source as training)
    final = curves.sort_values("epoch").groupby("config").tail(1)[["config", "test_acc", "test_loss"]]
    final = final.rename(columns={"test_acc": "acc_from_curve", "test_loss": "loss_from_curve"})
    merged = runs.merge(final, on="config", how="left")

    out = {"runs": merged.to_dict(orient="records"),
           "mia": mia.to_dict(orient="records"),
           "delta": 1e-5,
           "hyper": {"batch": 256, "lr": 0.1, "momentum": 0.9, "epochs": 15,
                     "clip_norms": [1.0, 5.0], "noise_multipliers": [0.4, 0.8, 1.2],
                     "dataset": "Fashion-MNIST balanced subset", "n_train": 12000, "n_test": 2000}}
    with open(os.path.join(RES, "dp_results.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)

    # ---- figure 1: privacy-utility ----
    dp = merged[(merged.sigma > 0) & (~merged.config.str.startswith("ablation"))]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for C, g in dp.groupby("clip"):
        g = g.sort_values("epsilon_rdp")
        ax.plot(g["epsilon_rdp"], g["acc_from_curve"], "o-", label=f"DP-SGD, C={C:g}")
    base = merged[merged.config == "baseline_nonprivate"]
    if len(base):
        ax.axhline(base["acc_from_curve"].iloc[0], color="k", ls="--", label="non-private (15 epochs)")
    over = merged[merged.config == "overfit_baseline_30ep"]
    if len(over):
        ax.axhline(over["acc_from_curve"].iloc[0], color="grey", ls=":", label="non-private overfit (30 epochs)")
    ax.set_xscale("log")
    ax.set_xlabel("privacy budget $\\varepsilon$ ($\\delta=10^{-5}$)")
    ax.set_ylabel("test accuracy")
    ax.set_title("Q3: privacy-utility trade-off of DP-SGD")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_q3_acc_vs_eps.png"), dpi=200); plt.close(fig)

    # ---- figure 2: convergence ----
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for name, g in curves.groupby("config"):
        ax.plot(g["epoch"], g["test_acc"], lw=1.5, label=name)
    ax.set_xlabel("epoch"); ax.set_ylabel("test accuracy")
    ax.set_title("Q3: convergence with clipping and noise")
    ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_q3_curves.png"), dpi=200); plt.close(fig)

    # ---- figure 3: MIA ----
    m = mia[np.isfinite(mia.epsilon) | (mia.config == "overfit_baseline_30ep")].copy()
    dpm = m[m.epsilon < 1e6].sort_values("epsilon")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(dpm["epsilon"], dpm["loss_auc"], "o-", label="loss-threshold attack")
    axes[0].plot(dpm["epsilon"], dpm["shadow_auc"], "s-", label="shadow-model attack")
    ov = m[m.config == "overfit_baseline_30ep"]
    axes[0].axhline(ov["loss_auc"].iloc[0], color="grey", ls=":", lw=1.2,
                    label="overfit non-private (loss)")
    axes[0].axhline(0.5, color="k", ls="-.", lw=0.8)
    axes[0].set_xscale("log"); axes[0].set_xlabel("$\\varepsilon$"); axes[0].set_ylabel("MIA ROC-AUC")
    axes[0].set_title("Membership inference vs privacy budget")
    axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)
    axes[1].plot(dpm["epsilon"], dpm["shadow_advantage@10%"], "s-", color="firebrick")
    axes[1].axhline(0, color="k", ls=":")
    axes[1].set_xscale("log"); axes[1].set_xlabel("$\\varepsilon$")
    axes[1].set_ylabel("shadow attack advantage @10% FPR")
    axes[1].set_title("Attack advantage above random guessing")
    axes[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_q3_mia.png"), dpi=200); plt.close(fig)

    # ---- figure 4: ablations ----
    abl = merged[merged.config.str.startswith("ablation")].copy()
    if len(abl):
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
        order = ["ablation_epochs5_C1.0_sigma0.8", "C1.0_sigma0.8",
                 "ablation_epochs30_C1.0_sigma0.8"]
        labels = ["5 epochs", "15 epochs", "30 epochs"]
        vals = [abl.set_index("config")["acc_from_curve"].get(c, np.nan) for c in order]
        axes[0].bar(labels, vals, color=["#8fbcd4", "#5b9bd5", "#2e5f8a"])
        axes[0].set_ylim(0.6, 0.9); axes[0].set_ylabel("test accuracy")
        axes[0].set_title("Effect of training duration (C=1, $\\sigma$=0.8)")
        epss = [abl.set_index("config")["epsilon_rdp"].get(c, np.nan) for c in order]
        ax2 = axes[0].twinx(); ax2.plot(labels, epss, "ko--"); ax2.set_ylabel("$\\varepsilon$")
        d = abl.set_index("config")
        vals2 = [d["acc_from_curve"].get("ablation_n6k_C1.0_sigma0.8", np.nan),
                 d["acc_from_curve"].get("C1.0_sigma0.8", np.nan)]
        axes[1].bar(["n=6,000", "n=12,000"], vals2, color=["#d49a8f", "#b55b45"])
        axes[1].set_ylim(0.6, 0.9); axes[1].set_ylabel("test accuracy")
        axes[1].set_title("Effect of dataset size (C=1, $\\sigma$=0.8)")
        fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_q3_ablation.png"), dpi=200); plt.close(fig)

    print(merged[["config", "sigma", "clip", "epochs", "acc_from_curve", "epsilon_rdp",
                  "epsilon_prv", "train_time_s", "mean_clip_fraction"]].to_string(index=False))
    print("\nMIA:\n", mia[["config", "epsilon", "test_acc", "loss_auc", "shadow_auc",
                           "shadow_tpr@10%fpr"]].to_string(index=False))


if __name__ == "__main__":
    main()
