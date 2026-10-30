#!/usr/bin/env python3
"""
Q3 - Differentially private CNN training (DP-SGD) and membership inference.

Core implementation (all in this file)
--------------------------------------
Per-example gradients are computed with `torch.func.vmap(grad(...))`, i.e. one vectorised
Jacobian-vector sweep over the mini-batch, then:

    g_i_clipped = g_i * min(1, C / ||g_i||_2)              (per-example clipping, eq. 2)
    g_tilde     = (1/B) * ( sum_i g_i_clipped + N(0, sigma^2 C^2 I) )   (eq. 3)

and the noisy sum is assigned to `.grad` before the SGD step.  Clipping bounds the
sensitivity of the batch update to any single example (to C/B); the Gaussian noise hides
which example was present.  BatchNorm is deliberately absent from the architecture
because batch statistics leak cross-example information and break the per-example
gradient decomposition.

Privacy accounting uses the RDP accountant for the Poisson-subsampled Gaussian mechanism
(`opacus.accountants.RDPAccountant`, the standard implementation of Mironov et al. 2019);
epsilon is converted to (epsilon, delta=1e-5).  The accountant only needs (q, sigma, steps),
so it is independent of the training code and can be audited separately.

Membership inference attack (MIA)
---------------------------------
The adversary sees only the trained model.  Four attacks are evaluated on the 12k training
examples (members) versus the 2k held-out test examples (non-members):
  * loss-threshold attack        (Yeom et al. 2018)
  * maximum-confidence attack
  * entropy attack
  * shadow-model attack          (Shokri et al. 2017): four non-private shadow models are
    trained on random halves of the training set; a logistic regression maps
    (loss, confidence, entropy, correctness) to membership and is applied to the target.
Reported: ROC-AUC, TPR at FPR = 1% and 10%, and advantage (TPR - FPR) at 10% FPR.

Outputs: results/dp_results.json, results/dp_runs.csv, results/mia.csv,
         results/grad_norms.npz, figures/*.png
"""
import argparse
import copy
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from torch.func import functional_call, grad, vmap
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "q2"))
from q2_fedavg import SmallCNN, load_data  # identical architecture and preprocessing

RES = os.path.join(HERE, "results")
FIGS = os.path.join(HERE, "figures")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DELTA = 1e-5
BATCH = 256
LR = 0.1
MOMENTUM = 0.9
WD = 0.0
EPOCHS = 15
SEED = 42
CLIP_NORMS = [1.0, 5.0]  # measured per-example norms ~5-6 after 2 epochs: C=1 clips everything, C=5 clips ~20%
NOISE_MULTIPLIERS = [0.4, 0.8, 1.2]


# --------------------------------------------------------------------------------------
# Manual DP-SGD machinery
# --------------------------------------------------------------------------------------
def per_sample_gradients(model, x, y):
    """Return a dict param_name -> tensor[B, *param_shape] of per-example gradients."""
    params = {k: v for k, v in model.named_parameters()}
    buffers = {k: v for k, v in model.named_buffers()}

    def loss_one(p, b, xi, yi):
        out = functional_call(model, (p, b), (xi.unsqueeze(0),))
        return F.cross_entropy(out, yi.unsqueeze(0), reduction="mean")

    return vmap(grad(loss_one), in_dims=(None, None, 0, 0))(params, buffers, x, y)


def dp_step(model, opt, x, y, clip, sigma):
    """One DP-SGD update. Returns (mean gradient norm, clip fraction)."""
    grads = per_sample_gradients(model, x, y)
    B = x.size(0)

    # Per-example L2 norm over all parameters
    sq = None
    for g in grads.values():
        s = (g.reshape(B, -1) ** 2).sum(1)
        sq = s if sq is None else sq + s
    norms = torch.sqrt(sq + 1e-12)

    # Clip: factor = min(1, C / ||g_i||)
    factors = torch.clamp(clip / (norms + 1e-12), max=1.0)
    clip_frac = float((factors < 1).float().mean())

    for name, p in model.named_parameters():
        g = grads[name].reshape(B, -1)
        summed = (g * factors.unsqueeze(1)).sum(0).reshape(p.shape)
        # Gaussian mechanism: N(0, sigma^2 C^2 I) on the *sum*, then divide by B
        noise = torch.randn_like(summed) * (sigma * clip)
        p.grad = (summed + noise) / B

    opt.step()
    opt.zero_grad(set_to_none=True)
    return float(norms.mean()), clip_frac


def plain_step(model, opt, x, y):
    opt.zero_grad()
    F.cross_entropy(model(x), y).backward()
    opt.step()


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    corr, tot, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss_sum += F.cross_entropy(out, y, reduction="sum").item()
        corr += (out.argmax(1) == y).sum().item()
        tot += y.size(0)
    return corr / tot, loss_sum / tot


def epsilon_at(sigma, q, steps, delta=DELTA):
    """RDP -> (eps, delta) conversion for the subsampled Gaussian mechanism."""
    from opacus.accountants import RDPAccountant
    acc = RDPAccountant()
    for _ in range(steps):
        acc.step(noise_multiplier=sigma, sample_rate=q)
    return float(acc.get_epsilon(delta=delta))


def eps_via_prv(sigma, q, steps, delta=DELTA):
    from opacus.accountants import PRVAccountant
    acc = PRVAccountant()
    for _ in range(steps):
        acc.step(noise_multiplier=sigma, sample_rate=q)
    return float(acc.get_epsilon(delta=delta))


# --------------------------------------------------------------------------------------
def train_run(train_set, test_set, cfg, seed=SEED):
    """Train one model. cfg["sigma"]=0 -> non-private baseline (no clipping, no noise)."""
    torch.manual_seed(seed)
    model = SmallCNN().to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=cfg.get("lr", LR),
                          momentum=MOMENTUM, weight_decay=WD)
    train_loader = DataLoader(train_set, batch_size=cfg.get("batch", BATCH), shuffle=True,
                              generator=torch.Generator().manual_seed(seed), drop_last=True)
    test_loader = DataLoader(test_set, batch_size=512)
    steps_per_epoch = len(train_loader)
    curve, norms, fracs = [], [], []
    t0 = time.time()
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if cfg["sigma"] > 0:
                n, cf = dp_step(model, opt, x, y, cfg["clip"], cfg["sigma"])
                norms.append(n); fracs.append(cf)
            else:
                plain_step(model, opt, x, y)
        acc, loss = evaluate(model, test_loader)
        curve.append({"epoch": ep, "test_acc": acc, "test_loss": loss,
                      "elapsed_s": time.time() - t0})
        print(f"  [{cfg['name']}] epoch {ep:2d} acc={acc:.4f} loss={loss:.4f}", flush=True)
    steps = steps_per_epoch * cfg["epochs"]
    eps = epsilon_at(cfg["sigma"], cfg.get("batch", BATCH) / len(train_set), steps) if cfg["sigma"] > 0 else float("inf")
    eps_prv = eps_via_prv(cfg["sigma"], cfg.get("batch", BATCH) / len(train_set), steps) if cfg["sigma"] > 0 else float("inf")
    return model, curve, {"epsilon_rdp": eps, "epsilon_prv": eps_prv, "steps": steps,
                          "mean_grad_norm": float(np.mean(norms)) if norms else None,
                          "mean_clip_fraction": float(np.mean(fracs)) if fracs else None,
                          "train_time_s": time.time() - t0}


# --------------------------------------------------------------------------------------
# Membership inference
# --------------------------------------------------------------------------------------
@torch.no_grad()
def example_features(model, dataset, batch=512):
    """Per-example loss, max confidence, entropy and correctness."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch)
    L, C, E, Y = [], [], [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = F.cross_entropy(out, y, reduction="none")
        p = torch.softmax(out, dim=1)
        conf = p.max(1).values
        ent = -(p * torch.log(p + 1e-12)).sum(1)
        L.append(loss.cpu()); C.append(conf.cpu()); E.append(ent.cpu())
        Y.append((out.argmax(1) == y).float().cpu())
    return {"loss": torch.cat(L).numpy(), "conf": torch.cat(C).numpy(),
            "entropy": torch.cat(E).numpy(), "correct": torch.cat(Y).numpy()}


def attack_scores(feats, kind):
    if kind == "loss":
        return -feats["loss"]          # lower loss -> more likely a member
    if kind == "confidence":
        return feats["conf"]
    if kind == "entropy":
        return -feats["entropy"]
    raise ValueError(kind)


def tpr_at_fpr(member_scores, nonmember_scores, fpr_target):
    y = np.r_[np.ones(len(member_scores)), np.zeros(len(nonmember_scores))]
    s = np.r_[member_scores, nonmember_scores]
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.interp(fpr_target, fpr, tpr))


def attack_report(member_scores, nonmember_scores):
    y = np.r_[np.ones(len(member_scores)), np.zeros(len(nonmember_scores))]
    s = np.r_[member_scores, nonmember_scores]
    auc = roc_auc_score(y, s)
    return {"auc": float(auc), "tpr@1%fpr": tpr_at_fpr(member_scores, nonmember_scores, 0.01),
            "tpr@10%fpr": tpr_at_fpr(member_scores, nonmember_scores, 0.10),
            "advantage@10%": tpr_at_fpr(member_scores, nonmember_scores, 0.10) - 0.10}


def train_shadows(train_set, n_shadows=4, epochs=8, seed=0):
    """Non-private shadow models on random halves; returns features labelled member/not."""
    rng = np.random.default_rng(seed)
    feats_in, feats_out = [], []
    n = len(train_set)
    for s in range(n_shadows):
        perm = rng.permutation(n)
        members = perm[: n // 2]
        nonmembers = perm[n // 2:]
        m = SmallCNN().to(DEVICE)
        opt = torch.optim.SGD(m.parameters(), lr=LR, momentum=MOMENTUM)
        loader = DataLoader(Subset(train_set, members.tolist()), batch_size=BATCH, shuffle=True,
                            generator=torch.Generator().manual_seed(seed + s), drop_last=True)
        for _ in range(epochs):
            m.train()
            for x, y in loader:
                plain_step(m, opt, x.to(DEVICE), y.to(DEVICE))
        feats_in.append(example_features(m, Subset(train_set, members.tolist())))
        feats_out.append(example_features(m, Subset(train_set, nonmembers.tolist())))
        print(f"  shadow {s + 1}/{n_shadows} trained", flush=True)
    return feats_in, feats_out


def shadow_attack(shadow_in, shadow_out, target_member, target_nonmember):
    """Logistic-regression attack trained on shadow member/non-member features."""
    def stack(fs):
        """Stack (loss, confidence, entropy, correctness) features from several datasets."""
        return np.vstack([np.column_stack([f["loss"], f["conf"], f["entropy"], f["correct"]])
                          for f in fs])
    Xm, Xn = stack(shadow_in), stack(shadow_out)
    X = np.vstack([Xm, Xn]); y = np.r_[np.ones(len(Xm)), np.zeros(len(Xn))]
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=1000).fit(sc.transform(X), y)
    sm = clf.decision_function(sc.transform(np.column_stack(
        [target_member["loss"], target_member["conf"], target_member["entropy"], target_member["correct"]])))
    sn = clf.decision_function(sc.transform(np.column_stack(
        [target_nonmember["loss"], target_nonmember["conf"], target_nonmember["entropy"], target_nonmember["correct"]])))
    return sm, sn


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer epochs, for a smoke test")
    args = ap.parse_args()
    global EPOCHS
    if args.quick:
        EPOCHS = 3

    train_set, test_set, _, _ = load_data()
    print(f"train={len(train_set)} test={len(test_set)} device={DEVICE}")

    configs = [{"name": "baseline_nonprivate", "sigma": 0.0, "clip": None, "epochs": EPOCHS}]
    for C in CLIP_NORMS:
        for sigma in NOISE_MULTIPLIERS:
            configs.append({"name": f"C{C}_sigma{sigma}", "sigma": sigma, "clip": C, "epochs": EPOCHS})
    # Ablations: training duration and dataset size at the middle privacy setting.
    configs += [
        {"name": "ablation_epochs5_C1.0_sigma0.8", "sigma": 0.8, "clip": 1.0, "epochs": 5},
        {"name": "ablation_epochs30_C1.0_sigma0.8", "sigma": 0.8, "clip": 1.0, "epochs": 30},
        {"name": "ablation_n6k_C1.0_sigma0.8", "sigma": 0.8, "clip": 1.0, "epochs": EPOCHS,
         "subset": 6000},
    ]

    results, runs = [], []
    models = {}
    for cfg in configs:
        cfg = dict(cfg)
        tr = train_set
        if cfg.get("subset"):
            idx = np.random.default_rng(SEED).permutation(len(train_set))[: cfg["subset"]].tolist()
            tr = Subset(train_set, idx)
        print(f"\n=== {cfg['name']} (n={len(tr)}) ===", flush=True)
        model, curve, info = train_run(tr, test_set, cfg)
        acc, loss = curve[-1]["test_acc"], curve[-1]["test_loss"]
        row = {"config": cfg["name"], "sigma": cfg["sigma"], "clip": cfg["clip"],
               "epochs": cfg["epochs"], "n_train": len(tr), "test_acc": acc,
               "test_loss": loss, "final_train_loss_curve": curve[-1]["test_loss"],
               **info}
        runs.append(row)
        results.append({"cfg": cfg, "curve": curve, "info": info, "acc": acc})
        models[cfg["name"]] = model
        for c in curve:
            c.update({"config": cfg["name"], "sigma": cfg["sigma"], "clip": cfg["clip"]})
        torch.save(model.state_dict(), os.path.join(RES, f"model_{cfg['name']}.pt"))
        # Persist after every configuration so a failure later cannot lose completed runs.
        import pandas as _pd
        _pd.DataFrame(runs).to_csv(os.path.join(RES, "dp_runs.csv"), index=False)
        _pd.DataFrame([{"config": rr["cfg"]["name"], "epoch": c["epoch"],
                        "test_acc": c["test_acc"], "test_loss": c["test_loss"]}
                       for rr in results for c in rr["curve"]]).to_csv(
            os.path.join(RES, "dp_curves.csv"), index=False)

    # ---------------- membership inference ----------------
    print("\n=== shadow models for MIA ===", flush=True)
    shadow_in, shadow_out = train_shadows(train_set, n_shadows=4, epochs=8)
    mia_rows = []
    for cfg_name, model in models.items():
        fm = example_features(model, train_set)
        fn = example_features(model, test_set)
        row = {"config": cfg_name}
        for kind in ("loss", "confidence", "entropy"):
            r = attack_report(attack_scores(fm, kind), attack_scores(fn, kind))
            for k, v in r.items():
                row[f"{kind}_{k}"] = v
        sm, sn = shadow_attack(shadow_in, shadow_out, fm, fn)
        r = attack_report(sm, sn)
        for k, v in r.items():
            row[f"shadow_{k}"] = v
        cfg = next(c for c in configs if c["name"] == cfg_name)
        row["epsilon"] = epsilon_at(cfg["sigma"], cfg.get("batch", BATCH) / len(train_set),
                                    (len(train_set) if not cfg.get("subset") else cfg["subset"]) // BATCH * cfg["epochs"]) if cfg["sigma"] > 0 else float("inf")
        row["test_acc"] = next(r0["test_acc"] for r0 in runs if r0["config"] == cfg_name)
        mia_rows.append(row)
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items() if "auc" in k or k in ("config", "epsilon")}, flush=True)

    import pandas as pd
    pd.DataFrame(runs).to_csv(os.path.join(RES, "dp_runs.csv"), index=False)
    pd.DataFrame(mia_rows).to_csv(os.path.join(RES, "mia.csv"), index=False)
    curve_rows = []
    for r in results:
        for c in r["curve"]:
            curve_rows.append({"config": r["cfg"]["name"], "epoch": c["epoch"],
                               "test_acc": c["test_acc"], "test_loss": c["test_loss"]})
    pd.DataFrame(curve_rows).to_csv(os.path.join(RES, "dp_curves.csv"), index=False)
    with open(os.path.join(RES, "dp_results.json"), "w") as f:
        json.dump({"runs": runs, "mia": mia_rows, "delta": DELTA,
                   "hyper": {"batch": BATCH, "lr": LR, "momentum": MOMENTUM, "epochs": EPOCHS,
                             "clip_norms": CLIP_NORMS, "noise_multipliers": NOISE_MULTIPLIERS,
                             "dataset": "Fashion-MNIST balanced subset", "n_train": len(train_set),
                             "n_test": len(test_set)}}, f, indent=2, default=float)

    # ---------------- figures ----------------
    df = pd.DataFrame(runs)
    dp = df[df.sigma > 0].drop_duplicates("epsilon_rdp")
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for C, g in dp.groupby("clip"):
        g = g.sort_values("epsilon_rdp")
        ax.plot(g["epsilon_rdp"], g["test_acc"], "o-", label=f"DP-SGD C={C}")
    ax.axhline(df[df.sigma == 0]["test_acc"].iloc[0], color="k", ls="--", label="non-private")
    ax.set_xscale("log"); ax.set_xlabel("privacy budget epsilon (delta=1e-5)")
    ax.set_ylabel("test accuracy"); ax.set_title("Privacy-utility trade-off")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_q3_acc_vs_eps.png"), dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, g in pd.read_csv(os.path.join(RES, "dp_curves.csv")).groupby("config"):
        ax.plot(g["epoch"], g["test_acc"], label=name, lw=1.4)
    ax.set_xlabel("epoch"); ax.set_ylabel("test accuracy"); ax.set_title("Convergence under different noise levels")
    ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_q3_curves.png"), dpi=200); plt.close(fig)

    m = pd.DataFrame(mia_rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    dpm = m[np.isfinite(m.epsilon)].sort_values("epsilon")
    axes[0].plot(dpm["epsilon"], dpm["loss_auc"], "o-", label="loss attack")
    axes[0].plot(dpm["epsilon"], dpm["shadow_auc"], "s-", label="shadow-model attack")
    axes[0].axhline(0.5, color="k", ls=":", label="random guessing")
    axes[0].set_xscale("log"); axes[0].set_xlabel("epsilon"); axes[0].set_ylabel("MIA ROC-AUC")
    axes[0].set_title("Privacy leakage vs epsilon"); axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)
    axes[1].plot(dpm["epsilon"], dpm["shadow_advantage@10%"], "s-", color="firebrick")
    axes[1].axhline(0, color="k", ls=":")
    axes[1].set_xscale("log"); axes[1].set_xlabel("epsilon")
    axes[1].set_ylabel("attack advantage @10% FPR")
    axes[1].set_title("Membership advantage"); axes[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_q3_mia.png"), dpi=200); plt.close(fig)

    print("\n", df[["config", "test_acc", "epsilon_rdp", "epsilon_prv", "test_loss"]].to_string(index=False))


if __name__ == "__main__":
    main()
