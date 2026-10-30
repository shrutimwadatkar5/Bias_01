#!/usr/bin/env python3
"""
Q3 supplementary run - deliberately overfit non-private baseline for the MIA.

Why: the 15-epoch baseline generalises almost as well as it fits, so a membership
inference attack has very little signal to exploit (AUC 0.56).  A model that is allowed
to overfit is the realistic worst case for privacy and is the condition in which the
protection offered by DP-SGD should be visible.  This script trains one non-private
model with 30 epochs, no weight decay and a higher learning rate, then runs exactly the
same attacks and appends the result to results/dp_runs.csv and results/mia.csv.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q3_dpsgd import (BATCH, DEVICE, RES, SEED, SmallCNN, attack_report,  # noqa: E402
                      attack_scores, example_features, load_data, shadow_attack,
                      train_shadows)

import torch.nn.functional as F
from torch.utils.data import DataLoader


def main():
    train_set, test_set, _, _ = load_data()
    cfg = {"name": "overfit_baseline_30ep", "sigma": 0.0, "clip": None, "epochs": 30, "lr": 0.15}

    torch.manual_seed(SEED)
    model = SmallCNN().to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9, weight_decay=0.0)
    dl = DataLoader(train_set, batch_size=BATCH, shuffle=True,
                    generator=torch.Generator().manual_seed(SEED), drop_last=True)
    te = DataLoader(test_set, batch_size=512)
    curve = []
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        for x, y in dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()
        model.eval()
        corr = tot = 0
        loss = 0.0
        with torch.no_grad():
            for x, y in te:
                x, y = x.to(DEVICE), y.to(DEVICE)
                o = model(x); loss += F.cross_entropy(o, y, reduction="sum").item()
                corr += (o.argmax(1) == y).sum().item(); tot += len(y)
        curve.append((ep, corr / tot, loss / tot))
        if ep % 5 == 0:
            print(f"epoch {ep} test_acc={corr / tot:.4f}", flush=True)

    acc, loss = curve[-1][1], curve[-1][2]
    tr_acc = None
    model.eval()
    with torch.no_grad():
        c = t = 0
        for x, y in DataLoader(train_set, batch_size=512):
            x, y = x.to(DEVICE), y.to(DEVICE)
            c += (model(x).argmax(1) == y).sum().item(); t += len(y)
        tr_acc = c / t
    print(f"overfit baseline: train_acc={tr_acc:.4f} test_acc={acc:.4f}")

    # identical attack suite as in the main script
    fm = example_features(model, train_set)
    fn = example_features(model, test_set)
    row = {"config": cfg["name"], "epsilon": float("inf"), "test_acc": acc, "test_loss": loss}
    for kind in ("loss", "confidence", "entropy"):
        for k, v in attack_report(attack_scores(fm, kind), attack_scores(fn, kind)).items():
            row[f"{kind}_{k}"] = v
    shadow_in, shadow_out = train_shadows(train_set, n_shadows=4, epochs=8)
    sm, sn = shadow_attack(shadow_in, shadow_out, fm, fn)
    for k, v in attack_report(sm, sn).items():
        row[f"shadow_{k}"] = v
    print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}, indent=1))

    runs = pd.read_csv(os.path.join(RES, "dp_runs.csv"))
    runs = runs[runs["config"] != cfg["name"]]
    runs = pd.concat([runs, pd.DataFrame([{
        "config": cfg["name"], "sigma": 0.0, "clip": np.nan, "epochs": cfg["epochs"],
        "n_train": len(train_set), "test_acc": acc, "test_loss": loss,
        "epsilon_rdp": float("inf"), "epsilon_prv": float("inf"), "steps": 690,
        "train_acc": tr_acc}])], ignore_index=True)
    runs.to_csv(os.path.join(RES, "dp_runs.csv"), index=False)

    mia = pd.read_csv(os.path.join(RES, "mia.csv"))
    mia = mia[mia["config"] != cfg["name"]]
    mia = pd.concat([mia, pd.DataFrame([row])], ignore_index=True)
    mia.to_csv(os.path.join(RES, "mia.csv"), index=False)
    torch.save(model.state_dict(), os.path.join(RES, "model_overfit_baseline.pt"))

    with open(os.path.join(RES, "overfit_curve.json"), "w") as f:
        json.dump(curve, f)
    print("appended overfit baseline to dp_runs.csv and mia.csv")


if __name__ == "__main__":
    main()
