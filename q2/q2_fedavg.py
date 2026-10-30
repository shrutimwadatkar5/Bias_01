#!/usr/bin/env python3
"""
Q2 - Federated Averaging (FedAvg) versus centralised CNN training on Fashion-MNIST.

What is implemented from scratch
--------------------------------
* A small, fixed CNN (no BatchNorm: BatchNorm's batch statistics are not compatible
  with per-example gradient clipping used in Q3, so the same architecture is reused
  there for comparability).
* Client simulation: each client owns a disjoint shard of the *same* training set.
* FedAvg: for each communication round every client performs E local epochs of SGD on
  its shard; the server forms the example-weighted average of the client parameters
  (McMahan et al., 2017).  No client data ever leaves the client object.
* Centralised baseline: plain mini-batch SGD on the union of the same examples with
  the *same* architecture, preprocessing, optimiser and total number of sample passes.
* Partitions: IID, Dirichlet(0.5), Dirichlet(0.1), and a 2-class-per-client label skew,
  which lets us measure how label heterogeneity changes convergence.

Fairness of the comparison
--------------------------
Budget is matched in sample passes: FedAvg runs R rounds x E local epochs, so the
centralised model runs R x E x (1/num_clients) ... careful: FedAvg touches every
example E times per round, i.e. R*E passes over the union; the centralised model
therefore runs R*E epochs on the union.  Both use identical batch size, learning rate,
momentum, weight decay and initial weights (same seed).

Outputs: results/*.json / *.csv and figures/*.png
"""
import copy
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIGS = os.path.join(HERE, "figures")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)

DATA = os.path.expanduser("~/data")
SEED = 42
N_TRAIN, N_TEST = 12000, 2000          # balanced subset for a tractable, reproducible run
N_CLIENTS = 5
ROUNDS = 20
LOCAL_EPOCHS = 2                        # -> centralised sees 40 epochs over the union
BATCH = 64
LR = 0.01
MOMENTUM = 0.9
WD = 5e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------------------
class SmallCNN(nn.Module):
    """~0.42M parameters. Two conv blocks (no BatchNorm, see Q3) + two FC layers."""

    def __init__(self, n_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)   # 28 -> 14
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)   # 14 -> 7
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def load_data():
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    tr = datasets.FashionMNIST(DATA, train=True, download=True, transform=tfm)
    te = datasets.FashionMNIST(DATA, train=False, download=True, transform=tfm)
    g = torch.Generator().manual_seed(SEED)
    # Stratified balanced subsets: 1200 training / 200 per class for the test set.
    tr_idx, te_idx = [], []
    targets = np.array(tr.targets)
    for c in range(10):
        idx = np.where(targets == c)[0]
        tr_idx.extend(np.random.default_rng(SEED + c).permutation(idx)[:N_TRAIN // 10].tolist())
    te_targets = np.array(te.targets)
    for c in range(10):
        idx = np.where(te_targets == c)[0]
        te_idx.extend(np.random.default_rng(SEED + 100 + c).permutation(idx)[:N_TEST // 10].tolist())
    return Subset(tr, tr_idx), Subset(te, te_idx), tr_idx, te_idx


def labels_of(subset):
    return np.array([subset.dataset.targets[i] for i in subset.indices])


# --------------------------------------------------------------------------------------
# Client partitions
# --------------------------------------------------------------------------------------
def partition_iid(labels, n_clients, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(labels))
    return np.array_split(idx, n_clients)


def partition_dirichlet(labels, n_clients, alpha, seed=SEED):
    """Class-wise Dirichlet allocation: low alpha -> each client sees few classes."""
    rng = np.random.default_rng(seed)
    by_class = {c: np.where(labels == c)[0] for c in np.unique(labels)}
    shards = [[] for _ in range(n_clients)]
    for c, idx in by_class.items():
        rng.shuffle(idx)
        props = rng.dirichlet([alpha] * n_clients)
        cuts = (np.cumsum(props)[:-1] * len(idx)).astype(int)
        for k, part in enumerate(np.split(idx, cuts)):
            shards[k].extend(part.tolist())
    return [np.array(sorted(s)) for s in shards]


def partition_label_skew(labels, n_clients, seed=SEED):
    """Each client gets exactly two classes (0-1, 2-3, ...), equal number of examples."""
    shards = [[] for _ in range(n_clients)]
    classes = np.unique(labels)
    per_client = len(classes) // n_clients
    rng = np.random.default_rng(seed)
    for k in range(n_clients):
        cs = classes[k * per_client:(k + 1) * per_client]
        idx = np.concatenate([np.where(labels == c)[0] for c in cs])
        rng.shuffle(idx)
        shards[k] = idx[:len(labels) // n_clients]
    return shards


# --------------------------------------------------------------------------------------
# Training / evaluation
# --------------------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss_sum += F.cross_entropy(out, y, reduction="sum").item()
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total, loss_sum / total


@torch.no_grad()
def per_client_accuracy(model, test_set, shards, labels):
    """Accuracy of the global model on the test examples whose labels this client owns.

    This measures whether federation distributes performance unevenly across clients
    (a fairness angle that a single aggregate test accuracy hides).
    """
    model.eval()
    test_labels = labels_of(test_set)
    out = []
    for idx in shards:
        owned = np.unique(labels[idx])
        mask = np.isin(test_labels, owned)
        sub = Subset(test_set, np.where(mask)[0].tolist())
        acc, _ = evaluate(model, DataLoader(sub, batch_size=256))
        out.append(acc)
    return out


def local_train(model, dataset, epochs, seed):
    """Train one client copy for E epochs. Returns the updated state dict and #examples."""
    model = copy.deepcopy(model)
    model.train()
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=True,
                        generator=torch.Generator().manual_seed(seed), drop_last=False)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WD)
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}, len(dataset)


def fedavg(init_state, train_set, shards, test_set, labels, tag, seed=SEED):
    """Server-side weighted averaging of client parameters."""
    global_state = copy.deepcopy(init_state)
    global_model = SmallCNN().to(DEVICE)
    global_model.load_state_dict(global_state)

    test_loader = DataLoader(test_set, batch_size=256)
    curve, client_curves = [], []
    t0 = time.time()
    for r in range(1, ROUNDS + 1):
        client_states, sizes = [], []
        for k, idx in enumerate(shards):
            local_model = SmallCNN().to(DEVICE)
            local_model.load_state_dict(global_state)          # broadcast global model
            state, n = local_train(local_model, Subset(train_set, idx.tolist()),
                                   LOCAL_EPOCHS, seed + 100 * r + k)
            client_states.append(state); sizes.append(n)

        # Weighted average: theta = sum_k (n_k / n) theta_k
        total = sum(sizes)
        avg = {}
        for key in client_states[0]:
            stack = torch.stack([cs[key].float() * (n / total)
                                 for cs, n in zip(client_states, sizes)])
            avg[key] = stack.sum(0)
        global_state = avg
        global_model.load_state_dict(global_state)
        acc, loss = evaluate(global_model, test_loader)
        curve.append({"round": r, "test_acc": acc, "test_loss": loss,
                      "seconds": time.time() - t0})
        print(f"[{tag}] round {r:2d}  acc={acc:.4f}  loss={loss:.4f}", flush=True)
    client_curves = per_client_accuracy(global_model, test_set, shards, labels)
    return curve, client_curves, global_state


def centralized(init_state, train_set, test_set, epochs=ROUNDS * LOCAL_EPOCHS, seed=SEED):
    model = SmallCNN().to(DEVICE)
    model.load_state_dict(init_state)
    loader = DataLoader(train_set, batch_size=BATCH, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WD)
    test_loader = DataLoader(test_set, batch_size=256)
    curve = []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        acc, loss = evaluate(model, test_loader)
        # Record one point every LOCAL_EPOCHS epochs so the x-axis is comparable:
        # one FedAvg round = LOCAL_EPOCHS centralised epochs = one pass over the union
        # by every client.
        if ep % LOCAL_EPOCHS == 0:
            curve.append({"round": ep // LOCAL_EPOCHS, "epoch": ep, "test_acc": acc,
                          "test_loss": loss, "seconds": time.time() - t0})
            print(f"[central] epoch {ep:2d}  acc={acc:.4f}  loss={loss:.4f}", flush=True)
    return curve


# --------------------------------------------------------------------------------------
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    train_set, test_set, tr_idx, te_idx = load_data()
    labels = labels_of(train_set)
    print(f"train={len(train_set)} test={len(test_set)} params={n_params(SmallCNN())} device={DEVICE}")

    partitions = {
        "iid": partition_iid(labels, N_CLIENTS),
        "dirichlet_a0.5": partition_dirichlet(labels, N_CLIENTS, 0.5),
        "dirichlet_a0.1": partition_dirichlet(labels, N_CLIENTS, 0.1),
        "label_skew_2class": partition_label_skew(labels, N_CLIENTS),
    }

    # Identical initial weights for every condition.
    torch.manual_seed(SEED)
    init_state = copy.deepcopy(SmallCNN().state_dict())

    # Persist the partitions so the exact experiment can be reproduced / inspected.
    np.savez(os.path.join(RES, "partitions.npz"),
             **{f"{k}_{i}": v for k, sh in partitions.items() for i, v in enumerate(sh)})

    results = {"config": {"n_train": len(train_set), "n_test": len(test_set), "clients": N_CLIENTS,
                          "rounds": ROUNDS, "local_epochs": LOCAL_EPOCHS, "batch": BATCH, "lr": LR,
                          "momentum": MOMENTUM, "weight_decay": WD, "params": n_params(SmallCNN()),
                          "central_epochs": ROUNDS * LOCAL_EPOCHS, "device": DEVICE,
                          "dataset": "Fashion-MNIST (balanced subset)", "date": time.strftime("%Y-%m-%d")},
               "runs": {}}

    curves = []
    client_rows = []
    for tag, shards in partitions.items():
        curve, cacc, state = fedavg(init_state, train_set, shards, test_set, labels, tag)
        for row in curve:
            curves.append({"method": "fedavg", "partition": tag, **row})
        for k, a in enumerate(cacc):
            client_rows.append({"method": "fedavg", "partition": tag, "client": k, "test_acc": a,
                                "n_examples": int(len(shards[k])),
                                "classes": ",".join(map(str, np.unique(labels[shards[k]]).tolist()))})
        results["runs"][tag] = {
            "final_acc": curve[-1]["test_acc"], "best_acc": max(c["test_acc"] for c in curve),
            "final_loss": curve[-1]["test_loss"], "time_s": curve[-1]["seconds"],
            "rounds_to_90pct_of_final": next((c["round"] for c in curve
                                              if c["test_acc"] >= 0.9 * curve[-1]["test_acc"]), None),
            "client_acc_mean": float(np.mean(cacc)), "client_acc_std": float(np.std(cacc)),
            "client_acc_min": float(np.min(cacc)), "client_acc_max": float(np.max(cacc)),
            "comm_mb_total": 2 * n_params(SmallCNN()) * 4 * N_CLIENTS * ROUNDS / 1e6,
        }

    ccurve = centralized(init_state, train_set, test_set)
    for row in ccurve:
        curves.append({"method": "centralized", "partition": "centralized", **row})
    results["runs"]["centralized"] = {
        "final_acc": ccurve[-1]["test_acc"], "best_acc": max(c["test_acc"] for c in ccurve),
        "final_loss": ccurve[-1]["test_loss"], "time_s": ccurve[-1]["seconds"],
        "comm_mb_total": len(train_set) * 28 * 28 * 4 / 1e6,  # one upload of the raw data
    }
    results["gaps_vs_centralized"] = {
        tag: results["runs"]["centralized"]["final_acc"] - results["runs"][tag]["final_acc"]
        for tag in partitions}

    pd.DataFrame(curves).to_csv(os.path.join(RES, "curves.csv"), index=False)
    pd.DataFrame(client_rows).to_csv(os.path.join(RES, "client_acc.csv"), index=False)
    with open(os.path.join(RES, "fedavg_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ---------------- figures ----------------
    fig, ax = plt.subplots(figsize=(7, 4.4))
    cdf = pd.read_csv(os.path.join(RES, "curves.csv"))
    for tag, g in cdf[cdf.method == "fedavg"].groupby("partition"):
        ax.plot(g["round"], g["test_acc"], marker="o", ms=3, label=f"FedAvg {tag}")
    g = cdf[cdf.method == "centralized"]
    ax.plot(g["round"], g["test_acc"], "k--", lw=2, label="centralised (same passes)")
    ax.set_xlabel("communication round (= 2 centralised epochs)")
    ax.set_ylabel("test accuracy")
    ax.set_title("FedAvg vs centralised training under different client partitions")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_q2_curves.png"), dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    for tag, g in cdf[cdf.method == "fedavg"].groupby("partition"):
        ax.plot(g["round"], g["test_loss"], label=f"FedAvg {tag}")
    ax.set_xlabel("round"); ax.set_ylabel("test loss"); ax.set_title("Test loss convergence")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_q2_loss.png"), dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    cadf = pd.DataFrame(client_rows)
    width = 0.2
    for k, (tag, g) in enumerate(cadf.groupby("partition")):
        ax.bar(np.arange(len(g)) + k * width, g["test_acc"], width, label=tag)
    ax.set_xticks(np.arange(N_CLIENTS) + 1.5 * width, [f"client {i}" for i in range(N_CLIENTS)])
    ax.set_ylabel("accuracy on owned classes"); ax.set_ylim(0, 1)
    ax.set_title("Per-client accuracy of the final global model")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_q2_client_acc.png"), dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    mat = np.zeros((N_CLIENTS, 10))
    for k, idx in enumerate(partitions["dirichlet_a0.5"]):
        for c in range(10):
            mat[k, c] = np.sum(labels[idx] == c)
    im = ax.imshow(mat, cmap="Blues", aspect="auto")
    ax.set_xlabel("class"); ax.set_ylabel("client")
    ax.set_title("Client label distribution, Dirichlet(0.5)")
    fig.colorbar(im, ax=ax, label="#examples")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_q2_partition.png"), dpi=200); plt.close(fig)

    print(json.dumps(results["runs"], indent=2))
    print(json.dumps(results["gaps_vs_centralized"], indent=2))


if __name__ == "__main__":
    import pandas as pd
    main()
