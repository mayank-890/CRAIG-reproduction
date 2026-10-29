#!/usr/bin/env python3
"""Publication figures, loaded directly from results/raw CSVs (latest run per prefix)."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from analysis.runtime_analysis import latest_csv, breakeven

FIG = "results/figures"
COL = {"FULL": "k", "RANDOM": "tab:blue", "CRAIG": "tab:red", "CRAIG_UNWEIGHTED": "tab:orange"}


def _save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(FIG, name), dpi=150); plt.close(fig); print("wrote", name)


def acc_vs_ratio(df, ds):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for m in ["RANDOM", "CRAIG", "CRAIG_UNWEIGHTED"]:
        d = df[df.method == m].groupby("subset_ratio").accuracy.agg(["mean", "std"])
        if len(d):
            ax.errorbar(d.index * 100, d["mean"] * 100, yerr=d["std"].fillna(0) * 100, marker="o", capsize=3, label=m, color=COL[m])
    f = df[df.method == "FULL"].accuracy
    if len(f):
        ax.axhline(f.mean() * 100, color="k", ls="--", label=f"FULL ({f.mean()*100:.2f}%)")
    n = df.seed.nunique()
    ax.set(xlabel="Subset size (% of training set)", ylabel="Test accuracy (%)", title=f"{ds}: accuracy vs subset size ({n} seeds)")
    ax.legend(fontsize=8); ax.grid(alpha=.3); _save(fig, f"{ds}_accuracy_vs_ratio.png")


def time_breakdown(df, ds):
    d = df[df.method == "CRAIG"].groupby("subset_ratio")[["gradient_computation_time", "similarity_computation_time",
                                                           "facility_location_time", "training_time"]].mean()
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(5, 3.5))
    labels = [f"{r*100:g}%" for r in d.index]; bottom = 0
    for c, lab in zip(d.columns, ["gradients", "similarity", "facility location", "subset training"]):
        ax.bar(labels, d[c], bottom=bottom, label=lab); bottom = bottom + d[c]
    full = df[df.method == "FULL"].training_time.mean()
    ax.axhline(full, color="k", ls="--", label=f"FULL training ({full:.1f}s)")
    ax.set(xlabel="Subset size", ylabel="Wall-clock time (s)", title=f"{ds}: CRAIG time breakdown")
    ax.legend(fontsize=7); _save(fig, f"{ds}_craig_time_breakdown.png")


def speedup(df, ds):
    b = breakeven(df)
    if b.empty:
        return
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for m, d in b.groupby("method"):
        ax.plot(d.subset_ratio * 100, d.end_to_end_speedup, marker="o", color=COL[m], label=f"{m} end-to-end")
        if m == "CRAIG":
            ax.plot(d.subset_ratio * 100, d.training_only_speedup, marker="x", ls=":", color=COL[m], label="CRAIG training-only")
    ax.axhline(1, color="k", ls="--", label="break-even")
    ax.set_yscale("log"); ax.set(xlabel="Subset size (%)", ylabel="Speedup vs FULL training", title=f"{ds}: speedup / break-even")
    ax.legend(fontsize=7); ax.grid(alpha=.3); _save(fig, f"{ds}_speedup_breakeven.png")


def grad_error(df, ds):
    d = df[df.gradient_error.notna()]
    if d.empty:
        return
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.5))
    for m, g in d.groupby("method"):
        a = g.groupby("subset_ratio")[["gradient_error", "gradient_cosine_similarity"]].mean()
        axs[0].plot(a.index * 100, a.gradient_error, marker="o", color=COL[m], label=m)
        axs[1].plot(a.index * 100, a.gradient_cosine_similarity, marker="o", color=COL[m], label=m)
    axs[0].set(xlabel="Subset size (%)", ylabel="||G_full - G_S|| / ||G_full||", title=f"{ds}: relative gradient error")
    axs[1].set(xlabel="Subset size (%)", ylabel="cos(G_full, G_S)", title=f"{ds}: gradient cosine")
    for a in axs: a.legend(fontsize=7); a.grid(alpha=.3)
    _save(fig, f"{ds}_gradient_error.png")


def acc_vs_time(df, ds):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for m, g in df.groupby("method"):
        a = g.groupby("subset_ratio")[["total_time", "accuracy"]].mean()
        ax.plot(a.total_time, a.accuracy * 100, marker="o", ls="-" if len(a) > 1 else "", color=COL.get(m), label=m)
    ax.set_xscale("log"); ax.set(xlabel="Total wall-clock (selection + training, s)", ylabel="Test accuracy (%)",
                                 title=f"{ds}: accuracy vs total time")
    ax.legend(fontsize=7); ax.grid(alpha=.3); _save(fig, f"{ds}_accuracy_vs_total_time.png")


def scaling(df, key, ds, xlabel):
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.5))
    c = df[df.method == "CRAIG"].groupby(key)[["gradient_computation_time", "similarity_computation_time",
                                               "facility_location_time", "selection_time", "training_time"]].mean()
    for col in c.columns:
        axs[0].plot(c.index, c[col], marker="o", label=col.replace("_computation", "").replace("_", " "))
    f = df[df.method == "FULL"].groupby(key).training_time.mean()
    axs[0].plot(f.index, f, "k--o", label="FULL training")
    axs[0].set_xscale("log"); axs[0].set_yscale("log"); axs[0].set(xlabel=xlabel, ylabel="time (s)", title=f"{ds}: CRAIG cost scaling")
    axs[0].legend(fontsize=6)
    for m, g in df.groupby("method"):
        a = g.groupby(key).accuracy.mean()
        axs[1].plot(a.index, a * 100, marker="o", color=COL.get(m), label=m)
    axs[1].set_xscale("log"); axs[1].set(xlabel=xlabel, ylabel="Test accuracy (%)", title=f"{ds}: accuracy")
    axs[1].legend(fontsize=7)
    for a in axs: a.grid(alpha=.3)
    _save(fig, f"{ds}.png")


def main():
    os.makedirs(FIG, exist_ok=True)
    import glob
    for pre in ["mnist", "cifar10", "timeseries"]:
        for f in sorted(glob.glob(f"results/raw/{pre}_20*.csv")):
            df = pd.read_csv(f)
            if "CRAIG" not in set(df.method):
                print("skip (no CRAIG rows)", f); continue
            name = os.path.splitext(os.path.basename(f))[0]  # figures named after their source CSV
            for fn in (acc_vs_ratio, time_breakdown, speedup, grad_error, acc_vs_time):
                fn(df, name)
    f = latest_csv("scaling_size")
    if f: scaling(pd.read_csv(f), "total_train_samples", "scaling_dataset_size", "training set size N")
    f = latest_csv("scaling_seq")
    if f: scaling(pd.read_csv(f), "seq_length", "scaling_sequence_length", "sequence length L")


if __name__ == "__main__":
    main()
