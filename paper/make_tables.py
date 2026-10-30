#!/usr/bin/env python3
"""Generate LaTeX table bodies for paper.tex directly from results/raw and results/processed CSVs.
Every number in the tables is computed here; nothing is typed by hand."""
import os
import pandas as pd
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
T = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables")
raw = lambda f: pd.read_csv(f"{R}/raw/{f}.csv")
proc = lambda f: pd.read_csv(f"{R}/processed/{f}.csv")

def agg(d, keys):
    return d.groupby(keys).agg(n=("seed", "count"), acc=("accuracy", "mean"), acc_sd=("accuracy", "std"),
                               sel=("selection_time", "mean"), sel_sd=("selection_time", "std"),
                               grad=("gradient_computation_time", "mean"), sim=("similarity_computation_time", "mean"),
                               fl=("facility_location_time", "mean"), tr=("training_time", "mean"),
                               tr_sd=("training_time", "std"), tot=("total_time", "mean"),
                               gerr=("gradient_error", "mean"), gerr_sd=("gradient_error", "std"),
                               cos=("gradient_cosine_similarity", "mean"), k=("subset_size", "mean"))

pm = lambda m, s, d=2: f"${m:.{d}f}\\pm{s:.{d}f}$"
pct = lambda m, s: pm(100 * m, 100 * s)
def w(name, rows):
    open(f"{T}/{name}.tex", "w").write("\n".join(rows) + "\n")

def acc_table(stem, name, ratios):
    g = agg(raw(stem), ["method", "subset_ratio"])
    rows = []
    for r in ratios:
        c, u, rn = g.loc[("CRAIG", r)], g.loc[("CRAIG_UNWEIGHTED", r)], g.loc[("RANDOM", r)]
        f = lambda x: pct(x.acc, x.acc_sd)
        rows.append(f"{int(round(100*r))}\\% & {int(round(rn.k))} & {f(c)} & {f(u)} & {f(rn)} \\\\")
    fu = g.loc[("FULL", 1.0)]
    rows.append(r"\midrule")
    rows.append(f"100\\% (FULL) & {int(fu.k)} & \\multicolumn{{3}}{{c}}{{{pct(fu.acc, fu.acc_sd)}}} \\\\")
    w(name, rows)
    return g

def gerr_table(g, name, ratios):
    rows = []
    for r in ratios:
        c, u, rn = g.loc[("CRAIG", r)], g.loc[("CRAIG_UNWEIGHTED", r)], g.loc[("RANDOM", r)]
        rows.append(f"{int(round(100*r))}\\% & {pm(c.gerr, c.gerr_sd, 3)} & {pm(u.gerr, u.gerr_sd, 3)} & {pm(rn.gerr, rn.gerr_sd, 3)} & {c.cos:.4f} & {rn.cos:.4f} \\\\")
    w(name, rows)

def time_table(g, be, name, ratios):
    rows = []
    for r in ratios:
        c = g.loc[("CRAIG", r)]
        b = be[(be.method == "CRAIG") & (be.subset_ratio == r)].iloc[0]
        other = c.sel - c.grad - c.sim - c.fl
        rows.append(f"{int(round(100*r))}\\% & {c.grad:.2f} & {c.sim:.2f} & {c.fl:.2f} & {other:.2f} & {pm(c.sel, c.sel_sd)} & {c.tr:.2f} & {c.tot:.2f} & {b.training_only_speedup:.2f} & {b.end_to_end_speedup:.2f} \\\\")
    fu = g.loc[("FULL", 1.0)]
    rows.append(r"\midrule")
    rows.append(f"FULL & -- & -- & -- & -- & -- & {pm(fu.tr, fu.tr_sd)} & {fu.tot:.2f} & 1.00 & 1.00 \\\\")
    w(name, rows)

# MNIST
mr = [0.05, 0.1, 0.2, 0.3, 0.5]
g = acc_table("mnist_2026_10_29_001", "mnist_acc", mr)
gerr_table(g, "mnist_gerr", mr)
time_table(g, proc("breakeven_mnist"), "mnist_time", mr)
# CIFAR-10 (reduced)
cr = [0.1, 0.2, 0.3, 0.5]
g = acc_table("cifar10_2026_10_29_001", "cifar_acc", cr)
be = proc("breakeven_cifar10")
rows = []
for r in cr:
    c, rn = g.loc[("CRAIG", r)], g.loc[("RANDOM", r)]
    b = be[(be.method == "CRAIG") & (be.subset_ratio == r)].iloc[0]
    rows.append(f"{int(round(100*r))}\\% & {pm(c.gerr, c.gerr_sd, 3)} & {pm(rn.gerr, rn.gerr_sd, 3)} & {c.grad:.2f} & {c.sel:.2f} & {c.tr:.2f} & {b.end_to_end_speedup:.2f} \\\\")
fu = g.loc[("FULL", 1.0)]
rows += [r"\midrule", f"FULL & -- & -- & -- & -- & {pm(fu.tr, fu.tr_sd)} & 1.00 \\\\"]
w("cifar_gerr_time", rows)
# Time series (hard, sigma=2.0) and easy (sigma=0.2)
g = acc_table("timeseries_2026_10_29_002", "ts_hard_acc", mr)
be = proc("breakeven_timeseries")
rows = []
for r in mr:
    c, rn = g.loc[("CRAIG", r)], g.loc[("RANDOM", r)]
    b = be[(be.method == "CRAIG") & (be.subset_ratio == r)].iloc[0]
    rows.append(f"{int(round(100*r))}\\% & {pm(c.gerr, c.gerr_sd, 3)} & {pm(rn.gerr, rn.gerr_sd, 3)} & {c.sel:.2f} & {c.tr:.2f} & {b.end_to_end_speedup:.2f} \\\\")
fu = g.loc[("FULL", 1.0)]
rows += [r"\midrule", f"FULL & -- & -- & -- & {pm(fu.tr, fu.tr_sd)} & 1.00 \\\\"]
w("ts_hard_gerr_time", rows)
ge = agg(raw("timeseries_2026_10_29_001"), ["method", "subset_ratio"])
print("easy TS acc min/max:", ge.acc.min(), ge.acc.max())
# Scaling: dataset size
g = agg(raw("scaling_size_2026_10_29_001"), ["total_train_samples", "method"])
be = proc("breakeven_scaling_size")
rows = []
for n in sorted(raw("scaling_size_2026_10_29_001").total_train_samples.unique()):
    c, rn, fu = g.loc[(n, "CRAIG")], g.loc[(n, "RANDOM")], g.loc[(n, "FULL")]
    b = be[(be.method == "CRAIG") & (be.total_train_samples == n)].iloc[0]
    rows.append(f"{n:,} & {c.grad:.3f} & {c.sim:.3f} & {c.fl:.3f} & {c.sel:.3f} & {fu.tr:.3f} & {b.end_to_end_speedup:.2f} & {100*c.acc:.2f} & {100*rn.acc:.2f} & {100*fu.acc:.2f} \\\\".replace(",", "{,}"))
w("scaling_size", rows)
# Scaling: sequence length
g = agg(raw("scaling_seq_2026_10_29_001"), ["seq_length", "method"])
be = proc("breakeven_scaling_seq")
rows = []
for L in sorted(raw("scaling_seq_2026_10_29_001").seq_length.unique()):
    c, rn, fu = g.loc[(L, "CRAIG")], g.loc[(L, "RANDOM")], g.loc[(L, "FULL")]
    b = be[(be.method == "CRAIG") & (be.seq_length == L)].iloc[0]
    rows.append(f"{L} & {c.grad:.3f} & {c.sim:.3f} & {c.fl:.3f} & {c.sel:.3f} & {fu.tr:.3f} & {b.end_to_end_speedup:.2f} & {pct(c.acc, c.acc_sd)} & {pct(rn.acc, rn.acc_sd)} & {pct(fu.acc, fu.acc_sd)} \\\\")
w("scaling_seq", rows)
# Paired (same-seed) accuracy differences, percentage points
def paired(stem, ratios):
    d = raw(stem)
    p = d.pivot_table(index=["subset_ratio", "seed"], columns="method", values="accuracy")
    out = []
    for r in ratios:
        grp = p.loc[r]
        cells = []
        for b in ["RANDOM", "CRAIG_UNWEIGHTED"]:
            diff = 100 * (grp["CRAIG"] - grp[b])
            cells.append(f"${diff.mean():+.2f}\\pm{diff.std():.2f}$ ({int((diff > 0).sum())}/{len(diff)})")
        out.append(cells)
    return out
rows = []
for name, stem, rs in [("MNIST", "mnist_2026_10_29_001", mr), ("CIFAR-10$^\\dagger$", "cifar10_2026_10_29_001", cr),
                       ("TS ($\\sigma{=}2$)", "timeseries_2026_10_29_002", mr)]:
    for i, (r, cells) in enumerate(zip(rs, paired(stem, rs))):
        rows.append(f"{name if i == 0 else ''} & {int(round(100*r))}\\% & {cells[0]} & {cells[1]} \\\\")
    rows.append(r"\midrule")
w("paired", rows[:-1])
print("ok")
