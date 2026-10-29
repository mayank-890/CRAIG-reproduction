#!/usr/bin/env python3
"""Break-even analysis from raw CSVs: is T_selection + T_subset_training < T_full_training?

training_only_speedup = T_full_train / T_subset_train
end_to_end_speedup    = T_full_train / (T_selection + T_subset_train)   (selection overhead included)
Writes results/processed/summary_<prefix>.csv and breakeven_<prefix>.csv.
"""
import os, sys, glob, argparse
import pandas as pd

RAW = "results/raw"; OUT = "results/processed"


def latest_csv(prefix):
    files = sorted(f for f in glob.glob(os.path.join(RAW, f"{prefix}_20*.csv")))
    return files[-1] if files else None


def summarize(df, keys=("method", "subset_ratio")):
    g = df.groupby(list(keys))
    return g.agg(n_runs=("accuracy", "size"), acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
                 selection_time=("selection_time", "mean"), gradient_time=("gradient_computation_time", "mean"),
                 similarity_time=("similarity_computation_time", "mean"),
                 fl_time=("facility_location_time", "mean"), training_time=("training_time", "mean"),
                 total_time=("total_time", "mean"), grad_rel_error=("gradient_error", "mean"),
                 grad_cosine=("gradient_cosine_similarity", "mean")).reset_index()


def breakeven(df, group_extra=()):
    rows = []
    keys = list(group_extra)
    for gk, d in (df.groupby(keys) if keys else [((), df)]):
        full = d[d.method == "FULL"]
        if full.empty:
            continue
        t_full = full.training_time.mean(); a_full = full.accuracy.mean()
        for (m, r), s in d[d.method != "FULL"].groupby(["method", "subset_ratio"]):
            tsel, ttr = s.selection_time.mean(), s.training_time.mean()
            row = dict(zip(keys, gk if isinstance(gk, tuple) else (gk,)))
            row.update(method=m, subset_ratio=r, T_full_train=t_full, T_selection=tsel, T_subset_train=ttr,
                       T_total=tsel + ttr, training_only_speedup=t_full / ttr, end_to_end_speedup=t_full / (tsel + ttr),
                       breaks_even=bool(tsel + ttr < t_full), acc=s.accuracy.mean(), acc_full=a_full,
                       acc_gap_vs_full=s.accuracy.mean() - a_full)
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prefixes", nargs="+", default=["mnist", "cifar10", "timeseries", "scaling_size", "scaling_seq"])
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for pre in a.prefixes:
        f = latest_csv(pre)
        if not f:
            print(f"[skip] no CSV for {pre}"); continue
        df = pd.read_csv(f)
        extra = ("total_train_samples",) if pre == "scaling_size" else ("seq_length",) if pre == "scaling_seq" else ()
        s = summarize(df, extra + ("method", "subset_ratio")); b = breakeven(df, extra)
        s.to_csv(f"{OUT}/summary_{pre}.csv", index=False); b.to_csv(f"{OUT}/breakeven_{pre}.csv", index=False)
        print(f"\n=== {pre}  (source: {f}) ===")
        with pd.option_context("display.width", 250, "display.max_columns", 30):
            print(s.round(4).to_string(index=False)); print(b.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
