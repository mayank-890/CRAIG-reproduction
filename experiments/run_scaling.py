#!/usr/bin/env python3
"""Scaling studies.

seq:  time-series sequence length L in {50,100,200,500,1000} (fixed N) -> results/raw/scaling_seq_*.csv
size: MNIST training-set size N in {1000,2000,5000,10000,20000} (fixed ratio) -> results/raw/scaling_size_*.csv
"""
import os, sys, argparse, logging
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training.reproducibility import get_device
from training.logger import ExperimentLogger
from models.logistic_regression import LogisticRegression
from datasets.mnist import get_mnist_datasets
from experiments.common import run_benchmark
from experiments.run_timeseries import run as run_ts

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--study", choices=["seq", "size", "both"], default="both")
    p.add_argument("--seq-lengths", type=int, nargs="+", default=[50, 100, 200, 500, 1000])
    p.add_argument("--sizes", type=int, nargs="+", default=[1000, 2000, 5000, 10000, 20000])
    p.add_argument("--ratio", type=float, default=0.1)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123])
    p.add_argument("--epochs", type=int, default=10)
    a = p.parse_args()
    methods = ["FULL", "RANDOM", "CRAIG"]
    if a.study in ("seq", "both"):
        lg = ExperimentLogger("./results", "scaling_seq")
        for L in a.seq_lengths:
            run_ts(2000, L, a.seeds, [a.ratio], a.epochs, methods, 4, lg, tag="scaling_seq")
        print("CSV:", lg.save({"cli_args": vars(a), "study": "seq"}))
    if a.study in ("size", "both"):
        lg = ExperimentLogger("./results", "scaling_size")
        for n in a.sizes:
            tr, te = get_mnist_datasets("./data", flatten=True, max_train_samples=n)
            run_benchmark(dataset="mnist", model_name="logistic_regression", model_fn=lambda: LogisticRegression(784, 10),
                          train_ds=tr, test_ds=te, seeds=a.seeds, ratios=[a.ratio], methods=methods,
                          epochs=a.epochs, batch_size=128, opt_cfg=dict(lr=0.01, momentum=0.9, weight_decay=1e-4),
                          device=get_device("auto"), exp_logger=lg, extra=dict(experiment_tag="scaling_size"))
        print("CSV:", lg.save({"cli_args": vars(a), "study": "size"}))


if __name__ == "__main__":
    main()
