#!/usr/bin/env python3
"""Time-series classification with TimeSeriesCNN (1D CNN): FULL vs RANDOM vs CRAIG.

Data: the repo's synthetic generator (class-dependent sinusoid frequency + Gaussian noise).
Real UCR datasets (e.g. FordA) are not bundled; get_timeseries_datasets falls back to synthetic.
"""
import os, sys, argparse, logging
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from training.reproducibility import get_device
from training.logger import ExperimentLogger
from models.timeseries_cnn import TimeSeriesCNN
from datasets.timeseries import create_synthetic_timeseries
from experiments.common import run_benchmark

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)


def run(n_train, seq_length, seeds, ratios, epochs, methods, num_classes, lg, layers="all", tag="timeseries"):
    train_ds, test_ds = create_synthetic_timeseries(num_train=n_train, num_test=1000, seq_length=seq_length,
                                                    num_classes=num_classes, seed=0)
    run_benchmark(dataset="timeseries_synthetic", model_name="timeseries_cnn",
                  model_fn=lambda: TimeSeriesCNN(1, num_classes), train_ds=train_ds, test_ds=test_ds, seeds=seeds,
                  ratios=ratios, methods=methods, epochs=epochs, batch_size=64,
                  opt_cfg=dict(optimizer="adam", lr=1e-3, weight_decay=1e-4), device=get_device("auto"),
                  exp_logger=lg, craig_cfg=dict(layers=layers, chunk_size=128, per_class=True),
                  extra=dict(seq_length=seq_length, num_classes=num_classes, experiment_tag=tag))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=4000)
    p.add_argument("--seq-length", type=int, default=100)
    p.add_argument("--num-classes", type=int, default=4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    p.add_argument("--ratios", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3, 0.5])
    p.add_argument("--methods", nargs="+", default=["FULL", "RANDOM", "CRAIG", "CRAIG_UNWEIGHTED"])
    a = p.parse_args()
    lg = ExperimentLogger("./results", "timeseries")
    run(a.n_train, a.seq_length, a.seeds, a.ratios, a.epochs, a.methods, a.num_classes, lg)
    print("CSV:", lg.save({"cli_args": vars(a), "data": "synthetic"}))


if __name__ == "__main__":
    main()
