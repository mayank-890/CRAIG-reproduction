#!/usr/bin/env python3
"""CIFAR-10: FULL vs RANDOM vs CRAIG with CIFARCNN.

CRAIG uses last-layer (classifier head, fc2) per-sample gradients, as in Mirzasoleiman et al.
for deep nets; selection is done once at initialisation (no per-epoch reselection).
"""
import os, sys, argparse, logging
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import yaml
from training.reproducibility import get_device
from training.logger import ExperimentLogger
from models.mnist_cnn import CIFARCNN
from datasets.cifar10 import get_cifar10_datasets
from experiments.common import run_benchmark

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/cifar10.yaml")
    p.add_argument("--n-train", type=int, default=10000)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--ratios", type=float, nargs="+", default=None)
    p.add_argument("--layers", default="last")
    p.add_argument("--methods", nargs="+", default=["FULL", "RANDOM", "CRAIG", "CRAIG_UNWEIGHTED"])
    a = p.parse_args()
    cfg = yaml.safe_load(open(a.config)); tr = cfg["training"]
    device = get_device(cfg.get("device", {}).get("type", "auto"))
    train_ds, test_ds = get_cifar10_datasets(cfg["dataset"]["data_dir"], max_train_samples=a.n_train)
    lg = ExperimentLogger(cfg["logging"]["results_dir"], "cifar10")
    run_benchmark(dataset="cifar10", model_name="cifar_cnn", model_fn=lambda: CIFARCNN(10), train_ds=train_ds,
                  test_ds=test_ds, seeds=a.seeds or cfg["experiments"]["seeds"],
                  ratios=a.ratios or cfg["craig"]["subset_ratios"], methods=a.methods,
                  epochs=a.epochs or tr["epochs"], batch_size=tr["batch_size"],
                  opt_cfg=dict(lr=tr["learning_rate"], momentum=tr["momentum"], weight_decay=tr["weight_decay"]),
                  device=device, exp_logger=lg,
                  craig_cfg=dict(metric=cfg["craig"]["distance_metric"], layers=a.layers, chunk_size=64, per_class=True))
    cfg["cli_args"] = vars(a)
    print("CSV:", lg.save(cfg))


if __name__ == "__main__":
    main()
