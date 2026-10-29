#!/usr/bin/env python3
"""MNIST: FULL vs RANDOM vs CRAIG (weighted) vs CRAIG_UNWEIGHTED, logistic regression."""
import os, sys, argparse, logging
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import yaml, torch
from training.reproducibility import get_device
from training.logger import ExperimentLogger
from models.logistic_regression import LogisticRegression
from datasets.mnist import get_mnist_datasets
from experiments.common import run_benchmark

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/mnist.yaml")
    p.add_argument("--n-train", type=int, default=None, help="Use first N training samples (default: all 60000)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--ratios", type=float, nargs="+", default=None)
    p.add_argument("--methods", nargs="+", default=["FULL", "RANDOM", "CRAIG", "CRAIG_UNWEIGHTED"])
    p.add_argument("--device", default=None)
    a = p.parse_args()
    cfg = yaml.safe_load(open(a.config))
    tr = cfg["training"]
    device = get_device(a.device or cfg.get("device", {}).get("type", "auto"))
    n = a.n_train if a.n_train is not None else cfg["dataset"].get("subset_of_training_data")
    train_ds, test_ds = get_mnist_datasets(cfg["dataset"]["data_dir"], flatten=True, max_train_samples=n)
    lg = ExperimentLogger(cfg["logging"]["results_dir"], "mnist")
    run_benchmark(dataset="mnist", model_name="logistic_regression", model_fn=lambda: LogisticRegression(784, 10),
                  train_ds=train_ds, test_ds=test_ds, seeds=a.seeds or cfg["experiments"]["seeds"],
                  ratios=a.ratios or cfg["craig"]["subset_ratios"], methods=a.methods,
                  epochs=a.epochs or tr["epochs"], batch_size=tr["batch_size"],
                  opt_cfg=dict(lr=tr["learning_rate"], momentum=tr["momentum"], weight_decay=tr["weight_decay"]),
                  device=device, exp_logger=lg,
                  craig_cfg=dict(metric=cfg["craig"]["distance_metric"], layers=cfg["craig"]["gradient_layer"],
                                 chunk_size=cfg["craig"]["chunk_size"], per_class=True))
    a_dict = vars(a); cfg["cli_args"] = a_dict
    print("CSV:", lg.save(cfg))


if __name__ == "__main__":
    main()
