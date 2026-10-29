#!/usr/bin/env python3
"""Run MNIST baseline experiments: Full Dataset and Random Subsets across multiple seeds."""

import os
import sys
import argparse
import time
import logging
from typing import Dict, Any, List
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.reproducibility import set_seed, get_device, get_memory_info_mb
from training.logger import ExperimentLogger
from training.trainer import train_model
from training.evaluator import evaluate_model
from models.logistic_regression import LogisticRegression
from datasets.mnist import get_mnist_datasets
from baselines.full_dataset import FullDatasetSelector
from baselines.random_selector import RandomSubsetSelector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_config(config_path: str, profile: str = None) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if profile and "profiles" in config and profile in config["profiles"]:
        prof_cfg = config["profiles"][profile]
        logger.info("Applying configuration profile: '%s'", profile)
        if "epochs" in prof_cfg:
            config["training"]["epochs"] = prof_cfg["epochs"]
        if "batch_size" in prof_cfg:
            config["training"]["batch_size"] = prof_cfg["batch_size"]
        if "subset_of_training_data" in prof_cfg:
            config["dataset"]["subset_of_training_data"] = prof_cfg["subset_of_training_data"]
        if "seeds" in prof_cfg:
            config["experiments"]["seeds"] = prof_cfg["seeds"]
        if "craig" in prof_cfg and "subset_ratios" in prof_cfg["craig"]:
            config["craig"]["subset_ratios"] = prof_cfg["craig"]["subset_ratios"]

    return config


def run_experiment(config: Dict[str, Any], methods: List[str] = None) -> str:
    if methods is None:
        methods = ["FULL", "RANDOM"]

    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    craig_cfg = config.get("craig", {})
    exp_cfg = config.get("experiments", {})
    
    device = get_device(config.get("device", {}).get("type", "auto"))
    logger.info("Starting MNIST Experiment on device: %s", device)

    # 1. Load Datasets
    t0_data = time.perf_counter()
    train_dataset, test_dataset = get_mnist_datasets(
        data_dir=dataset_cfg.get("data_dir", "./data"),
        flatten=True,
        max_train_samples=dataset_cfg.get("subset_of_training_data", None),
        download=True,
    )
    data_load_time = time.perf_counter() - t0_data
    total_train_samples = len(train_dataset)
    logger.info("Loaded MNIST: %d train samples, %d test samples in %.2fs", total_train_samples, len(test_dataset), data_load_time)

    test_loader = DataLoader(
        test_dataset,
        batch_size=train_cfg.get("batch_size", 128),
        shuffle=False,
        num_workers=dataset_cfg.get("num_workers", 0),
    )

    exp_logger = ExperimentLogger(
        results_dir=config.get("logging", {}).get("results_dir", "./results"),
        dataset_name="mnist",
    )
    logger.info("Initialized Experiment ID: %s", exp_logger.experiment_id)

    seeds = exp_cfg.get("seeds", [42])
    subset_ratios = craig_cfg.get("subset_ratios", [0.05, 0.10, 0.20, 0.30, 0.50])
    epochs = train_cfg.get("epochs", 20)
    batch_size = train_cfg.get("batch_size", 128)
    lr = train_cfg.get("learning_rate", 0.01)
    momentum = train_cfg.get("momentum", 0.9)
    weight_decay = train_cfg.get("weight_decay", 0.0001)

    criterion = nn.CrossEntropyLoss()

    for seed in seeds:
        logger.info("=" * 60)
        logger.info(">>> RUNNING SEED: %d", seed)
        logger.info("=" * 60)

        # ----------------------------------------------------
        # FULL DATASET BASELINE
        # ----------------------------------------------------
        if "FULL" in methods:
            set_seed(seed)
            full_selector = FullDatasetSelector()
            indices, weights, sel_time = full_selector.select(total_train_samples)
            
            full_subset = Subset(train_dataset, indices)
            full_loader = DataLoader(full_subset, batch_size=batch_size, shuffle=True)

            model = LogisticRegression(input_dim=model_cfg.get("input_dim", 784), num_classes=model_cfg.get("num_classes", 10))
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

            logger.info("[FULL DATA] Seed %d: Training %d samples for %d epochs...", seed, len(indices), epochs)
            t_train_start = time.perf_counter()
            train_res = train_model(
                model=model,
                train_loader=full_loader,
                val_loader=test_loader,
                criterion=criterion,
                optimizer=optimizer,
                epochs=epochs,
                device=device,
                use_weights=False,
                log_interval=max(1, epochs // 4),
            )
            training_time = train_res["training_time"]
            total_time = sel_time + training_time
            eval_metrics = train_res["final_val_metrics"]

            rec = {
                "dataset": "mnist",
                "model": "logistic_regression",
                "method": "FULL",
                "seed": seed,
                "device": str(device),
                "subset_ratio": 1.0,
                "subset_size": len(indices),
                "total_train_samples": total_train_samples,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": lr,
                "selection_time": sel_time,
                "gradient_computation_time": 0.0,
                "similarity_computation_time": 0.0,
                "training_time": training_time,
                "total_time": total_time,
                "train_loss": train_res["final_train_loss"],
                "test_loss": eval_metrics.loss if eval_metrics else 0.0,
                "accuracy": eval_metrics.accuracy if eval_metrics else 0.0,
                "gradient_error": None,
                "gradient_cosine_similarity": None,
                "peak_memory_mb": get_memory_info_mb(device),
            }
            exp_logger.log_run(rec)

        # ----------------------------------------------------
        # RANDOM SUBSET BASELINE
        # ----------------------------------------------------
        if "RANDOM" in methods:
            for ratio in subset_ratios:
                set_seed(seed)
                rand_selector = RandomSubsetSelector(seed=seed)
                indices, weights, sel_time = rand_selector.select(
                    num_samples=total_train_samples,
                    subset_ratio=ratio,
                    weighted=False,
                )
                
                subset_ds = Subset(train_dataset, indices)
                sub_loader = DataLoader(subset_ds, batch_size=min(batch_size, len(indices)), shuffle=True)

                model = LogisticRegression(input_dim=model_cfg.get("input_dim", 784), num_classes=model_cfg.get("num_classes", 10))
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

                logger.info("[RANDOM %4.1f%%] Seed %d: Training %d samples for %d epochs...", ratio * 100, seed, len(indices), epochs)
                train_res = train_model(
                    model=model,
                    train_loader=sub_loader,
                    val_loader=test_loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    epochs=epochs,
                    device=device,
                    use_weights=False,
                    log_interval=max(1, epochs // 4),
                )
                training_time = train_res["training_time"]
                total_time = sel_time + training_time
                eval_metrics = train_res["final_val_metrics"]

                rec = {
                    "dataset": "mnist",
                    "model": "logistic_regression",
                    "method": "RANDOM",
                    "seed": seed,
                    "device": str(device),
                    "subset_ratio": float(ratio),
                    "subset_size": len(indices),
                    "total_train_samples": total_train_samples,
                    "epochs": epochs,
                    "batch_size": min(batch_size, len(indices)),
                    "learning_rate": lr,
                    "selection_time": sel_time,
                    "gradient_computation_time": 0.0,
                    "similarity_computation_time": 0.0,
                    "training_time": training_time,
                    "total_time": total_time,
                    "train_loss": train_res["final_train_loss"],
                    "test_loss": eval_metrics.loss if eval_metrics else 0.0,
                    "accuracy": eval_metrics.accuracy if eval_metrics else 0.0,
                    "gradient_error": None,
                    "gradient_cosine_similarity": None,
                    "peak_memory_mb": get_memory_info_mb(device),
                }
                exp_logger.log_run(rec)

    csv_path = exp_logger.save(config)
    print("\n" + "=" * 80)
    print(" EXPERIMENT COMPLETE — RESULTS SAVED")
    print(f" CSV:  {csv_path}")
    print("=" * 80 + "\n")
    return csv_path


def main():
    parser = argparse.ArgumentParser(description="Run MNIST baseline experiments.")
    parser.add_argument("--config", default="configs/mnist.yaml", help="Path to YAML config.")
    parser.add_argument("--profile", choices=["debug", "standard", "large"], default=None, help="Execution profile.")
    parser.add_argument("--device", default=None, help="Device override ('cpu', 'mps', 'cuda').")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Override seeds.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs.")
    args = parser.parse_args()

    config = load_config(args.config, profile=args.profile)
    if args.device:
        config["device"] = {"type": args.device}
    if args.seeds:
        config["experiments"]["seeds"] = args.seeds
    if args.epochs:
        config["training"]["epochs"] = args.epochs

    run_experiment(config)


if __name__ == "__main__":
    main()
