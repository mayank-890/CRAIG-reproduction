"""Shared FULL / RANDOM / CRAIG benchmark loop used by all experiment runners.

Timing honesty: for CRAIG, total_time = gradient + similarity + facility location + weights
+ training. Nothing is cached across runs: selection is recomputed for every (seed, ratio).
"""
import logging, time
from typing import Callable, Dict, List, Any
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from training.reproducibility import set_seed, get_memory_info_mb
from training.trainer import train_model
from training.evaluator import evaluate_model
from datasets.mnist import TensorSubsetDataset
from baselines.full_dataset import FullDatasetSelector
from baselines.random_selector import RandomSubsetSelector
from craig.selector import CraigSelector
from craig.gradients import compute_per_sample_gradients
from analysis.gradient_error import gradient_error

logger = logging.getLogger(__name__)


def materialize(ds: Dataset):
    """Load a dataset into (X, y) tensors once so subset loaders are cheap and identical across methods."""
    loader = DataLoader(ds, batch_size=2048, shuffle=False)
    xs, ys = zip(*[(b[0], b[1]) for b in loader])
    return torch.cat(xs), torch.cat(ys)


def make_optimizer(model, opt_cfg):
    if opt_cfg.get("optimizer", "sgd") == "adam":
        return torch.optim.Adam(model.parameters(), lr=opt_cfg["lr"], weight_decay=opt_cfg.get("weight_decay", 0))
    return torch.optim.SGD(model.parameters(), lr=opt_cfg["lr"], momentum=opt_cfg.get("momentum", 0.9),
                           weight_decay=opt_cfg.get("weight_decay", 0))


def _train(model_fn, X, y, idx, w, test_loader, opt_cfg, epochs, batch_size, device, use_weights):
    ds = TensorSubsetDataset(X[idx], y[idx], torch.as_tensor(w, dtype=torch.float32) if use_weights else None)
    loader = DataLoader(ds, batch_size=min(batch_size, len(idx)), shuffle=True)
    model = model_fn().to(device)
    # val_loader=None so training_time measures optimisation only (no per-epoch test eval inside the timer)
    res = train_model(model, loader, None, nn.CrossEntropyLoss(), make_optimizer(model, opt_cfg),
                      epochs, device, use_weights=use_weights, log_interval=max(1, epochs))
    res["final_val_metrics"] = evaluate_model(model, test_loader, criterion=nn.CrossEntropyLoss(), device=device)
    return res


def run_benchmark(*, dataset: str, model_name: str, model_fn: Callable[[], nn.Module], train_ds, test_ds,
                  seeds: List[int], ratios: List[float], methods: List[str], epochs: int, batch_size: int,
                  opt_cfg: Dict[str, Any], device, exp_logger, craig_cfg: Dict[str, Any] = None,
                  extra: Dict[str, Any] = None) -> None:
    craig_cfg = craig_cfg or {}
    extra = extra or {}
    X, y = materialize(train_ds)
    N = len(y)
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False)
    base = dict(dataset=dataset, model=model_name, device=str(device), total_train_samples=N, epochs=epochs,
                learning_rate=opt_cfg["lr"], **extra)

    def log(method, seed, ratio, idx, w, sel_t, tm, res, gerr=None, bs=None):
        m = res["final_val_metrics"]
        rec = dict(base, method=method, seed=seed, subset_ratio=float(ratio), subset_size=len(idx),
                   batch_size=bs or min(batch_size, len(idx)), selection_time=sel_t,
                   gradient_computation_time=tm.get("gradient_time", 0.0),
                   similarity_computation_time=tm.get("similarity_time", 0.0),
                   facility_location_time=tm.get("facility_location_time", 0.0),
                   training_time=res["training_time"], total_time=sel_t + res["training_time"],
                   train_loss=res["final_train_loss"], test_loss=m.loss, accuracy=m.accuracy,
                   gradient_error=(gerr or {}).get("relative_error"),
                   gradient_cosine_similarity=(gerr or {}).get("cosine_similarity"),
                   peak_memory_mb=get_memory_info_mb(device))
        exp_logger.log_run(rec)

    for seed in seeds:
        logger.info("==== seed %d ====", seed)
        G_ref = None
        if "FULL" in methods:
            set_seed(seed)
            idx, w, t = FullDatasetSelector().select(N)
            res = _train(model_fn, X, y, idx, w, test_loader, opt_cfg, epochs, batch_size, device, False)
            log("FULL", seed, 1.0, idx, w, t, {}, res)
        for ratio in ratios:
            sel_model = None
            if "CRAIG" in methods or "CRAIG_UNWEIGHTED" in methods:
                set_seed(seed)
                sel_model = model_fn()  # same init the trained model gets under this seed
                selector = CraigSelector(metric=craig_cfg.get("metric", "euclidean"),
                                         layers=craig_cfg.get("layers", "all"),
                                         per_class=craig_cfg.get("per_class", True),
                                         projection_dim=craig_cfg.get("projection_dim"),
                                         chunk_size=craig_cfg.get("chunk_size", 128), device=device,
                                         keep_gradients=True)
                cres = selector.select(sel_model, TensorSubsetDataset(X, y), nn.CrossEntropyLoss(), subset_ratio=ratio)
                G_ref = cres.gradients
                gerr_c = gradient_error(G_ref, cres.indices, cres.weights)
                logger.info("CRAIG %.2f: k=%d sel=%.2fs (grad %.2f sim %.2f fl %.2f) relerr=%.3f", ratio,
                            len(cres.indices), cres.selection_time, cres.timings["gradient_time"],
                            cres.timings["similarity_time"], cres.timings["facility_location_time"], gerr_c["relative_error"])
                for meth, uw in (("CRAIG", True), ("CRAIG_UNWEIGHTED", False)):
                    if meth in methods:
                        set_seed(seed)
                        res = _train(model_fn, X, y, cres.indices, cres.weights, test_loader, opt_cfg, epochs,
                                     batch_size, device, uw)
                        ge = gerr_c if uw else gradient_error(G_ref, cres.indices, np.full(len(cres.indices), N / len(cres.indices)))
                        log(meth, seed, ratio, cres.indices, cres.weights, cres.selection_time, cres.timings, res, ge)
            if "RANDOM" in methods:
                set_seed(seed)
                idx, w, t = RandomSubsetSelector(seed=seed).select(N, subset_ratio=ratio)
                ge = None
                if G_ref is not None:  # evaluated at the same init params (not counted in RANDOM time)
                    ge = gradient_error(G_ref, idx, np.full(len(idx), N / len(idx)))
                set_seed(seed)
                res = _train(model_fn, X, y, idx, w, test_loader, opt_cfg, epochs, batch_size, device, False)
                log("RANDOM", seed, ratio, idx, w, t, {}, res, ge)
