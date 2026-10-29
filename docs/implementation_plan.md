# Implementation Plan: CRAIG Research Project — Phases 2 to 5

This document details the multi-phase implementation roadmap for reproducing and evaluating **CRAIG** (*Mirzasoleiman et al., ICML 2020*), proceeding from per-sample gradient extraction to end-to-end benchmarking, break-even analysis, and time-series extensions.

---

## Roadmap Overview

```
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 1: Baseline Foundation (COMPLETED & VERIFIED)                    │
│ ✔ Scaffolding, config, reproducibility, MNIST Logistic, Full vs Random │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 2: Per-Sample Gradient Engine & Validation (Sections 7, 8, 9, 10)│
│ • PyTorch vmap/grad per-sample gradient extraction                     │
│ • Memory-aware chunking & gradient representations (full/projected)    │
│ • Autograd exactness validation test suite                             │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 3: Similarity Engine & Facility Location (Sections 11, 12, 33)   │
│ • Pairwise Euclidean & Cosine distance with blockwise memory safety    │
│ • Monotone submodular similarity transformation                        │
│ • Accelerated greedy facility-location selector                        │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 4: Coreset Weights, Selector & Gradient Error (Sec 13, 14, 16)   │
│ • Voronoi partition cluster weights (sum(w) = N, w_i >= 1)             │
│ • Unified CraigSelector pipeline (indices, weights, timing breakdown)  │
│ • Gradient approximation error (relative error, cosine similarity)     │
│ • Weighted vs unweighted subset training ablation                      │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Phase 5: Empirical Benchmarks, Break-Even & Scaling (Sec 15, 17-29)   │
│ • MNIST FULL vs RANDOM vs CRAIG (5 seeds, 5% to 50% ratios)            │
│ • CIFAR-10 + CNN experiments                                           │
│ • Time-Series Classification (1D CNN) & Sequence Length Scaling        │
│ • Rigorous Break-Even Analysis (T_select + T_subset vs T_full)         │
│ • Publication-quality automated plotting engine (results/figures/)     │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Specifications by Phase

### Phase 2: Per-Sample Gradient Engine & Autograd Validation

#### 1. Core Objectives
- Extract true per-sample gradients $g_i = \nabla_\theta \ell(f(x_i; \theta), y_i) \in \mathbb{R}^D$ without batch-averaging.
- Implement memory-safe chunking to prevent OOM when $N \times D$ is large.
- Provide gradient representation options: `full_gradient`, `last_layer` (classifier head), and optional `random_projection` (Johnson-Lindenstrauss).
- Prove exact equivalence against PyTorch standard single-sample autograd.

#### 2. Modules to Implement
- [NEW] `craig/gradients.py`:
  - `compute_per_sample_gradients(model, dataloader, loss_fn, device, layers='all', chunk_size=128, max_samples=None)`
  - Uses native vectorized PyTorch `torch.func.vmap(torch.func.grad(loss_fn))` for models supporting functional call, with an efficient micro-batch autograd fallback for arbitrary architectures.
  - Returns 2D tensor `[N, D]` and parameter metadata.
- [NEW] `craig/representations.py`:
  - `GradientRepresentation`: handles layer filtering (e.g. output layer vs all layers) and Gaussian/Rademacher random projection to dimension $d \ll D$ with fixed projection seed.
- [NEW] `tests/test_gradients.py`:
  - `test_per_sample_gradient_shape`: Verifies output is exactly $[N, D]$.
  - `test_per_sample_gradient_correctness`: Compares $g_i$ from engine against standard single-sample `loss.backward()` autograd via `torch.testing.assert_close(atol=1e-5, rtol=1e-4)`.
  - `test_layer_selection`: Verifies parameter filtering.
  - `test_chunking_invariance`: Proves chunk_size=32 and chunk_size=128 produce identical gradients.

---

### Phase 3: Similarity Engine & Facility Location

#### 1. Core Objectives
- Compute pairwise gradient distance matrix $D_{ij} = \|g_i - g_j\|_2$ and cosine distance with memory checks.
- Transform distances to monotone submodular similarity:
  $$s(i, j) = M - d(i, j), \quad M = \max_{i,j} d(i, j)$$
- Implement the greedy submodular Facility Location algorithm:
  $$F(S) = \sum_{i \in V} \max_{j \in S} s(i, j)$$
  maximizing marginal gain $\Delta(c \mid S) = \sum_{i \in V} \max(0, s(i, c) - \max_{j \in S} s(i, j))$.

#### 2. Modules to Implement
- [NEW] `craig/similarity.py`:
  - `compute_distance_matrix(gradients, metric="euclidean", block_size=1024)`
  - Blockwise calculation to keep memory $\mathcal{O}(B \times N)$ instead of requiring all $N^2$ in GPU VRAM simultaneously.
  - Safety guard: raises informative error or warns if $N > \text{max\_samples\_safety}$ unless `--allow-large-computation` is set.
- [NEW] `craig/facility_location.py`:
  - `greedy_facility_location(similarity_matrix, k, lazy_greedy=True)`
  - Implements greedy selection with tracking of upper bound running maxima $\mu_i = \max_{j \in S} s(i, j)$.
  - Returns `selected_indices` of exact length $k$ and trajectory of objective values.
- [NEW] `tests/test_similarity.py` & `tests/test_facility_location.py`:
  - Verifies distance symmetry ($d(i, j) = d(j, i)$), zero diagonal ($d(i, i) = 0$), and non-negativity.
  - Verifies monotonic submodular gains ($\Delta(c \mid S_1) \ge \Delta(c \mid S_2)$ for $S_1 \subseteq S_2$).
  - Verifies exact output subset size $k$.

---

### Phase 4: Coreset Weights, CRAIG Selector & Gradient Error

#### 1. Core Objectives
- Partition $V$ into Voronoi clusters $\{C_j\}_{j \in S}$ where each element $i \in V$ is assigned to its most similar selected representative:
  $$j^*(i) = \arg\max_{j \in S} s(i, j)$$
- Assign representative coreset weights $w_j = |C_j|$ ensuring $\sum_{j \in S} w_j = N$ and $w_j \ge 1$.
- Implement quantitative gradient approximation error metrics:
  $$\text{Relative Error} = \frac{\|G_{\text{full}} - G_{\text{subset}}\|_2}{\|G_{\text{full}}\|_2}, \quad \text{Cosine Sim} = \frac{G_{\text{full}} \cdot G_{\text{subset}}}{\|G_{\text{full}}\|_2 \|G_{\text{subset}}\|_2}$$
- Bundle into a unified, modular `CraigSelector`.

#### 2. Modules to Implement
- [NEW] `craig/weights.py`:
  - `compute_coreset_weights(similarity_matrix, selected_indices)`
  - Vectorized assignment and cluster counting; returns integer/float weights array of shape $[k]$.
- [NEW] `craig/selector.py`:
  - `CraigSelector`: End-to-end interface:
    `select(model, dataloader, criterion, subset_ratio, device, ...)`
    $\to$ `(selected_indices, weights, RuntimeMetrics)`
- [NEW] `analysis/gradient_error.py`:
  - `compute_gradient_error(full_gradients, selected_indices, weights=None)`
  - Computes exact relative error, absolute error, and cosine similarity.
- [NEW] `tests/test_selector.py`:
  - Verifies weight sum matches $N$, positivity of weights, and end-to-end selector reproducibility.

---

### Phase 5: Empirical Benchmarks, Break-Even & Scaling

#### 1. Core Objectives
- Run complete multi-seed evaluation on **MNIST**:
  - FULL vs RANDOM vs CRAIG (weighted) vs CRAIG (unweighted ablation).
  - 5 seeds (`42, 123, 456, 789, 1000`), 5 subset ratios (`0.05, 0.10, 0.20, 0.30, 0.50`).
- Run **CIFAR-10** benchmark with `CIFARCNN`.
- Run **Time-Series Classification** benchmark with `TimeSeriesCNN` and sequence length scaling (50, 100, 200, 500, 1000).
- Run **Dataset Size Scaling** ($N = 1000, 2000, 5000, 10000$) to evaluate computational scaling.
- Empirically test the **Break-Even Hypothesis**:
  Evaluate whether $T_{\text{selection}} + T_{\text{subset\_training}} < T_{\text{full\_training}}$.
- Generate all 12 publication-quality figures in `results/figures/` directly from CSV logs.

#### 2. Modules to Implement
- [NEW] `experiments/run_cifar10.py`: CIFAR-10 benchmark runner.
- [NEW] `experiments/run_timeseries.py`: Time-series benchmark runner.
- [NEW] `experiments/run_scaling.py`: Sequence length and dataset size scaling runner.
- [NEW] `analysis/runtime_analysis.py`: Break-even calculation and dual speedup reporting (`training_only_speedup` vs `end_to_end_speedup`).
- [NEW] `analysis/statistical_analysis.py`: Aggregates mean $\pm$ standard deviation across seeds into publication tables (CSV & Markdown).
- [NEW] `analysis/plots.py`: Generates publication figures:
  1. Accuracy vs Subset Size (FULL, RANDOM, CRAIG)
  2. Loss vs Subset Size
  3. Gradient Approximation Error vs Subset Size
  4. Gradient Cosine Similarity vs Subset Size
  5. Training Time vs Subset Size
  6. Break-Even: Total CRAIG Time vs Full Training Time
  7. Speedup vs Subset Size (Training-only vs End-to-end)
  8. Sequence Length vs Selection Time & Total Time
  9. Dataset Size vs Selection Time & Memory Usage
- [NEW] `scripts/run_all.py`: Master script orchestrating the entire suite.

---

## Verification & Acceptance Plan

| Phase | Automated Verification Method | Expected Outcome |
| :--- | :--- | :--- |
| **Phase 2** | `python -m unittest tests/test_gradients.py` | Per-sample gradients match autograd within $10^{-4}$ tolerance; chunking invariant. |
| **Phase 3** | `python -m unittest tests/test_similarity.py tests/test_facility_location.py` | Distances symmetric & positive; greedy gains monotonically submodular; exactly $k$ samples selected. |
| **Phase 4** | `python -m unittest tests/test_selector.py` | Coreset weights sum to $N$; relative gradient error computed accurately. |
| **Phase 5 (MNIST)** | `python experiments/run_mnist.py --config configs/mnist.yaml --profile debug` | CRAIG runs end-to-end; outperforms Random at 5% and 10% data budgets. |
| **Phase 5 (Full)** | `python experiments/run_timeseries.py && python experiments/run_scaling.py` | Time-series and scaling experiments log to CSV; figures generated in `results/figures/`. |

---

## Safety & Resource Controls
- **Memory Safety**: Blockwise distance calculation and gradient projection option ensure we never allocate huge memory arrays.
- **Hardware Independence**: Automatically uses Apple Silicon MPS on macOS, CUDA on GPU, and CPU elsewhere.
- **Research Integrity**: Results are loaded strictly from recorded CSV files, with both positive and negative findings reported.
