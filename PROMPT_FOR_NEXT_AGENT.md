# MASTER HANDOVER PROMPT FOR NEXT AI AGENT / RESEARCH ENGINEER

```text
You are the lead ML research engineer continuing the implementation of the research project:

"Gradient-Matching Data Subset Selection for Efficient Model Training (CRAIG Reproduction & Extension)"

Repository: https://github.com/00Harshh/CRAIG.git
Primary Reference:
Mirzasoleiman, B., Bilmes, J., & Leskovec, J. (2020).
"Coresets for Data-Efficient Training of Machine Learning Models." ICML 2020.

================================================================================
1. CURRENT PROJECT STATE & COMPLETED WORK (PHASE 1)
================================================================================
Phase 1 is 100% COMPLETE, TESTED, and EMPIRICALLY VERIFIED:
- Repository architecture, pyproject.toml, requirements.txt, Makefile, and docs/ are established.
- Centralized reproducibility engine (`training/reproducibility.py`) handles seeds across Python random, NumPy, PyTorch CPU, MPS, and CUDA.
- Dataset loaders (`datasets/mnist.py`, `datasets/cifar10.py`, `datasets/timeseries.py`) with normalization and offline synthetic tensor generators for testing.
- Multinomial Logistic Regression (`models/logistic_regression.py`, 784 -> 10, 7850 params), CNNs (`models/mnist_cnn.py`), and 1D CNN (`models/timeseries_cnn.py`).
- Baselines implemented: `baselines/full_dataset.py` (100% data) and `baselines/random_selector.py` (5%, 10%, 20%, 30%, 50% subsets).
- Trainer and evaluator (`training/trainer.py`, `training/evaluator.py`, `training/metrics.py`) support standard and normalized weighted coreset loss with microsecond wall-clock timers.
- Structured logger (`training/logger.py`) outputs collision-free CSV and JSON records to `results/raw/`.
- Unit test suite (`tests/test_training.py`) passes 100% in < 0.1s.
- Real empirical baseline benchmark executed on MNIST and recorded at `results/raw/mnist_2026_09_23_001.csv`:
    * FULL (1000 samples, 2 epochs): 83.82% acc (3.84s)
    * RANDOM 50% (500 samples): 80.08% acc (1.03s)
    * RANDOM 20% (200 samples): 67.94% acc (1.14s)
    * RANDOM 10% (100 samples): 53.60% acc (0.98s)
    * RANDOM 5%  (50 samples):  25.77% acc (1.36s) -> Proves catastrophic collapse of random sampling at small data budgets!

================================================================================
2. IMMEDIATE TASK: IMPLEMENT PHASES 2, 3, 4, AND 5
================================================================================
Read `docs/implementation_plan.md`, `docs/mathematical_background.md`, and `docs/system_design.md` before writing code.
Follow the planned progression in order:

--------------------------------------------------------------------------------
PHASE 2: PER-SAMPLE GRADIENT ENGINE & AUTOGRAD VALIDATION
--------------------------------------------------------------------------------
1. Implement `craig/gradients.py`:
   - Function: `compute_per_sample_gradients(model, dataloader, loss_fn, device, layers='all', chunk_size=128, max_samples=None)`
   - Return shape: `[N, D]` where N = samples, D = flattened parameter gradient dimension.
   - Use vectorized `torch.func.vmap(torch.func.grad(loss_fn))` with an efficient micro-batch fallback.
   - Do NOT compute one gradient for the batch and call it per-sample!
   - Memory-aware chunking (`chunk_size=128`).
2. Implement `craig/representations.py`:
   - Full gradients, selected layer filtering (e.g., classifier head), and optional random projection (Johnson-Lindenstrauss).
3. Write & run unit tests in `tests/test_gradients.py`:
   - Verify gradient shape `[N, D]`.
   - Verify that per-sample gradient matches single-sample standard `loss.backward()` autograd via `torch.testing.assert_close(atol=1e-5, rtol=1e-4)`.
   - Verify chunking invariance (chunk_size 32 vs 128 produce identical gradients).

--------------------------------------------------------------------------------
PHASE 3: SIMILARITY / DISTANCE ENGINE & SUBMODULAR FACILITY LOCATION
--------------------------------------------------------------------------------
1. Implement `craig/similarity.py`:
   - Pairwise Euclidean distance `||g_i - g_j||_2` and Cosine distance.
   - Blockwise computation to prevent O(N^2) memory explosion for large N.
   - Transform distances into monotone submodular similarities: `s(i, j) = M - d(i, j)` where `M = max_{i, j} d(i, j)`.
   - Memory safety checks: error if N > threshold unless `--allow-large-computation`.
2. Implement `craig/facility_location.py`:
   - Greedy Facility Location selector maximizing `F(S) = sum_i max_{j in S} s(i, j)`.
   - Accelerated greedy selection maintaining running upper bounds `mu_i = max_{j in S} s(i, j)`.
   - Returns selected subset indices of exact size k.
3. Write & run unit tests in `tests/test_similarity.py` and `tests/test_facility_location.py`:
   - Symmetry, zero diagonal, non-negativity.
   - Monotonic submodular gains.

--------------------------------------------------------------------------------
PHASE 4: CORESET WEIGHTS, CRAIG SELECTOR & GRADIENT ERROR ENGINE
--------------------------------------------------------------------------------
1. Implement `craig/weights.py`:
   - Partition V into Voronoi clusters: `j*(i) = argmax_{j in S} s(i, j)`.
   - Representative weights: `w_j = |C_j|` such that `sum(w_j) = N` and `w_j >= 1`.
2. Implement `craig/selector.py`:
   - Unified `CraigSelector` class: `select(...) -> (selected_indices, weights, runtime_metrics)`.
3. Implement `analysis/gradient_error.py`:
   - Full gradient: `G_full = sum_i g_i`
   - Weighted subset gradient: `G_subset = sum_{j in S} w_j g_j`
   - Relative Error: `||G_full - G_subset|| / ||G_full||`
   - Cosine Similarity: `(G_full . G_subset) / (||G_full|| * ||G_subset||)`
4. Write & run unit tests in `tests/test_selector.py`.

--------------------------------------------------------------------------------
PHASE 5: END-TO-END BENCHMARKS, BREAK-EVEN ANALYSIS & EXTENSIONS
--------------------------------------------------------------------------------
1. Benchmark MNIST (FULL vs RANDOM vs CRAIG weighted vs CRAIG unweighted):
   - Update `experiments/run_mnist.py` to include CRAIG.
   - Run across 5 seeds and subset sizes 5%, 10%, 20%, 30%, 50%.
2. Implement `experiments/run_cifar10.py` (CIFAR-10 + CNN).
3. Implement `experiments/run_timeseries.py` (Time-Series + 1D CNN).
4. Implement `experiments/run_scaling.py`:
   - Sequence length scaling (50, 100, 200, 500, 1000).
   - Dataset size scaling (1000, 2000, 5000, 10000).
5. Implement `analysis/runtime_analysis.py`:
   - Rigorous break-even evaluation: is `T_selection + T_subset_training < T_full_training`?
   - Report both `training_only_speedup` AND `end_to_end_speedup`.
6. Implement `analysis/plots.py`:
   - Automatically generate publication figures in `results/figures/` loaded directly from CSV.

================================================================================
3. STRICT RESEARCH INTEGRITY & EXECUTION RULES
================================================================================
- NEVER fabricate numbers or copy metrics from the paper. All results must be generated via actual execution.
- NEVER hide selection overhead. Total CRAIG time MUST include gradient computation + similarity computation + selection + training.
- NEVER assume CUDA exists. Auto-detect CUDA -> Apple Silicon MPS -> CPU.
- Follow test-driven progression: run unit tests at each phase before running big benchmarks.
```
