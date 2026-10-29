# System Design: CRAIG Research Framework

## 1. System Architecture

The framework is organized into decoupled layers:
```
                ┌────────────────────────────────┐
                │          CLI & Config          │
                │ (argparse / PyYAML / profiles) │
                └───────────────┬────────────────┘
                                │
        ┌───────────────────────┴───────────────────────┐
        ▼                                               ▼
┌──────────────────┐                           ┌──────────────────┐
│ Datasets Engine  │                           │   Models Layer   │
│ - MNIST          │                           │ - Logistic Reg.  │
│ - CIFAR-10       │                           │ - MNIST/CIFAR CNN│
│ - TimeSeries     │                           │ - 1D CNN         │
└────────┬─────────┘                           └────────┬─────────┘
         │                                              │
         └──────────────────────┬───────────────────────┘
                                ▼
         ┌──────────────────────────────────────────────┐
         │            Subset Selection Layer            │
         │  - FullDatasetSelector (100% baseline)       │
         │  - RandomSubsetSelector (Random baseline)    │
         │  - CraigSelector (Gradient + Facility Loc.)  │
         └──────────────────────┬───────────────────────┘
                                ▼
         ┌──────────────────────────────────────────────┐
         │        Training & Evaluation Engine          │
         │  - Standard & Normalized Weighted Loss       │
         │  - Microsecond Profiling & Memory Tracker    │
         │  - Reproducibility & Multi-Device Dispatch   │
         └──────────────────────┬───────────────────────┘
                                ▼
         ┌──────────────────────────────────────────────┐
         │       Logging, Analysis & Storage            │
         │  - CSV/JSON Raw Records (results/raw/)       │
         │  - Analysis & Plots (results/figures/)       │
         └──────────────────────────────────────────────┘
```

## 2. Module Responsibilities

1. `training/reproducibility.py`: Single source of truth for PRNG seeds across Python, NumPy, PyTorch CPU, MPS, and CUDA. Device selection and unified memory tracking.
2. `models/`: Minimal, clean architectures providing `forward(x)` and `count_parameters()`.
3. `datasets/`: Dataset loading, preprocessing, standard normalizations, and fast synthetic dataset generators for unit testing.
4. `baselines/`: Exact implementations of the Full Dataset and uniform Random Subset baselines with timing and coreset weight assignments.
5. `training/trainer.py` & `evaluator.py`: Multi-epoch training loops supporting both unweighted training and normalized weighted training where each sample is scaled by $w_i$.
6. `training/logger.py`: Guaranteed collision-free experiment IDs and dual-format output (CSV for tabular analysis, JSON for deep configuration metadata).

## 3. Memory & Computational Complexity

- **Model Parameters ($D$)**:
  - Logistic Regression (MNIST): $784 \times 10 + 10 = 7,850$
  - Small CNN: $~10^5$
- **Gradient Matrix**: Shape $[N, D]$. Storing the full gradient matrix for $N=60,000$ and $D=7,850$ in float32 requires:
  $$\frac{60,000 \times 7,850 \times 4 \text{ bytes}}{1024^3} \approx 1.75 \text{ GB}$$
- **Pairwise Distance Matrix**: Shape $[N, N]$. For $N=60,000$:
  $$\frac{60,000^2 \times 4 \text{ bytes}}{1024^3} \approx 13.4 \text{ GB}$$
  To prevent out-of-memory crashes on memory-constrained machines, the framework features chunked gradient computation, blockwise distance calculation, and safety thresholds (`max_samples_safety`).
