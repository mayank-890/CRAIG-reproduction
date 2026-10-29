# Gradient-Matching Data Subset Selection for Efficient Model Training

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An empirical research reproduction and extension of the **CRAIG** framework for data-efficient machine learning model training based on submodular gradient matching:

> **Mirzasoleiman, B., Bilmes, J., & Leskovec, J. (2020).**  
> *"Coresets for Data-Efficient Training of Machine Learning Models."*  
> International Conference on Machine Learning (ICML 2020).

---

## 1. Central Research Question

> **"Can gradient-matching data subset selection reduce the amount of training data while maintaining comparable model performance, and when does the computational cost of subset selection become worthwhile?"**

### Core Research Questions
- **RQ1**: Can the CRAIG approach select a substantially smaller subset while preserving model performance?
- **RQ2**: How does subset size affect gradient approximation error and cosine similarity?
- **RQ3**: How does CRAIG compare with random subset selection at equivalent subset budgets?
- **RQ4**: Does gradient-matching subset selection remain effective for 1D time-series classification?
- **RQ5**: How does selection cost scale with dataset size, sequence length, and model complexity?
- **RQ6**: Break-Even Analysis: Under what conditions does $T_{\text{selection}} + T_{\text{subset\_training}} < T_{\text{full\_training}}$?

---

## 2. Background & Mathematical Formulation

Training deep neural networks on massive datasets requires immense computational resources. Rather than training on all $N$ examples uniformly, **Coreset Selection** identifies a representative subset $S \subseteq V$ of size $k \ll N$ with non-negative coreset weights $\{w_j\}_{j \in S}$ such that the weighted subset gradient closely matches the full dataset gradient:

$$\min_{S \subseteq V, w \ge 0} \left\| \sum_{i \in V} \nabla_\theta \ell_i(\theta) - \sum_{j \in S} w_j \nabla_\theta \ell_j(\theta) \right\|$$

CRAIG upper-bounds this gradient approximation error using the triangle inequality over partitions $\{C_j\}_{j \in S}$:

$$\mathcal{E}(S, w) \le \frac{1}{N} \sum_{j \in S} \sum_{i \in C_j} \|\nabla_\theta \ell_i(\theta) - \nabla_\theta \ell_j(\theta)\|$$

Setting $w_j = |C_j|$ transforms the problem into a monotone submodular **Facility Location** objective:

$$F(S) = \sum_{i \in V} \max_{j \in S} \big(M - \|\nabla \ell_i(\theta) - \nabla \ell_j(\theta)\|\big)$$

where $M \ge \max_{i, j} d(i, j)$. By submodularity, greedy selection guarantees a $(1 - 1/e) \approx 63.2\%$ approximation bound.

Detailed derivations and proofs are provided in [`docs/mathematical_background.md`](docs/mathematical_background.md).

---

## 3. System Architecture

```text
craig-research/
├── configs/            # Experiment YAML configurations (MNIST, CIFAR-10, Time-Series)
├── craig/              # Per-sample gradients, representations, facility location, weights
├── models/             # Logistic Regression, MNIST/CIFAR CNN, Time-Series 1D CNN
├── datasets/           # MNIST, CIFAR-10, and Time-series benchmark loaders
├── baselines/          # Full Dataset (100%) and Random Subset selectors
├── training/           # Reproducibility, Trainer, Evaluator, Metrics, Logger
├── experiments/        # CLI experiment runners (run_mnist.py, etc.)
├── analysis/           # Gradient approximation error, runtime, scaling, plots
├── scripts/            # verify_installation.py, download_data.py, run_all.py
├── tests/              # Unit tests (test_training.py, etc.)
├── notebooks/          # Interactive walkthrough notebooks
├── docs/               # In-depth mathematical and system documentation
├── results/            # Raw CSV/JSON records, processed metrics, figures
└── checkpoints/        # Resumable experiment checkpoints and cache
```

---

## 4. Installation & Quick Start

### Prerequisites
- Python 3.11+
- PyTorch (supports Apple Silicon MPS, NVIDIA CUDA, and CPU)

```bash
# 1. Clone repository
git clone <repo-url>
cd craig-research

# 2. Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Verification
Run the system self-check:
```bash
python scripts/verify_installation.py
```

Run unit tests:
```bash
python -m unittest discover -s tests -p "test_*.py"
```

---

## 5. Running Experiments

### Phase 1: Baseline Verification (MNIST + Logistic Regression)
Run debug baseline (1,000 samples, fast validation):
```bash
python experiments/run_mnist.py --config configs/mnist.yaml --profile debug
```

Run standard baseline (Full dataset + Random subsets 5%, 10%, 20%, 30%, 50% across 5 seeds):
```bash
python experiments/run_mnist.py --config configs/mnist.yaml --profile standard
```

All experimental outputs are saved directly to `results/raw/mnist_*.csv` and `results/raw/mnist_*.json`.

---

## 6. Experimental Results

> **Note on Research Integrity**: Results are loaded strictly from actual experimental runs. No numbers are fabricated or estimated.

### Measured Baseline Results (MNIST, Debug Profile, Seed 42, 2 Epochs)

| Method | Subset Ratio | Samples ($k$) | Test Accuracy | Train Loss | Test Loss | Selection Time | Training Time | Total Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FULL** | 100% | 1,000 | **83.82%** | 0.4175 | 0.5152 | 0.02 ms | 3.84s | 3.84s |
| **RANDOM** | 50% | 500 | 80.08% | 0.6320 | 0.5980 | 0.05 ms | 1.03s | 1.03s |
| **RANDOM** | 20% | 200 | 67.94% | 1.3050 | 1.0720 | 0.05 ms | 1.14s | 1.14s |
| **RANDOM** | 10% | 100 | 53.60% | 1.7641 | 1.6283 | 0.09 ms | 0.98s | 0.98s |
| **RANDOM** | 5% | 50 | 25.77% | 2.1203 | 2.0798 | 0.99 ms | 1.36s | 1.36s |

*(Standard 5-seed benchmark results and CRAIG comparisons will be populated as further phases execute).*

---

## 7. Limitations & Computational Considerations

1. **Selection Overhead**: Computing per-sample gradients and the $\mathcal{O}(N^2)$ pairwise distance matrix incurs significant upfront overhead.
2. **Break-Even Regime**: For simple architectures trained for only a few epochs, selection cost will dominate. The coreset advantage manifests primarily during long multi-epoch training or hyperparameter grid sweeps where the subset is reused.
3. **Memory Footprint**: Full gradient and similarity matrices for large $N$ require blockwise calculation or gradient projection to avoid exceeding device RAM.

---

## 8. References

- Mirzasoleiman, B., Bilmes, J., & Leskovec, J. (2020). *Coresets for Data-Efficient Training of Machine Learning Models.* ICML 2020.
- Nemhauser, G. L., Wolsey, L. A., & Fisher, M. L. (1978). *An analysis of approximations for maximizing submodular set functions—I.* Mathematical Programming.
