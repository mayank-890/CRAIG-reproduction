# CRAIG Research Project — Presentation Guide & Defense Cheat Sheet

This guide is designed for presenting the project to an academic advisor, professor, or project evaluation committee.

---

## 1. Project Title & Overview

- **Project Title**: Gradient-Matching Data Subset Selection for Efficient Model Training
- **Foundational Paper**: Mirzasoleiman, Bilmes, & Leskovec (ICML 2020) — *"Coresets for Data-Efficient Training of Machine Learning Models"*
- **Core Research Question**: Can gradient-matching coreset selection reduce training data requirements while maintaining model accuracy, and when does the computational overhead break even?

---

## 2. 60-Second Opening Pitch

> "Respected Professor, my major project focuses on **efficient machine learning through gradient-matching data subset selection (CRAIG)**.
>
> Modern neural network training is bottle-necked by massive data volume and GPU costs, yet much of the data is redundant. While uniform random sampling can reduce data size, it drops rare examples and leads to severe performance degradation at small budgets.
>
> CRAIG solves this by selecting a small coreset $S$ with weights $w$ such that the weighted gradient $\sum_{j \in S} w_j \nabla \ell_j$ closely matches the full gradient $\sum_{i \in V} \nabla \ell_i$.
>
> We have built a clean, reproducible research framework from scratch. We established our baseline benchmarks comparing Full dataset training against Random subsets across multiple data ratios (5% to 50%) and multiple seeds, backed by automated unit tests and structured CSV logging."

---

## 3. Key Accomplishments to Highlight

1. **Strict Research Methodology**:
   - No hardcoded numbers or fake results.
   - Built a rigorous baseline before implementing complex submodular optimization.
2. **Reproducibility & Hardware Abstraction**:
   - Centralized PRNG seeding across Python, NumPy, PyTorch CPU, and Apple Silicon MPS.
   - Auto-detection of MPS / CUDA / CPU accelerators.
3. **Verified Empirical Baseline**:
   - MNIST + Multinomial Logistic Regression ($784 \to 10$, 7,850 trainable parameters).
   - Trained across Full (100%) and Random subsets (5%, 10%, 20%, 50%).
4. **Machine-Readable Experiment Records**:
   - Every run logs to `results/raw/mnist_*.csv` and `.json` with unique experiment IDs.
5. **Quality Assurance**:
   - 100% passing unit tests in `tests/test_training.py` using synthetic in-memory tensors (runs in < 0.1s without internet dependency).

---

## 4. Empirical Data to Present

From our verified baseline run (`results/raw/mnist_2026_09_23_001.csv`):

| Method | Data Budget | Training Time | Test Accuracy | Key Takeaway |
| :--- | :---: | :---: | :---: | :--- |
| **FULL Dataset** | 100% (1,000) | 3.84 s | **83.82%** | Upper performance bound |
| **RANDOM** | 50% (500) | 1.03 s | **80.08%** | 73% speedup with minimal accuracy loss (-3.7%) |
| **RANDOM** | 20% (200) | 1.14 s | **67.94%** | Accuracy begins dropping |
| **RANDOM** | 10% (100) | 0.98 s | **53.60%** | Severe degradation (-30.2%) |
| **RANDOM** | 5% (50) | 1.36 s | **25.77%** | **Catastrophic failure** — random sampling cannot survive at 5% |

**Explanation for Teacher**:
*"This empirical curve demonstrates the core motivation: Random sampling performs adequately at 50% data, but catastrophically fails at 5% and 10%. CRAIG's gradient matching is designed to preserve critical boundary samples to prevent this collapse."*

---

## 5. Live Demo Commands

1. **System & Environment Self-Check**:
   ```bash
   .venv/bin/python scripts/verify_installation.py
   ```
2. **Unit Tests (Fast Validation)**:
   ```bash
   .venv/bin/python -m unittest discover -s tests -p "test_*.py"
   ```
3. **Run Baseline Experiment**:
   ```bash
   .venv/bin/python experiments/run_mnist.py --config configs/mnist.yaml --profile debug
   ```

---

## 6. Likely Teacher Questions & Defenses

- **Q: Why start with Logistic Regression?**
  - *A: To mathematically verify the subset selection mechanics on a clean, well-understood model without confounding factors like batch normalization or stochastic dropout before scaling to CNNs.*
- **Q: How does CRAIG select samples?**
  - *A: By minimizing the upper bound on gradient approximation error via a submodular facility-location objective: $F(S) = \sum_i \max_{j \in S} (M - \|g_i - g_j\|)$, solved with a $(1 - 1/e)$-greedy approximation.*
- **Q: Does subset selection actually save time?**
  - *A: That is our Break-Even analysis (RQ6). For short 2-epoch runs, selection overhead dominates. For long multi-epoch training or hyperparameter search where the coreset is reused, CRAIG provides net speedup.*
