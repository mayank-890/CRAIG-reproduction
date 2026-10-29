# Mathematical Background: CRAIG Data Subset Selection

This document provides the formal mathematical foundation for Gradient-Matching Coreset Selection (CRAIG), based on:

> Mirzasoleiman, B., Bilmes, J., & Leskovec, J. (2020).  
> **"Coresets for Data-Efficient Training of Machine Learning Models."** ICML 2020.

---

## 1. Supervised Learning Objective & Full Gradient

Consider a dataset $V = \{(x_i, y_i)\}_{i=1}^N$ with $N$ samples, and a model parameterized by $\theta \in \mathbb{R}^D$.  
The empirical risk minimization (ERM) objective is:

$$L(\theta) = \frac{1}{N} \sum_{i \in V} \ell(f(x_i; \theta), y_i) = \frac{1}{N} \sum_{i \in V} \ell_i(\theta)$$

where $\ell_i(\theta)$ is the loss of sample $i$.  
The full-data gradient across all $N$ training examples is:

$$G_V(\theta) = \nabla_\theta L(\theta) = \frac{1}{N} \sum_{i \in V} \nabla_\theta \ell_i(\theta) = \frac{1}{N} \sum_{i \in V} g_i(\theta)$$

where $g_i(\theta) = \nabla_\theta \ell_i(\theta) \in \mathbb{R}^D$ denotes the per-sample gradient.

---

## 2. Coreset & Gradient-Matching Formulation

A coreset $S \subseteq V$ of size $k \ll N$, with positive coreset weights $w_j \ge 0$ for each representative $j \in S$, aims to approximate the full training gradient:

$$G_S(\theta) = \frac{1}{N} \sum_{j \in S} w_j g_j(\theta)$$

The gradient approximation error is measured as the Euclidean norm difference:

$$\mathcal{E}(S, w) = \|G_V(\theta) - G_S(\theta)\| = \left\| \frac{1}{N} \sum_{i \in V} g_i(\theta) - \frac{1}{N} \sum_{j \in S} w_j g_j(\theta) \right\|$$

To minimize this error, suppose we partition $V$ into disjoint clusters $\{C_j\}_{j \in S}$, where each cluster $C_j$ is represented by element $j \in S$, and set:

$$w_j = |C_j|$$

such that $\sum_{j \in S} w_j = \sum_{j \in S} |C_j| = N$.

Applying the triangle inequality:

$$\mathcal{E}(S, w) = \frac{1}{N} \left\| \sum_{j \in S} \sum_{i \in C_j} (g_i(\theta) - g_j(\theta)) \right\| \le \frac{1}{N} \sum_{j \in S} \sum_{i \in C_j} \|g_i(\theta) - g_j(\theta)\|$$

---

## 3. Submodular Facility Location Formulation

To minimize the upper bound $\sum_{j \in S} \sum_{i \in C_j} \|g_i(\theta) - g_j(\theta)\|$, each sample $i \in V$ should be assigned to its closest representative in $S$:

$$\min_{S \subseteq V, |S| \le k} \sum_{i \in V} \min_{j \in S} \|g_i(\theta) - g_j(\theta)\|$$

Let $d(i, j) = \|g_i(\theta) - g_j(\theta)\|$ be the pairwise gradient distance. Defining pairwise similarity with respect to an upper bound $M \ge \max_{i, j} d(i, j)$:

$$s(i, j) = M - d(i, j)$$

The minimization problem transforms into the monotone submodular **Facility Location** objective:

$$F(S) = \sum_{i \in V} \max_{j \in S} s(i, j)$$

Because $F(S)$ is monotone and submodular, the greedy algorithm achieves a $(1 - 1/e) \approx 63.2\%$ approximation to the optimal subset selection!

---

## 4. Greedy Selection Algorithm

1. Initialize $S_0 = \emptyset$.
2. For iteration $t = 1, \dots, k$:
   - For each candidate $c \in V \setminus S_{t-1}$, compute the marginal gain:
     $$\Delta(c \mid S_{t-1}) = F(S_{t-1} \cup \{c\}) - F(S_{t-1}) = \sum_{i \in V} \max(0, s(i, c) - \max_{j \in S_{t-1}} s(i, j))$$
   - Select candidate with maximum marginal gain:
     $$j^* = \arg\max_{c \in V \setminus S_{t-1}} \Delta(c \mid S_{t-1})$$
   - Update: $S_t = S_{t-1} \cup \{j^*\}$.
3. Assign each element $i \in V$ to cluster $C_{j^*(i)}$ where $j^*(i) = \arg\max_{j \in S} s(i, j)$.
4. Coreset weights: $w_j = |C_j|$, ensuring $\sum_{j \in S} w_j = N$.

---

## 5. Relative Gradient Approximation Error

To empirically assess how well the subset gradient approximates the full dataset gradient:

$$\text{Relative Error} = \frac{\|G_V(\theta) - G_S(\theta)\|_2}{\|G_V(\theta)\|_2}$$

$$\text{Cosine Similarity} = \frac{G_V(\theta) \cdot G_S(\theta)}{\|G_V(\theta)\|_2 \|G_S(\theta)\|_2}$$

---

## 6. Computational Trade-Off & Break-Even Analysis

Let:
- $T_{\text{full}}$ = training time on the full dataset $V$ ($N$ samples for $E$ epochs).
- $T_{\text{selection}} = T_{\text{grad}} + T_{\text{sim}} + T_{\text{greedy}}$.
- $T_{\text{subset}}$ = training time on coreset $S$ ($k$ samples for $E$ epochs).

The total time for CRAIG is:

$$T_{\text{total\_CRAIG}} = T_{\text{selection}} + T_{\text{subset}}$$

**True Speedup**:
$$\text{Speedup}_{\text{total}} = \frac{T_{\text{full}}}{T_{\text{total\_CRAIG}}}$$

**Training-Only Speedup**:
$$\text{Speedup}_{\text{train}} = \frac{T_{\text{full}}}{T_{\text{subset}}}$$

A true computational win occurs **if and only if**:

$$T_{\text{selection}} + T_{\text{subset}} < T_{\text{full}} \iff T_{\text{selection}} < T_{\text{full}} - T_{\text{subset}}$$

This break-even condition is central to the experimental evaluation.
