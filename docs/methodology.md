# Experimental Methodology

## 1. Research Questions & Hypotheses

- **RQ1**: Can CRAIG select a substantially smaller subset while preserving model accuracy?
  - *Hypothesis*: By matching the full gradient direction, CRAIG subsets will outperform uniform random sampling, particularly at aggressive subset ratios (5%, 10%, 20%).
- **RQ2**: How does subset size affect gradient approximation error?
  - *Hypothesis*: Relative gradient error will decrease monotonically as subset size $k$ increases, with cosine similarity approaching 1.0.
- **RQ3**: How does CRAIG compare with random subset selection at equivalent subset sizes?
  - *Hypothesis*: CRAIG achieves higher accuracy and lower variance across random seeds than random sampling at identical computational subset budgets.
- **RQ4**: Does gradient-matching subset selection remain useful for time-series classification?
  - *Hypothesis*: 1D CNN representations capture temporal patterns whose gradient space enables effective representative sequence selection.
- **RQ5**: How does CRAIG selection cost scale with dataset size, model complexity, and sequence length?
  - *Hypothesis*: Selection cost scales quadratically $\mathcal{O}(N^2)$ with dataset size, and linearly with gradient dimension $D$.
- **RQ6**: When does $T_{\text{selection}} + T_{\text{subset\_training}} < T_{\text{full\_training}}$?
  - *Hypothesis*: For single-run training with modest epoch counts, selection overhead will exceed training savings. For hyperparameter search or long training regimes, CRAIG will deliver a net end-to-end speedup.

## 2. Experimental Rigor & Reproducibility Protocol

1. **Seeds**: All major experiments are run across 5 distinct random seeds (`42, 123, 456, 789, 1000`).
2. **Hyperparameter Invariance**: Exactly identical optimizer configurations, learning rates, batch sizes, weight decays, and epochs are maintained across FULL, RANDOM, and CRAIG.
3. **Dual Speedup Reporting**:
   - `training_only_speedup = T_full / T_subset_training`
   - `end_to_end_speedup = T_full / (T_selection + T_subset_training)`
   Both metrics must be reported to avoid misleading claims.
4. **Data Reduction**: Explicitly report the fraction of training examples used ($k / N$).
5. **No Data Fabrication**: Results are produced solely through actual programmatic execution and loaded directly from verified CSV output files.
