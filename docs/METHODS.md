# Methods and estimands

## 1. Data convention

Let $r_{i,t}$ be the return of institution $i$, $L$ the configured source lag, and $s_{t-1}$ a vector of predetermined state variables. The software treats negative returns as losses and focuses on a lower quantile $\tau<0.5$. All regressors are standardized within each estimation window. The default is $L=1$.

Every adjacency matrix uses

\[
A_{j,i}=\text{weight from source }j\text{ to target }i.
\]

Diagonal entries are zero.

## 2. Common-factor adjustment

With `factor_count = K`, principal components are estimated inside each rolling window:

\[
R_t = \Lambda f_t + u_t.
\]

The quantile network is estimated on residuals $u_t$. In cross-validation, PCA is fit only on the training fold and then applied to the later validation fold. This prevents validation information from entering factor estimates.

Factor adjustment changes the estimand. A raw-return network measures total conditional dependence; a residual network asks whether idiosyncratic tail dependence remains after removing pervasive contemporaneous variation.

## 3. Sparse conditional-tail network

For each target $i$, the main model estimates

\[
Q_{\tau}(r_{i,t}\mid r_{-i,t-L},r_{i,t-L},s_{t-1})
=\alpha_i+r_{-i,t-L}'\beta_i+\phi_i r_{i,t-L}+s_{t-1}'\gamma_i
\]

by minimizing pinball loss plus an L1 penalty. The penalty encourages a sparse graph when the number of peer institutions is large relative to the rolling sample.

The source stress contrast is

\[
d_{j,\tau}=q_{\tau}(r_{j,t-L})-q_{0.5}(r_{j,t-L})<0.
\]

The signed fitted response is 

\[
\Delta\widehat{CoVaR}_{j\rightarrow i}
=\widehat\beta_{j,i}d_{j,\tau}.
\]

The adverse network retains only fitted deterioration:

\[
A_{j,i}=\max\{0,-\Delta\widehat{CoVaR}_{j\rightarrow i}\}.
\]

Positive signed responses are preserved in `latest_signed_effects.csv` but are not counted as adverse spillovers.

### Penalty selection

If `alpha` is null, candidates are evaluated using expanding time-series folds and mean pinball loss across institutions. The selected penalty is fixed for every subsequent rolling window. This prevents time variation in the penalty from being mistaken for time variation in systemic risk.

## 4. Pairwise Delta-CoVaR benchmark

For each ordered pair $j\rightarrow i$, an unpenalized quantile regression estimates the target on the source and state variables. The outputs include both contemporaneous Delta-CoVaR ($L=0$) and a lag-matched predictive benchmark. Their stress contrasts use the same median-to-VaR move. Pairwise models are easy to interpret but do not jointly control for the remaining institutions, so omitted-network dependence may inflate links.

## 5. Tail co-exceedance benchmark

Define $I_{i,t}=1[r_{i,t}\le q_{\tau}(r_i)]$. The lag-matched diagnostic weight is

\[
U_{j\rightarrow i}
=\max\{0,\Pr(I_{i,t}=1\mid I_{j,t-L}=1)-\Pr(I_{i,t}=1)\}.
\]

Jeffreys smoothing avoids zero probabilities in small tail samples. This measure is distribution-free but provides weak directionality when source and target event counts are similar; it is used as a robustness diagnostic, not as a causal estimator.

## 6. Moving-block bootstrap stability

Each final-window panel is resampled in circular blocks, preserving short-run serial and cross-sectional dependence. The model is re-estimated for each draw. For every ordered edge the software reports:

- selection probability;
- median adverse weight;
- 10th and 90th percentiles of the bootstrap weight.

The default stable graph retains edges with selection probability above a configured threshold. These are stability diagnostics, not simultaneous confidence intervals or formal family-wise-error control.

## 7. Network summaries

For a non-negative adjacency matrix:

\[
\text{out-strength}_j=\sum_i A_{j,i},\qquad
\text{in-strength}_i=\sum_j A_{j,i}.
\]

Net transmitter status is out-strength minus in-strength. Size-adjusted measures use market-cap shares, not raw currency units:

\[
SI^{out}_j=m_j\sum_i A_{j,i}m_i.
\]

This keeps scale interpretable and allows network-only rankings to be inspected separately from size.

Sector tables report both total weight and mean weight across all possible non-self directed exposures. The mean is the primary cross-sector comparison when sector sample sizes differ.

## 8. Identification limits

Lagged quantile regressions encode predictive conditional dependence, not structural causality. Temporal ordering makes the direction more interpretable than in a purely contemporaneous regression, but correlated omitted shocks, measurement error, market microstructure, and balance-sheet links can produce the same statistical pattern. Causal language requires an external identification strategy or structural exposure data.

The current release does not implement the thesis's exact single-index MACE-SCAD estimator, a QVAR forecast-error variance decomposition, SRISK, or a balance-sheet clearing cascade. Those are distinct estimands and should not be silently combined.
