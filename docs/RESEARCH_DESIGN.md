# Research design

## Research origin

The motivating undergraduate thesis studied 34 listed Chinese financial institutions - 16 banks, 14 securities firms, and 4 insurers - using weekly observations, VaR/CoVaR estimation, single-index quantile regression, and rolling tail-event networks. It asked which institutions and sectors emitted or absorbed systemic tail risk during the 2015 market dislocation, the 2018 China-US trade conflict, and the 2020 pandemic shock.

TailRiskNet does not simply translate the thesis scripts. It reconstructs the research problem as a reusable and testable software package, and it separates claims that can be supported with public synthetic evidence from those requiring licensed Chinese financial data.

## Questions

1. Does lower-tail connectedness rise in stress periods after conditioning on observable states?
2. Which institutions are persistent risk emitters or receivers rather than one-window outliers?
3. How much apparent connectedness disappears after removing common contemporaneous factors?
4. Are sectoral differences robust after normalizing for the number of possible directed exposures?
5. Do rankings survive alternative estimators and block-bootstrap resampling?
6. Can the method recover directional edges in data for which the contagion graph is known?

## Core specification

The main estimator is a system of target-by-target L1-penalized quantile regressions. For each rolling window, the returns may first be residualized on a fixed number of PCA factors. Each target's lower conditional quantile is then modeled using lagged peer returns, its own lag, and predetermined state variables. The default one-period lag gives arrows a temporal order that a purely contemporaneous network cannot supply.

The penalty is either specified ex ante or selected once on the first calibration window by expanding-window time-series cross-validation. It is not re-tuned in each evaluation window. This deliberately prioritizes comparability of edge weights through time over local fit.

## Robustness layers

The repository makes four comparisons available from the same validated panel:

- joint lagged sparse quantile network versus contemporaneous Delta-CoVaR and lagged pairwise quantile benchmarks;
- conditional quantile weights versus nonparametric lagged tail co-exceedance;
- raw returns versus common-factor-adjusted returns;
- unfiltered edges versus block-bootstrap stable edges.

Agreement is summarized by off-diagonal rank correlation and top-edge Jaccard overlap. Disagreement is itself a result: it shows that the substantive conclusion depends on the estimand or conditioning set.

## Synthetic validation

The synthetic panel contains three sectors, heavy-tailed innovations, a common market factor, sector factors, short-run persistence, and a directed acyclic tail-contagion graph. Its true edges are stored separately. The pipeline reports precision, recall, and average precision after stability filtering.

This exercise is not evidence that the estimator is universally consistent. It is a falsifiable software check: the method must at least detect known asymmetric tail links in a controlled setting while facing realistic nuisance comovement.

## Empirical hypotheses for the Chinese application

The code supports, but does not hard-code, the following hypotheses:

- lower-tail connectedness is larger during market stress than in tranquil periods;
- banks and securities firms have stronger emitter and receiver roles than insurers;
- within-sector spillovers exceed cross-sector spillovers;
- institution rankings change when size and connectedness are reported separately;
- some crisis-network edges remain unstable after dependence-aware resampling.

These are empirical claims. They should be evaluated only after the licensed inputs, timing conventions, and event definitions have been independently verified.
