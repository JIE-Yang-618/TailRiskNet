# References and methodological provenance

## Core research

- Adrian, T., & Brunnermeier, M. K. (2016). [CoVaR](https://doi.org/10.1257/aer.20120555). *American Economic Review*, 106(7), 1705-1741.
- Hardle, W. K., Wang, W., & Yu, L. (2016). [TENET: Tail-Event driven NETwork risk](https://doi.org/10.1016/j.jeconom.2016.02.013). *Journal of Econometrics*, 192(2), 499-513.
- Fan, Y., Hardle, W. K., Wang, W., & Zhu, L. (2018). [Single-index-based CoVaR with very high-dimensional covariates](https://doi.org/10.1080/07350015.2016.1180990). *Journal of Business & Economic Statistics*, 36(2), 212-226.
- Diebold, F. X., & Yilmaz, K. (2014). [On the network topology of variance decompositions](https://doi.org/10.1016/j.jeconom.2014.04.012). *Journal of Econometrics*, 182(1), 119-134.
- Ando, T., Greenwood-Nimmo, M., & Shin, Y. (2022). [Quantile connectedness: Modeling tail behavior in the topology of financial networks](https://doi.org/10.1287/mnsc.2021.3984). *Management Science*, 68(4), 2401-2431.
- Belloni, A., & Chernozhukov, V. (2011). [L1-penalized quantile regression in high-dimensional sparse models](https://doi.org/10.1214/10-AOS827). *Annals of Statistics*, 39(1), 82-130.
- Kunsch, H. R. (1989). [The jackknife and the bootstrap for general stationary observations](https://doi.org/10.1214/aos/1176347265). *Annals of Statistics*, 17(3), 1217-1241.

## Replication and software resources reviewed

- Adrian and Brunnermeier's [AEA Data and Code Repository record](https://www.openicpsr.org/openicpsr/project/112877/version/V1/view) documents the original CoVaR replication materials.
- The authors' [QuantLet/TENET repository](https://github.com/QuantLet/TENET) provides the original TENET research code and data organization.
- Scikit-learn's [`QuantileRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.QuantileRegressor.html) supplies the L1-penalized linear quantile solver used here.
- NetworkX's [directed-network algorithms](https://networkx.org/documentation/stable/reference/algorithms/index.html) support graph representation and visualization.

TailRiskNet is an independent implementation. It does not copy code from these repositories and does not claim equivalence to the exact SIM/MACE-SCAD or QVAR procedures in the cited papers.

