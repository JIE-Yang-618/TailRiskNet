# Synthetic demonstration results

> These outputs use simulated data with a known contagion graph. They do not constitute
> evidence about any real financial institution or the Chinese financial system.

## Specification

- Lower-tail quantile: `0.05`
- Fixed L1 penalty: `0.01`
- Common factors removed: `1`
- Source-return lag: `1`
- Rolling windows: `29`
- Moving-block bootstrap repetitions: `40`

## Latest network

- Unfiltered mean adverse edge weight: `0.061643`
- Unfiltered active-edge density: `0.250`
- Stability-filtered active edges: `11`
- Stability-filtered density: `0.083`

Top emitters in the stability-filtered network:

| institution   |   out_strength |   net_transmitter |
|:--------------|---------------:|------------------:|
| S2            |         1.1230 |           -0.0013 |
| S1            |         0.7404 |            0.3594 |
| S3            |         0.7171 |            0.1929 |
| B2            |         0.7056 |            0.0361 |
| B1            |         0.6695 |            0.6695 |

## Known-graph recovery diagnostic

- Precision: `0.727`
- Recall: `0.889`
- Average precision: `0.841`

Recovery performance is a property of this stylized DGP and must not be read as
a guarantee for observational financial data.
