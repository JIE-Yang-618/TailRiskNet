# Data schema

TailRiskNet uses explicit CSV contracts. Dates must be unique, increasing, and aligned across files. Returns should be decimal log or simple returns used consistently across all institutions.

## `returns.csv`

Wide panel with one institution per column.

```csv
date,B1,B2,S1,S2,I1,I2
2020-01-03,-0.012,0.003,-0.021,-0.004,0.001,-0.006
```

Requirements:

- `date` is parseable by pandas;
- at least three institution columns;
- numeric finite values;
- no institution has more than 10% missing data;
- after validation, rows with incomplete observations are dropped rather than imputed.

## `states.csv`

Optional wide panel of predetermined controls. State variables should be lagged before they enter this file when contemporaneous values would create look-ahead or simultaneity concerns.

```csv
date,market_return_lag1,market_abs_4w,liquidity_state_lag1
2020-01-03,0.002,0.011,-0.130
```

The default thesis application may include market returns and volatility, liquidity, term and credit spreads, exchange rates, monetary growth, and global rates. Variable timing should be documented at the source level; a `_lag1` suffix is recommended when applicable.

## `institutions.csv`

```csv
institution,sector,display_name,market_cap
B1,Bank,Example Bank,125.4
```

`institution` must exactly match a returns column. `market_cap` must be positive and use one common unit and reference date. The package converts it to a share before size-adjusted metrics are calculated.

## `true_edges.csv`

Optional and intended for simulations only.

```csv
source,target,weight
B1,B2,0.9
```

Real observational applications normally do not have this file because the true contagion graph is unknown.

## Licensed data

The original application used Wind and firm-level accounting information. Those files are not redistributable through this repository. Store them under `data/raw/`, which is ignored by Git, and record vendor definitions, extraction date, corporate-action handling, delistings, and the market-cap reference date in a local data log.

