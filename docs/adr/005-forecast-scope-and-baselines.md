# Forecast scope and baseline selection

## Context
Top-50 SKUs vary widely in regularity. Sampling three high-CV products:
84347 sells 72% of weeks (CV 3.61), 16014 56% (CV 2.55), 23084 29% (CV 3.43).
Volatility and sparsity are distinct: the most volatile product is also one
of the most regular.

Baselines on the 52-week test period:
| Baseline | WAPE | MAPE |
|---|---|---|
| Naive (last week) | 75.5% | 256.2% |
| Seasonal naive (same week last year) | 92.4% | 226.2% |
| 4-week moving average | **67.1%** | 211.5% |

## Decision
1. Scope boundary: products selling in <25% of weeks are out of scope
   (intermittent demand needs Croston's/TSB, a different problem). No current
   product is excluded by this rule; it is stated in advance rather than tuned.
2. The 4-week moving average is the baseline to beat, not the weaker naive.
3. WAPE is the headline metric. MAPE is reported but exceeds 200% on every
   baseline; per-bucket diagnostics in scripts/diagnose_mape.py show both WAPE
   and MAPE inflate on the same erratic rows, so this is a data property, not
   a metric artifact.

## Consequence
The published improvement is measured against the strongest reasonable
baseline, and includes the products the model finds hardest.