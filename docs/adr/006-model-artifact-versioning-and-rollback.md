# Model artifact versioning and rollback

## Context
`train.py` writes two things on every run: a LightGBM model file
(`models/demand_forecast_v{n}.txt`) and a metadata contract
(`models/metrics.json`) that the forecasting API (step 1.10) will read at
startup via `forecasting-service/src/artifact.py`.

Model files are large, binary, and reproducible from data + code, so
`models/*.txt` stays gitignored. Metadata is small, human-readable, and is
the only durable record of what a given model version was, so
`models/metrics.json` (current) and `models/history/metrics_v{n}.json`
(archived on each retrain) stay tracked in git.

The version number itself is derived only from tracked sources -- the
current `metrics.json`'s `model_version` field and the archived
`models/history/metrics_v*.json` files -- never from scanning
`models/*.txt` on disk. A fresh clone or a teammate's machine may have zero
`.txt` files present even though git history already records v7; numbering
from the gitignored artifact would silently restart at v1 and collide with
that tracked history.

## Decision
1. Every retrain increments the version (`max(tracked versions) + 1`),
   writes a new `.txt` artifact, and never overwrites or deletes an older
   one.
2. Before writing the new `metrics.json`, the current one is copied to
   `models/history/metrics_v{n}.json` under its own recorded version
   number (not assumed to be `n-1`, in case a version was ever skipped).
3. Concurrent retrains and a mid-write crash are accepted, unhandled risks:
   the version read-then-write is not atomic, and a crash between writing
   the `.txt`, archiving history, and writing the new `metrics.json` could
   leave an orphaned artifact or a skipped version number. Acceptable for a
   local, single-operator training script; revisit if this ever runs
   concurrently or unattended.

## Consequence
Model artifacts are gitignored, so rollback restores the *active* model
only on a machine that still holds the target version's `.txt` file.
`models/history/metrics_v{n}.json` always preserves the contract -- what
that version's metrics, features, and training data were -- regardless of
whether the artifact itself is still present locally. A durable artifact
store (object storage or a model registry) that keeps both together is V2
scope.
