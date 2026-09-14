# Model artifact versioning and rollback

## Context
`train.py` writes two things on every run: a LightGBM model file
(`models/demand_forecast_v{n}.txt`) and a metadata contract
(`models/metrics.json`) that the forecasting API (step 1.10) will read at
startup via `forecasting-service/src/artifact.py`.

Model files are reproducible from data + code, so the original decision was
to keep `models/*.txt` gitignored. Metadata is small, human-readable, and is
the only durable record of what a given model version was, so
`models/metrics.json` (current) and `models/history/metrics_v{n}.json`
(archived on each retrain) stay tracked in git.

That original decision is revised below (2026-09-14): the artifact is now
tracked too, specifically to close the rollback gap described in the
original Consequence section.

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
4. (2026-09-14) `models/*.txt` is removed from `.gitignore` and tracked in
   git alongside `metrics.json`. At the current size -- one LightGBM text
   model at ~322 KB -- this is well within what git handles comfortably, and
   it means every tracked version's contract (`metrics_v{n}.json`) now has
   its actual artifact available in the same clone, closing the rollback gap
   in the original Consequence below. This is also what makes the Docker
   image self-contained: `forecasting_service/Dockerfile` copies `models/`
   directly from the build context, with no separate artifact-fetch step.

   This does not scale indefinitely. Once a single artifact or the
   accumulated history of versions gets into the tens of MB, git's
   per-object storage and full-history clone cost start to hurt -- a rough
   line is somewhere around 50-100 MB per artifact, well before git
   technically refuses anything. Past that point this decision should be
   revisited in favor of Git LFS (keeps the tracked-alongside-code workflow,
   stores blobs externally) or the object-storage/model-registry approach
   already named as V2 scope below, not by continuing to grow the
   plain-git repository.

## Consequence
Rollback restores the *active* model on any clone, not just a machine that
happened to keep the old `.txt` file locally -- that gap is closed as of the
decision above. `models/history/metrics_v{n}.json` continues to preserve
the contract -- what each version's metrics, features, and training data
were -- independently of the artifact, which matters for the versions that
predate this decision and for any future point where artifact tracking is
migrated to LFS or object storage and history stops moving with the repo by
default. A durable artifact store (object storage or a model registry) that
keeps both together, and that scales past the size ceiling noted above, is
V2 scope.
