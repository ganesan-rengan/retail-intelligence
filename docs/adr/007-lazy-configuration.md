# Lazy configuration

## Context
Two modules evaluated configuration at import time, not at first use:

1. `forecasting_service/app/config.py` -- `settings = Settings()` at module
   scope. `database_url` is a required field with no default, so
   constructing `Settings()` fails immediately if `DATABASE_URL` isn't set.
2. `shared/database.py` -- `DATABASE_URL = os.environ["DATABASE_URL"]` and
   `engine = create_engine(DATABASE_URL, ...)`, both at module scope.

Both modules are reached transitively from `tests/conftest.py`
(`from forecasting_service.app import main, service`, and `main.py`'s own
`from shared.database import get_session`), which every test in the suite
imports -- not just tests that touch a database. That meant importing the
app, and therefore running the suite at all, required a real `DATABASE_URL`
to be set, even though 45 of the 46 tests never open a connection; the one
that does is a dedicated integration test, marked and skipped independently
when no real database is reachable.

Found by running `uv run pytest -m "not integration"` with no `.env` present
and no `DATABASE_URL` in the environment -- the exact condition CI runs
under -- while setting up `.github/workflows/ci.yml`. Both modules produced
a full traceback at collection time: `pydantic.ValidationError` from
`Settings()`, then, once that was fixed, `KeyError: 'DATABASE_URL'` from
`shared/database.py`.

A third eager read exists in `migrations/env.py`, but that's Alembic's own
CLI entrypoint -- never imported by the app or the test suite. Requiring a
real `DATABASE_URL` there is correct, not the same defect, and was left
unchanged.

## Decision
Replace eager, import-time construction with cached, on-first-use accessors
in both modules:

- `forecasting_service/app/config.py`: `settings = Settings()` at import
  time becomes `get_settings()`, decorated with `functools.lru_cache`. Its
  one call site in `main.py` moved from module scope into `lifespan()`, so
  it now runs at actual app startup, not at import.
- `shared/database.py`: the module-level `DATABASE_URL` read and
  `create_engine(...)` call become `get_engine()`, also `lru_cache`'d, so
  every caller still shares exactly one engine and its connection pool,
  same as before. `get_session()` builds its session off `get_engine()`.

The loud failure is preserved -- it just moves from import time to first
use. `Settings()` still raises `ValidationError` naming the missing field,
and `os.environ["DATABASE_URL"]` still raises `KeyError`, whenever something
actually needs a setting or a connection. Verified directly: starting
`uvicorn` with no `DATABASE_URL` set fails during application startup
(inside `lifespan`, before "Uvicorn running on..." is logged), not on the
first incoming request.

Every call site was updated to match: `forecasting_service/src/demand.py`
and `scripts/verify_data.py` (`from shared.database import engine` ->
`get_engine()`), and `forecasting_service/app/main.py` (`settings` ->
`get_settings()`). `get_session()`'s call sites needed no changes -- it was
already called lazily wherever it was used.

One test-level gap surfaced alongside this. Three tests in
`tests/test_api.py` didn't override the `get_db` dependency, on the
assumption that FastAPI wouldn't resolve it for a request whose handler
never queries the database. That assumption was already false -- FastAPI
resolves every route dependency on every request regardless of whether the
handler body uses it -- it just never surfaced, because a real
`DATABASE_URL` had always been present in every environment those tests had
previously run in. Fixed by adding the same `dependency_overrides[main.get_db]`
pattern every other database-touching test in the file already uses.

## Consequence
`uv run pytest -m "not integration"` now passes with zero configuration: no
`.env`, no environment variables, on a bare fresh clone. That is what makes
`.github/workflows/ci.yml` possible with no secrets and no database. Running
the real app is unaffected: `uvicorn` still requires a working
`DATABASE_URL` and still fails loudly, at startup, when one isn't set --
laziness changed *when* configuration is evaluated, not whether a missing
value is tolerated.
