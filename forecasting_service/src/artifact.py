"""Load the current trained model and its metrics contract.

The forecasting API (step 1.10) imports this instead of reading models/
directly, so a missing artifact fails loudly here -- at import/startup time
-- rather than surfacing as an obscure error on the first prediction request.
"""
import json
from pathlib import Path

import lightgbm as lgb

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
METRICS_PATH = MODEL_DIR / "metrics.json"

TRAIN_COMMAND = "uv run python forecasting-service/src/train.py"


def load_metrics() -> dict:
    """Read the current model's metadata contract."""
    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            f"{METRICS_PATH} not found. Run: {TRAIN_COMMAND}"
        )
    return json.loads(METRICS_PATH.read_text())


def load_model() -> lgb.Booster:
    """Load the current model artifact named by metrics.json's model_file.

    models/*.txt is gitignored, so this is expected to fail on a fresh
    clone even though metrics.json (tracked) is present -- that's exactly
    why this check exists, so it fails here rather than on first request.
    """
    metrics = load_metrics()
    model_path = MODEL_DIR / metrics["model_file"]
    if not model_path.exists():
        raise FileNotFoundError(
            f"{METRICS_PATH} names model_file '{metrics['model_file']}', but "
            f"{model_path} does not exist (models/*.txt is gitignored, so "
            f"this is expected after a fresh clone). Run: {TRAIN_COMMAND}"
        )
    return lgb.Booster(model_file=str(model_path))
