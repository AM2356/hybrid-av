"""
Utilities for loading baseline models and vectorisers.
This assumes therepo structure is:
    hybrid-av/
      saved/models/
        drebin_static_rf_best.pkl
        drebin_static_xgb_best.pkl
        cape_behavior_rf_best.pkl
        cape_behavior_tfidf.pkl
and that this file lives in `hybrid-av/src/artifacts.py`.
"""

from pathlib import Path
from joblib import load


# Resolve repo root as "two levels up from this file".
# src/artifacts.py to src/ to hybrid-av/
ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "saved" / "models"


def _ensure_exists(path: Path) -> Path:
    """Nice error if a required artefact is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Expected artefact does not exist: {path}")
    return path


def load_static_rf():
    """Load the best Drebin Random Forest model (drebin_static_rf_best.pkl).
    Trained on 215-D static features with labels 'B'/'S'."""
    path = _ensure_exists(MODEL_DIR / "drebin_static_rf_best.pkl")
    return load(path)


def load_static_xgb():
    """Load the best Drebin XGBoost model (drebin_static_xgb_best.pkl).
    Trained on 215-D static features with encoded labels 0/1.
    """
    path = _ensure_exists(MODEL_DIR / "drebin_static_xgb_best.pkl")
    return load(path)


def load_behavior_rf():
    """
    Load the best CAPE behaviour Random Forest model
    (cape_behavior_rf_best.pkl).
    """
    path = _ensure_exists(MODEL_DIR / "cape_behavior_rf_best.pkl")
    return load(path)


def load_behavior_tfidf():
    """
    Load the CAPE behaviour TF-IDF vectoriser
    (cape_behavior_tfidf.pkl).
    """
    path = _ensure_exists(MODEL_DIR / "cape_behavior_tfidf.pkl")
    return load(path)


def load_behavior_model_and_vec():
    """
    Convenience helper: returns (rf_model, tfidf_vectoriser)
    for the behaviour-only baseline.
    """
    return load_behavior_rf(), load_behavior_tfidf()