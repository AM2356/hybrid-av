# src/hybrid_scorer.py

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from joblib import load

from src.hybrid import GatedLateFusionScorer

# Resolve project + model dirs
PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_DIR / "saved" / "models"


class HybridCAPEFamilyScorer:
    """
    Tiny design-level hybrid scoring module.
    It loads:
      -cape_tfidf_cheap.pkl
      -cape_tfidf_full.pkl
      -cape_rf_cheap.pkl
      -cape_rf_full.pkl
and exposes a score_text(api_text) method."""

    def __init__(
        self,
        alpha: float = 0.0,
        static_conf_thresh: float = 0.9,
    ):
        # Load artefacts saved from 03_hybrid_scoring.ipynb
        self.tfidf_cheap = load(MODEL_DIR / "cape_tfidf_cheap.pkl")
        self.tfidf_full = load(MODEL_DIR / "cape_tfidf_full.pkl")
        self.rf_cheap = load(MODEL_DIR / "cape_rf_cheap.pkl")
        self.rf_full = load(MODEL_DIR / "cape_rf_full.pkl")

        # Class order for probabilities (string family labels)
        self.class_order = list(self.rf_full.classes_)

        # Gated late-fusion scorer
        self.gated = GatedLateFusionScorer(
            static_model=self.rf_cheap,
            behavior_model=self.rf_full,
            alpha=alpha,
            class_order=self.class_order,
            static_conf_thresh=static_conf_thresh,
        )

    def score_text(self, api_text: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Given a space-separated API token string (like cape_df.api_text_full),
        return (predicted_family, confidence, proba_dict).

        proba_dict maps family_name -> probability.
        """
        # Transform into both views
        Xs = self.tfidf_cheap.transform([api_text])
        Xb = self.tfidf_full.transform([api_text])

        # Get fused probabilities using the gated scorer
        proba = self.gated.predict_proba(Xs, Xb)[0]
        probs = np.asarray(proba, dtype=float)

        classes = np.array(self.class_order)
        best_idx = int(probs.argmax())
        best_class = str(classes[best_idx])
        best_conf = float(probs[best_idx])

        proba_dict = {cls: float(p) for cls, p in zip(classes, probs)}

        return best_class, best_conf, proba_dict