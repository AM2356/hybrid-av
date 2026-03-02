"""
Hybrid scoring utilities: early fusion and late fusion.

The classes here are generic and can be used for any dataset where wehave:

  * A "static" feature view (e.g. 215-D Drebin-style features),
  * A "behaviour" feature view (e.g. TF-IDF of API sequences),
  * The same label space for both models.

They do NOT assume a particular malware dataset.
"""

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier


# ---------- feature concatenation for early fusion ----------

def concat_features(X_static, X_behavior):
    """
    Concatenate static and behaviour features along the last dimension.

    Works for:
      * dense numpy arrays
      * scipy sparse matrices
      * a mix of dense + sparse (converted to CSR under the hood)
    """
    if sparse.issparse(X_static) or sparse.issparse(X_behavior):
        Xs = X_static if sparse.issparse(X_static) else sparse.csr_matrix(X_static)
        Xb = X_behavior if sparse.issparse(X_behavior) else sparse.csr_matrix(X_behavior)
        return sparse.hstack([Xs, Xb]).tocsr()
    else:
        return np.hstack([X_static, X_behavior])


# late fusion (static + behaviour probs)

@dataclass
class LateFusionScorer:
    """
    Late fusion of static and behaviour classifiers.

    p_hybrid = alpha * p_static + (1 - alpha) * p_behavior

    Requirements:
      * static_model and behavior_model must implement:
            predict_proba(X) -> [n_samples, n_classes]
            have a .classes_ attribute with label order.
      * The label spaces must be the same. If class_order is provided,
        probabilities from both models are re-ordered to match it.
    """

    static_model: ClassifierMixin
    behavior_model: ClassifierMixin
    alpha: float = 0.5
    class_order: Optional[Sequence] = None  # e.g. list of family names

    def __post_init__(self):
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")

        # If user didn't specify a class order, and both models have the same
        # ordering, we can safely adopt static_model.classes_ as canonical.
        if self.class_order is None:
            if hasattr(self.static_model, "classes_") and hasattr(
                self.behavior_model, "classes_"
            ):
                if np.array_equal(
                    self.static_model.classes_, self.behavior_model.classes_
                ):
                    self.class_order = list(self.static_model.classes_)
                else:
                    # We leave class_order as None and will raise a clearer
                    # error when predict_proba is called.
                    self.class_order = None

    # internal helper

    def _proba_in_order(
        self, model: ClassifierMixin, X, class_order: Optional[Sequence]
    ) -> np.ndarray:
        """Get model.predict_proba(X) with columns re-ordered to class_order."""
        p = model.predict_proba(X)

        if class_order is None:
            # Make sure both models actually share the same class ordering.
            if hasattr(self.static_model, "classes_") and hasattr(
                self.behavior_model, "classes_"
            ):
                if not np.array_equal(
                    self.static_model.classes_, self.behavior_model.classes_
                ):
                    raise ValueError(
                        "LateFusionScorer: class_order is None and "
                        "static_model.classes_ != behavior_model.classes_.\n"
                        "Provide an explicit class_order or align the models "
                        "to the same label space before fusing."
                    )
            return p

        # Reorder columns of p to match the desired class_order.
        if not hasattr(model, "classes_"):
            raise AttributeError(
                "Model has no 'classes_' attribute; cannot align probability columns."
            )

        cls_to_idx = {cls: idx for idx, cls in enumerate(model.classes_)}
        try:
            col_indices = [cls_to_idx[c] for c in class_order]
        except KeyError as exc:
            raise ValueError(
                f"Class {exc} not found in model.classes_={model.classes_!r}"
            ) from exc

        return p[:, col_indices]

    # main API
    def predict_proba(self, X_static, X_behavior) -> np.ndarray:
        """
        Compute fused class probabilities for each sample.

        X_static and X_behavior must have the same number of rows.
        """
        if X_static.shape[0] != X_behavior.shape[0]:
            raise ValueError(
                "X_static and X_behavior must have the same number of samples "
                f"(got {X_static.shape[0]} vs {X_behavior.shape[0]})."
            )

        p_static = self._proba_in_order(self.static_model, X_static, self.class_order)
        p_beh = self._proba_in_order(self.behavior_model, X_behavior, self.class_order)

        return self.alpha * p_static + (1.0 - self.alpha) * p_beh

    def predict(self, X_static, X_behavior) -> np.ndarray:
        """
        Predict class indices/labels by argmax over fused probabilities.
        """
        p_h = self.predict_proba(X_static, X_behavior)

        # If we know the canonical class order, map indices back to labels.
        if self.class_order is not None:
            indices = p_h.argmax(axis=1)
            class_order_arr = np.array(self.class_order)
            return class_order_arr[indices]
        else:
            return p_h.argmax(axis=1)


@dataclass
class GatedLateFusionScorer(LateFusionScorer):
    """
    Late fusion with an optional gate on static confidence.

    Behaviour model is only invoked for samples where the maximum
    static probability is below static_conf_thresh. This is your
    cost-aware hybrid path from the methodology.
    """

    static_conf_thresh: float = 0.9

    def predict_proba(self, X_static, X_behavior, use_gate: bool = True) -> np.ndarray:
        if X_static.shape[0] != X_behavior.shape[0]:
            raise ValueError(
                "X_static and X_behavior must have the same number of samples "
                f"(got {X_static.shape[0]} vs {X_behavior.shape[0]})."
            )

        # Always compute static probabilities first
        p_static = self._proba_in_order(self.static_model, X_static, self.class_order)

        if not use_gate:
            # Fall back to full late fusion
            p_beh = self._proba_in_order(
                self.behavior_model, X_behavior, self.class_order
            )
            return self.alpha * p_static + (1.0 - self.alpha) * p_beh

        # Decide which samples are "uncertain" under static-only
        max_conf = p_static.max(axis=1)
        need_beh = max_conf < self.static_conf_thresh

        # Start with static-only probabilities
        p_hybrid = p_static.copy()

        # For uncertain samples, blend with behaviour probabilities
        if np.any(need_beh):
            X_beh_uncertain = (
                X_behavior[need_beh]
                if not sparse.isspmatrix(X_behavior)
                else X_behavior[need_beh, :]
            )
            p_beh_uncertain = self._proba_in_order(
                self.behavior_model, X_beh_uncertain, self.class_order
            )
            p_static_uncertain = p_static[need_beh]
            p_hybrid[need_beh] = (
                self.alpha * p_static_uncertain
                + (1.0 - self.alpha) * p_beh_uncertain
            )

        return p_hybrid


# ---------- early fusion (feature concatenation) ----------

@dataclass
class EarlyFusionModel:
    """
    Early-fusion classifier.

    Concatenates static and behaviour feature vectors, then fits
    a single base classifier (Random Forest by default).

    This matches the methodology's:
        "Early fusion: RF/XGBoost working in static || behavior mode."
    """

    base_clf: Optional[ClassifierMixin] = None

    def __post_init__(self):
        if self.base_clf is None:
            self.base_clf = RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                random_state=42,
                n_jobs=-1,
            )

    def fit(self, X_static, X_behavior, y):
        X_concat = concat_features(X_static, X_behavior)
        self.base_clf.fit(X_concat, y)
        return self

    def predict_proba(self, X_static, X_behavior):
        X_concat = concat_features(X_static, X_behavior)
        if not hasattr(self.base_clf, "predict_proba"):
            raise AttributeError(
                "Base classifier does not support predict_proba; "
                "use a probabilistic classifier for early-fusion."
            )
        return self.base_clf.predict_proba(X_concat)

    def predict(self, X_static, X_behavior):
        X_concat = concat_features(X_static, X_behavior)
        return self.base_clf.predict(X_concat)