"""
EnsembleDetector — combines scores from one or more backdoor detectors.

Right now this only runs Beatrix. To add another detector later
(STRIP, Neural Cleanse, Activation Clustering, ...):

  1. Write detectors/your_detector.py implementing BaseDetector
     (see detectors/base.py for the interface).
  2. Import it below and add it to the `detectors` dict in __init__,
     with a starting weight.

Nothing else needs to change — fit(), score(), and flagged() already
loop over whatever is in self.detectors.
"""

import numpy as np

from detectors.beatrix_detector import BeatrixDetector


class EnsembleDetector:
    def __init__(self, num_classes):
        self.detectors = {
            "beatrix": BeatrixDetector(num_classes=num_classes),
            # "strip": StripDetector(...),          # add later
            # "neural_cleanse": NeuralCleanseDetector(...),  # add later
        }
        # Equal weights for now. With more detectors, this is where an
        # "adaptive" scheme (e.g. weight by historical precision) would live.
        self.weights = {name: 1.0 for name in self.detectors}

    def fit(self, clean_features, clean_labels):
        for detector in self.detectors.values():
            detector.fit(clean_features, clean_labels)

    def score(self, features, labels):
        """Weighted average of each detector's per-sample anomaly score."""
        total_weight = sum(self.weights.values())
        combined = np.zeros(len(labels), dtype=float)

        for name, detector in self.detectors.items():
            raw_scores = detector.score(features, labels)
            normalized = self._normalize(raw_scores)
            combined += self.weights[name] * normalized

        return combined / total_weight

    def flagged(self, features, labels, percentile="p95"):
        """
        A sample is flagged if ANY detector flags it (logical OR).
        With more detectors, this is a natural place to switch to a
        majority vote instead.
        """
        flags = None
        for detector in self.detectors.values():
            detector_flags = detector.flagged(features, labels, percentile=percentile)
            flags = detector_flags if flags is None else (flags | detector_flags)
        return flags

    @staticmethod
    def _normalize(scores):
        """Min-max normalize so detectors with different score scales are comparable."""
        scores = np.asarray(scores, dtype=float)
        lo, hi = scores.min(), scores.max()
        if hi - lo < 1e-12:
            return np.zeros_like(scores)
        return (scores - lo) / (hi - lo)
