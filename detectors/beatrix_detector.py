"""
BeatrixDetector — adapts the Gram-matrix method from
defenses/Beatrix/Beatrix.py to the BaseDetector interface.

This does the per-class calibration itself (learns a clean profile
and a 95th-percentile threshold for every class), then scores new
samples with the same Gram-matrix deviation used in the original
Beatrix.py.
"""

import numpy as np
import torch

from detectors.base import BaseDetector
from defenses.Beatrix.Beatrix import Feature_Correlations, threshold_determine


class BeatrixDetector(BaseDetector):
    name = "beatrix"

    def __init__(self, num_classes, order_list=None, clean_data_perclass=30):
        self.num_classes = num_classes
        self.order_list = order_list if order_list is not None else np.arange(1, 9)
        self.clean_data_perclass = clean_data_perclass
        self._models = {}      # one Feature_Correlations profile per class
        self._thresholds = {}  # (p95, p99) per class

    def fit(self, clean_features, clean_labels):
        clean_labels = np.asarray(clean_labels)
        for class_id in range(self.num_classes):
            class_feats = clean_features[np.where(clean_labels == class_id)]
            class_feats = class_feats[: self.clean_data_perclass]
            if class_feats.shape[0] == 0:
                continue

            model = Feature_Correlations(POWER_list=self.order_list, mode="mad")
            p95, p99 = threshold_determine(class_feats, model)
            model.train(in_data=[class_feats])

            self._models[class_id] = model
            self._thresholds[class_id] = (p95, p99)

    def score(self, features, labels):
        labels = np.asarray(labels)
        scores = np.zeros(len(labels), dtype=float)

        for class_id, model in self._models.items():
            mask = np.where(labels == class_id)[0]
            if len(mask) == 0:
                continue
            class_feats = features[mask]
            deviations = model.get_deviations_([class_feats]).squeeze(-1)
            scores[mask] = deviations

        return scores

    def flagged(self, features, labels, percentile="p95"):
        """Convenience helper: return a boolean array, True = flagged as backdoored."""
        labels = np.asarray(labels)
        flags = np.zeros(len(labels), dtype=bool)
        idx = 0 if percentile == "p95" else 1

        for class_id, model in self._models.items():
            mask = np.where(labels == class_id)[0]
            if len(mask) == 0 or class_id not in self._thresholds:
                continue
            threshold = self._thresholds[class_id][idx]
            deviations = model.get_deviations_([features[mask]]).squeeze(-1)
            flags[mask] = deviations > threshold

        return flags
