"""
Common interface for backdoor detectors.

Every detector in this project (Beatrix now, others later — STRIP,
Neural Cleanse, Activation Clustering, ...) should subclass this so
ensemble.py can call them all the same way without knowing their
internals.
"""

from abc import ABC, abstractmethod


class BaseDetector(ABC):
    """Minimal contract a detector must satisfy to plug into the ensemble."""

    name = "base"

    @abstractmethod
    def fit(self, clean_features, clean_labels):
        """
        Learn what 'normal' looks like from clean (trusted) data.

        clean_features: torch.Tensor of intermediate-layer activations
                         for known-clean samples, shape (N, C, H, W)
        clean_labels:   torch.Tensor or np.ndarray of shape (N,) with
                         the class label for each sample
        """
        raise NotImplementedError

    @abstractmethod
    def score(self, features, labels):
        """
        Return a per-sample anomaly score (higher = more suspicious).

        features: torch.Tensor, shape (N, C, H, W) — same layer as fit()
        labels:   predicted or true class for each sample, shape (N,)

        Returns: np.ndarray of shape (N,) with one float score per sample.
        """
        raise NotImplementedError
