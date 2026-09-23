"""
detectors/beatrix_detector.py

Core Beatrix representation-level backdoor detector based on:
    W. Ma, D. Wang, R. Sun, M. Xue, S. Wen, and Y. Xiang,
    "The 'Beatrix' Resurrections: Robust Backdoor Detection via Gram Matrices,"
    Network and Distributed System Security Symposium (NDSS), 2023.

Key Capabilities:
    1. Higher-order Gram matrix formulation: G_p = (v^p (v^p)^T)^(1/p)
    2. Upper-triangular feature extraction with spatial and channel stabilization
    3. Class-conditional clean reference profiles (Median + MAD)
    4. Two-sided deviation scoring for individual sample anomaly detection
    5. Model-level Anomaly Index (J*) via Median Absolute Deviation (MAD) across classes
    6. Identification of the suspected infected target class
    7. Standardized confidence score in [0, 1] for Adaptive Ensemble Fusion (Module 5)
    8. Full compliance with the BaseDetector interface for ensemble integration
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from detectors.base import BaseDetector


@dataclass
class BeatrixResult:
    """Complete results from the Beatrix detection run."""
    is_backdoored: bool
    decision: str
    score: float
    suspected_target_class: Optional[int]
    anomaly_index: float
    class_anomaly_indices: Dict[int, float]
    scores: np.ndarray
    flags: np.ndarray
    backdoor_percentage: float
    threshold: float
    predictions: np.ndarray
    true_labels: np.ndarray
    metrics: dict = field(default_factory=dict)


def compute_gaussian_kernel(x1: torch.Tensor, x2: torch.Tensor, kernel_mul: float = 2.0, kernel_num: int = 5) -> torch.Tensor:
    """Computes a multi-scale Gaussian RBF kernel matrix between x1 and x2."""
    n1 = x1.shape[0]
    n2 = x2.shape[0]
    dim = x1.shape[1]

    tile_x1 = x1.unsqueeze(1).expand(n1, n2, dim)
    tile_x2 = x2.unsqueeze(0).expand(n1, n2, dim)

    dist_sq = torch.sum((tile_x1 - tile_x2) ** 2, dim=-1)

    # Median heuristic for bandwidth
    bandwidth = torch.median(dist_sq.view(-1))
    if bandwidth < 1e-8:
        bandwidth = torch.tensor(1.0, device=x1.device)

    bandwidth /= (kernel_mul ** (kernel_num // 2))
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

    kernel_val = sum(torch.exp(-dist_sq / (b + 1e-8)) for b in bandwidth_list)
    return kernel_val


def compute_kmmd_distance(x1: torch.Tensor, x2: torch.Tensor) -> float:
    """Computes Kernel Maximum Mean Discrepancy (KMMD) between two sample groups."""
    if x1.shape[0] == 0 or x2.shape[0] == 0:
        return 0.0

    n = x1.shape[0]
    m = x2.shape[0]

    x_total = torch.cat([x1, x2], dim=0)
    k_matrix = compute_gaussian_kernel(x_total, x_total, kernel_mul=2.0, kernel_num=3)

    x1x1 = k_matrix[:n, :n]
    x2x2 = k_matrix[n:, n:]
    x1x2 = k_matrix[:n, n:]

    diff = torch.mean(x1x1) + torch.mean(x2x2) - 2.0 * torch.mean(x1x2)
    # Scaling factor from paper: (m * n) / (m + n) * diff
    scaling = (m * n) / (m + n)
    kmmd = scaling * torch.clamp(diff, min=0.0)
    return float(kmmd.item())


class GramFeatureProfile:
    """
    Learns class-conditional representation statistics from clean samples
    using higher-order Gram matrices.
    """

    def __init__(self, orders: Tuple[int, ...] = (1, 2, 3), k: float = 10.0, max_channels: int = 512):
        self.orders = list(orders)
        self.k = k
        self.max_channels = max_channels

        self.medians: Dict[int, torch.Tensor] = {}
        self.mads: Dict[int, torch.Tensor] = {}
        self.lower_bounds: Dict[int, torch.Tensor] = {}
        self.upper_bounds: Dict[int, torch.Tensor] = {}
        self.threshold: Optional[float] = None

    def _prepare_features(self, features: torch.Tensor) -> torch.Tensor:
        """Ensures features are 4D (N, C, H, W) and bounds channel dimension."""
        if not torch.is_tensor(features):
            features = torch.tensor(features, dtype=torch.float32)
        features = features.float()

        if features.dim() == 2:
            # (N, C) -> (N, C, 1, 1)
            features = features.unsqueeze(-1).unsqueeze(-1)
        elif features.dim() == 3:
            # (N, C, L) -> (N, C, L, 1)
            features = features.unsqueeze(-1)

        c = features.shape[1]
        # For wide representations (e.g. ResNet50 with 2048 channels), pool channels to max_channels
        if c > self.max_channels:
            # Adaptive average pool across channels to avoid memory explosion
            features = F.adaptive_avg_pool3d(
                features.unsqueeze(1),
                (self.max_channels, features.shape[2], features.shape[3])
            ).squeeze(1)

        return features

    def gram_features(self, features: torch.Tensor, order: int) -> torch.Tensor:
        """
        Computes stabilized higher-order Gram matrix:
            G_p = ( (v^p (v^p)^T) / S )^(1/p)
        and extracts the upper-triangular elements.
        """
        features = self._prepare_features(features)
        n, c, h, w = features.shape
        s = max(h * w, 1)

        # Flatten spatial dimensions: (N, C, S)
        x = features.reshape(n, c, s)

        # Spatial normalization to prevent power overflow
        norm = torch.norm(x, p=2, dim=2, keepdim=True).clamp(min=1e-6)
        x_norm = x / norm

        # Compute v^p
        powered = torch.sign(x_norm) * torch.abs(x_norm).pow(order)

        # Compute Gram matrix: (N, C, C)
        gram = torch.bmm(powered, powered.transpose(1, 2)) / s
        gram = torch.nan_to_num(gram, nan=0.0, posinf=1.0, neginf=-1.0)

        # Compute ( ... )^(1/p) preserving sign
        gram = torch.sign(gram) * torch.abs(gram).clamp(min=1e-12).pow(1.0 / order)

        # Extract upper triangular indices
        idx = torch.triu_indices(c, c, device=gram.device)
        gram_vec = gram[:, idx[0], idx[1]]

        return gram_vec

    def calibrate(self, clean_features: torch.Tensor):
        """Builds clean reference profile (median and MAD) for each order."""
        clean_features = self._prepare_features(clean_features)
        if clean_features.shape[0] < 2:
            raise ValueError("At least 2 clean samples are required for calibration.")

        for order in self.orders:
            gram = self.gram_features(clean_features, order)
            median = torch.median(gram, dim=0).values
            mad = torch.median(torch.abs(gram - median), dim=0).values
            mad = torch.clamp(mad, min=1e-6)

            lower = median - self.k * mad
            upper = median + self.k * mad

            self.medians[order] = median
            self.mads[order] = mad
            self.lower_bounds[order] = lower
            self.upper_bounds[order] = upper

    def deviation(self, features: torch.Tensor) -> np.ndarray:
        """
        Computes two-sided deviation from the calibrated clean profile.
        Score is 0 if feature is inside [lower, upper], otherwise distance
        relative to the boundary.
        """
        if not self.lower_bounds:
            raise RuntimeError("Profile must be calibrated before deviation().")

        features = self._prepare_features(features)
        all_order_scores = []

        for order in self.orders:
            gram = self.gram_features(features, order)
            lower = self.lower_bounds[order].to(gram.device)
            upper = self.upper_bounds[order].to(gram.device)

            lower_denom = torch.clamp(torch.abs(lower), min=1e-6)
            upper_denom = torch.clamp(torch.abs(upper), min=1e-6)

            below = torch.where(gram < lower, (lower - gram) / lower_denom, torch.zeros_like(gram))
            above = torch.where(gram > upper, (gram - upper) / upper_denom, torch.zeros_like(gram))

            deviation = below + above
            order_score = deviation.mean(dim=1)
            all_order_scores.append(order_score)

        combined = torch.stack(all_order_scores, dim=1)
        scores = combined.mean(dim=1)
        return scores.detach().cpu().numpy()


class BeatrixDetector(BaseDetector):
    """
    Representation-level Backdoor Detector using Higher-Order Gram Matrices.
    Satisfies BaseDetector interface for ensemble compatibility.
    """

    name = "beatrix"

    def __init__(
        self,
        num_classes: int,
        orders: Tuple[int, ...] = (1, 2, 3),
        threshold_percentile: float = 95.0,
        k: float = 10.0,
        anomaly_threshold: float = 2.0,
        clean_data_perclass: int = 30
    ):
        self.num_classes = num_classes
        self.orders = list(orders)
        self.threshold_percentile = threshold_percentile
        self.k = k
        self.anomaly_threshold = anomaly_threshold
        self.clean_data_perclass = clean_data_perclass

        self.profiles: Dict[int, GramFeatureProfile] = {}
        self.thresholds: Dict[int, float] = {}
        self.global_threshold: Optional[float] = None
        self.clean_features_per_class: Dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # BaseDetector Interface: fit(), score(), flagged()
    # ------------------------------------------------------------------

    def fit(self, clean_features: Union[torch.Tensor, np.ndarray], clean_labels: Union[torch.Tensor, np.ndarray]):
        """
        BaseDetector interface implementation.
        Calibrates class-conditional profiles from clean data.
        """
        if not torch.is_tensor(clean_features):
            clean_features = torch.tensor(clean_features, dtype=torch.float32)
        if torch.is_tensor(clean_labels):
            clean_labels = clean_labels.detach().cpu().numpy()
        else:
            clean_labels = np.asarray(clean_labels)

        all_validation_scores = []

        for class_id in range(self.num_classes):
            indices = np.where(clean_labels == class_id)[0]
            if len(indices) < 2:
                continue

            class_feats = clean_features[indices]
            # Cap clean data per class if specified
            if self.clean_data_perclass and len(indices) > self.clean_data_perclass:
                class_feats = class_feats[:self.clean_data_perclass]

            self.clean_features_per_class[class_id] = class_feats

            # 80% reference, 20% validation for clean threshold
            n = len(class_feats)
            split = max(int(n * 0.8), 1)
            ref_feats = class_feats[:split]
            val_feats = class_feats[split:]
            if len(val_feats) == 0:
                val_feats = ref_feats

            profile = GramFeatureProfile(orders=self.orders, k=self.k)
            profile.calibrate(ref_feats)

            val_scores = profile.deviation(val_feats)
            thresh = float(np.percentile(val_scores, self.threshold_percentile))
            # Recalibrate on all available clean data
            profile.calibrate(class_feats)

            self.profiles[class_id] = profile
            self.thresholds[class_id] = thresh
            all_validation_scores.extend(val_scores.tolist())

        if not self.profiles:
            raise RuntimeError("No class profiles could be calibrated. Insufficient clean data.")

        self.global_threshold = float(
            np.percentile(np.asarray(all_validation_scores), self.threshold_percentile)
        )

    def calibrate(self, clean_features, clean_labels):
        """Extended alias for fit()."""
        return self.fit(clean_features, clean_labels)

    def score(self, features: Union[torch.Tensor, np.ndarray], labels: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        BaseDetector interface implementation.
        Returns per-sample anomaly scores (higher = more suspicious).
        """
        if not self.profiles:
            raise RuntimeError("Call fit() before score().")

        if not torch.is_tensor(features):
            features = torch.tensor(features, dtype=torch.float32)
        if torch.is_tensor(labels):
            labels = labels.detach().cpu().numpy()
        else:
            labels = np.asarray(labels)

        scores = np.zeros(len(labels), dtype=float)

        for class_id, profile in self.profiles.items():
            mask = np.where(labels == class_id)[0]
            if len(mask) == 0:
                continue
            class_scores = profile.deviation(features[mask])
            scores[mask] = class_scores

        return scores

    def flagged(self, features: Union[torch.Tensor, np.ndarray], labels: Union[torch.Tensor, np.ndarray], percentile: str = "p95") -> np.ndarray:
        """
        BaseDetector helper: returns boolean array of flagged samples.
        """
        sample_scores = self.score(features, labels)
        if torch.is_tensor(labels):
            labels = labels.detach().cpu().numpy()
        else:
            labels = np.asarray(labels)

        flags = np.zeros(len(labels), dtype=bool)
        for class_id in np.unique(labels):
            mask = np.where(labels == class_id)[0]
            if len(mask) == 0:
                continue
            thresh = self.thresholds.get(class_id, self.global_threshold)
            flags[mask] = sample_scores[mask] > thresh

        return flags

    # ------------------------------------------------------------------
    # High-level Detection with Model-level Anomaly Index (J*)
    # ------------------------------------------------------------------

    def detect(
        self,
        features: Union[torch.Tensor, np.ndarray],
        labels: Optional[Union[torch.Tensor, np.ndarray]] = None,
        predictions: Optional[Union[torch.Tensor, np.ndarray]] = None
    ) -> BeatrixResult:
        """
        Executes complete Beatrix detection:
            1. Per-sample Gramian deviation scoring and thresholding.
            2. Model-level Anomaly Index (J*) calculation using Median Absolute Deviation.
            3. Target class identification.
            4. Standardized confidence score mapping in [0, 1].
        """
        if not self.profiles:
            raise RuntimeError("Call calibrate() or fit() before detect().")

        if not torch.is_tensor(features):
            features = torch.tensor(features, dtype=torch.float32)

        # Decide which class assignments to evaluate against (predictions preferred)
        if predictions is not None:
            class_ids = predictions.detach().cpu().numpy() if torch.is_tensor(predictions) else np.asarray(predictions)
        elif labels is not None:
            class_ids = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)
        else:
            raise ValueError("Either predictions or labels must be provided.")

        n = len(class_ids)
        scores = np.zeros(n, dtype=np.float64)
        flags = np.zeros(n, dtype=bool)
        used_thresholds = np.zeros(n, dtype=np.float64)

        # 1. Per-sample scoring
        for class_id in np.unique(class_ids):
            indices = np.where(class_ids == class_id)[0]
            if class_id not in self.profiles:
                continue

            profile = self.profiles[class_id]
            class_scores = profile.deviation(features[indices])
            thresh = self.thresholds.get(class_id, self.global_threshold)

            scores[indices] = class_scores
            used_thresholds[indices] = thresh
            flags[indices] = (class_scores > thresh)

        # 2. Model-Level Anomaly Index (J*) across classes
        # For each class, compute discrepancy metric J_c (KMMD or top deviation)
        j_scores = {}
        for class_id in range(self.num_classes):
            indices = np.where(class_ids == class_id)[0]
            if len(indices) == 0:
                j_scores[class_id] = 0.0
                continue

            class_flags = flags[indices]
            flagged_indices = indices[class_flags]
            normal_indices = indices[~class_flags]

            # If there are flagged samples, compute KMMD between normal and flagged
            if len(flagged_indices) > 0 and len(normal_indices) > 0:
                norm_f = features[normal_indices].mean(dim=(2, 3)) if features.dim() == 4 else features[normal_indices]
                flag_f = features[flagged_indices].mean(dim=(2, 3)) if features.dim() == 4 else features[flagged_indices]
                kmmd = compute_kmmd_distance(norm_f, flag_f)
                j_scores[class_id] = kmmd
            elif len(flagged_indices) > 0 and class_id in self.clean_features_per_class:
                # Compare against stored clean reference
                clean_ref = self.clean_features_per_class[class_id]
                clean_flat = clean_ref.mean(dim=(2, 3)) if clean_ref.dim() == 4 else clean_ref
                flag_f = features[flagged_indices].mean(dim=(2, 3)) if features.dim() == 4 else features[flagged_indices]
                kmmd = compute_kmmd_distance(clean_flat, flag_f)
                j_scores[class_id] = kmmd
            else:
                # Discrepancy metric based on deviation magnitude
                class_devs = scores[indices]
                top_k = max(int(len(class_devs) * 0.1), 1)
                j_scores[class_id] = float(np.mean(np.sort(class_devs)[-top_k:]))

        # MAD analysis of J_c across classes (from NDSS 2023 Beatrix paper)
        j_values = np.array([j_scores[c] for c in range(self.num_classes)], dtype=float)
        j_median = float(np.median(j_values))
        j_mad = float(np.median(np.abs(j_values - j_median)))

        # J_star: anomaly index
        j_star = np.abs(j_values - j_median) / (1.4826 * (j_mad + 1e-6))
        class_anomaly_indices = {c: float(j_star[c]) for c in range(self.num_classes)}

        max_anomaly_index = float(np.max(j_star))
        suspected_target = int(np.argmax(j_star))

        # Decision rule: If max J* >= anomaly_threshold (default 2.0), model is backdoored
        is_backdoored = bool(max_anomaly_index >= self.anomaly_threshold)
        decision = "BACKDOOR" if is_backdoored else "CLEAN"

        # Map Anomaly Index to standardized confidence score in [0, 1] for Module 5
        # Sigmoid centered at anomaly_threshold (2.0)
        norm_score = float(1.0 / (1.0 + np.exp(-1.5 * (max_anomaly_index - self.anomaly_threshold))))

        backdoor_pct = float(flags.mean() * 100.0) if n else 0.0
        avg_thresh = float(used_thresholds.mean()) if n else 0.0

        return BeatrixResult(
            is_backdoored=is_backdoored,
            decision=decision,
            score=norm_score,
            suspected_target_class=suspected_target if is_backdoored else None,
            anomaly_index=max_anomaly_index,
            class_anomaly_indices=class_anomaly_indices,
            scores=scores,
            flags=flags,
            backdoor_percentage=backdoor_pct,
            threshold=avg_thresh,
            predictions=class_ids,
            true_labels=labels.detach().cpu().numpy() if torch.is_tensor(labels) else (np.asarray(labels) if labels is not None else np.full(n, -1)),
            metrics={}
        )

    # ------------------------------------------------------------------
    # Performance Evaluation Utility
    # ------------------------------------------------------------------

    @staticmethod
    def evaluate(flags: Union[np.ndarray, list], poison_labels: Union[np.ndarray, list]) -> dict:
        """Evaluates detection accuracy, precision, recall, and F1 score."""
        flags = np.asarray(flags, dtype=bool)
        poison_labels = np.asarray(poison_labels, dtype=bool)

        if len(flags) != len(poison_labels):
            raise ValueError("flags and poison_labels must have equal length.")

        tp = int(np.sum(flags & poison_labels))
        tn = int(np.sum((~flags) & (~poison_labels)))
        fp = int(np.sum(flags & (~poison_labels)))
        fn = int(np.sum((~flags) & poison_labels))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(flags) if len(flags) > 0 else 0.0

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        return {
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(accuracy),
            "FPR": float(fpr),
            "FNR": float(fnr)
        }