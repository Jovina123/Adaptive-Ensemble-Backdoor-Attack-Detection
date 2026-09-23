"""
evidence_extraction.py

Module 1 - Evidence Extraction

Extracts intermediate representations from a trained model.

The detector itself is NOT implemented here. This module only:
    1. Runs probe images through the model.
    2. Captures an intermediate feature map.
    3. Stores the model prediction.
    4. Stores the dataset label.
    5. Stores the features needed by representation-level detectors.

The Beatrix detector consumes the resulting Evidence object.
"""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Evidence:
    """
    Evidence extracted from a model.

    features:
        Intermediate representation.
        Shape:
            (N, C, H, W)
        for CNN image models.

    labels:
        Dataset labels supplied by ImageFolder.

    predictions:
        Model predictions.

    probabilities:
        Softmax confidence of the predicted class.
    """

    features: torch.Tensor
    labels: np.ndarray
    predictions: np.ndarray
    probabilities: np.ndarray


class EvidenceExtractor:

    def __init__(self, model, hook_layer_name, device):

        self.model = model.to(device).eval()
        self.device = device

        self._captured = None

        self._hook_handle = self._attach_hook(
            hook_layer_name
        )

    def _attach_hook(self, hook_layer_name):

        named_modules = dict(
            self.model.named_modules()
        )

        if hook_layer_name not in named_modules:

            available = [
                name
                for name, _ in self.model.named_modules()
            ]

            raise ValueError(
                f"\nLayer '{hook_layer_name}' was not found.\n\n"
                f"Available layers:\n"
                + "\n".join(available)
            )

        layer = named_modules[hook_layer_name]

        def hook_fn(module, module_input, module_output):

            if isinstance(module_output, tuple):
                module_output = module_output[0]

            self._captured = (
                module_output.detach()
            )

        return layer.register_forward_hook(
            hook_fn
        )

    @torch.no_grad()
    def extract(self, probe_loader):

        all_features = []
        all_labels = []
        all_predictions = []
        all_probabilities = []

        for images, labels in probe_loader:

            images = images.to(self.device)

            self._captured = None

            outputs = self.model(images)

            if self._captured is None:

                raise RuntimeError(
                    "The forward hook did not capture "
                    "any feature representation."
                )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            predictions = probabilities.argmax(
                dim=1
            )

            confidence = probabilities.max(
                dim=1
            ).values

            all_features.append(
                self._captured.cpu()
            )

            all_labels.append(
                labels.cpu().numpy()
            )

            all_predictions.append(
                predictions.cpu().numpy()
            )

            all_probabilities.append(
                confidence.cpu().numpy()
            )

        if not all_features:

            raise RuntimeError(
                "No probe images were found."
            )

        return Evidence(

            features=torch.cat(
                all_features,
                dim=0
            ),

            labels=np.concatenate(
                all_labels
            ),

            predictions=np.concatenate(
                all_predictions
            ),

            probabilities=np.concatenate(
                all_probabilities
            )
        )

    def remove_hook(self):

        if self._hook_handle is not None:

            self._hook_handle.remove()

            self._hook_handle = None