import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class IBDPSCDetector:

    def __init__(
        self,
        model,
        n=5,
        xi=0.6,
        threshold=0.9,
        scale=1.5,
        device=None
    ):
        self.model = model
        self.n = n
        self.xi = xi
        self.threshold = threshold
        self.scale = scale

        self.device = device or (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        self.model = self.model.to(self.device)
        self.model.eval()
        self.start_index = None

    def count_bn_layers(self):
        """Count the BatchNorm2d layers in the model."""
        count = 0

        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                count += 1

        return count

    def get_bn_indices(self):
        """Get BN layer indices in reverse order."""
        total_bn = self.count_bn_layers()

        return list(reversed(range(total_bn)))
    def get_bn_indices(self):
        """Get BN layer indices in reverse order."""
        total_bn = self.count_bn_layers()

        return list(reversed(range(total_bn)))
    def scale_model(self, model, bn_indices):
        """Create a copy of the model and scale selected BN parameters."""
        scaled_model = copy.deepcopy(model)

        bn_count = 0

        for module in scaled_model.modules():
            if isinstance(module, nn.BatchNorm2d):
                if bn_count in bn_indices:
                    module.weight.data *= self.scale
                    module.bias.data *= self.scale

                bn_count += 1

        scaled_model.eval()

        return scaled_model
    def calculate_psc(self, inputs):
        """Calculate the Parameter-oriented Scaling Consistency (PSC)."""

        inputs = inputs.to(self.device)

        # Prediction from the original model
        with torch.no_grad():
            original_output = self.model(inputs)
            original_prediction = torch.argmax(
                original_output, dim=1
            )

        confidence_scores = []

        # Get BN layers
        bn_indices = self.get_bn_indices()

        # Use adaptive starting point when available
        if self.start_index is not None:
            bn_indices = bn_indices[:self.start_index + 1]

        # Create and test scaled model copies
        for _ in range(self.n):
            scaled_model = self.scale_model(
                self.model,
                bn_indices
            )

            with torch.no_grad():
                output = scaled_model(inputs)
                probabilities = F.softmax(output, dim=1)

                # Confidence for the original predicted class
                confidence = probabilities[
                    torch.arange(inputs.size(0)),
                    original_prediction
                ]

                confidence_scores.append(confidence)

        # Average confidence across scaled models
        psc_score = torch.stack(
            confidence_scores
        ).mean(dim=0)

        return psc_score
        
    def find_start_index(self, val_loader):
        """
        Find the starting BN layer for parameter scaling
        using the validation set.
        """

        bn_indices = self.get_bn_indices()

        for i in range(len(bn_indices)):
            selected_indices = bn_indices[:i + 1]

            scaled_model = self.scale_model(
                self.model,
                selected_indices
            )

            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    outputs = scaled_model(images)
                    predictions = torch.argmax(outputs, dim=1)

                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)

            if total == 0:
                continue

            error_rate = 1 - (correct / total)

            if error_rate > self.xi:
                return i

        return len(bn_indices) - 1
    def detect(self, inputs):
        """
        Detect whether input samples are likely to be backdoor inputs.

        Returns:
            psc_scores: PSC score for each input
            is_backdoor: True if PSC >= threshold
        """

        psc_scores = self.calculate_psc(inputs)

        is_backdoor = psc_scores >= self.threshold

        return psc_scores, is_backdoor

    if __name__ == "__main__":
        print("IBD-PSC detector module loaded successfully.")