"""
Project-level adapter for the FLARE backdoor detection method.

The underlying FLARE implementation is provided by BackdoorBox.
This module gives our project a clean interface for calling FLARE.
"""


class FLAREDetector:
    """Adapter around the BackdoorBox FLARE implementation."""

    def __init__(self, model, xi=0.02, seed=666, deterministic=False):
        """
        Initialize the FLARE detector.

        Parameters
        ----------
        model:
            The suspicious/backdoored image-classification model.

        xi : float
            FLARE clustering stability parameter.

        seed : int
            Random seed used by FLARE.

        deterministic : bool
            Whether deterministic execution is requested.
        """
        try:
            from core import FLARE
        except ImportError as exc:
            raise ImportError(
                "BackdoorBox could not be imported. "
                "Make sure BackdoorBox is available in the Python environment."
            ) from exc

        self.detector = FLARE(
            model=model,
            xi=xi,
            seed=seed,
            deterministic=deterministic,
        )

    def detect(self, poisoned_trainset, y_true):
        """
        Run FLARE on a training dataset.

        Parameters
        ----------
        poisoned_trainset:
            Dataset containing the samples to analyze.

        y_true:
            Ground-truth poison labels used for experimental evaluation.

        Returns
        -------
        The result returned by the BackdoorBox FLARE implementation.
        """
        return self.detector.detect(
            poisoned_trainset=poisoned_trainset,
            y_true=y_true,
        )