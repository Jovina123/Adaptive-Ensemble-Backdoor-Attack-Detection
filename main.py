"""
main.py

Adaptive Multi-Detector Backdoor Detection Framework.

Current implementation:
    - Universal image dataset loading
    - Generic model checkpoint loading
    - torchvision ResNet-50 support
    - Beatrix intermediate feature extraction
    - Beatrix Gram-matrix detection
    - Other detectors reserved for other team members

The framework accepts arbitrary ImageFolder-style datasets.

Expected dataset structure:

dataset/
    class_1/
        image1.jpg
        image2.jpg

    class_2/
        image1.jpg
        image2.jpg

    class_3/
        image1.jpg
        image2.jpg

No dataset-specific class names are hard-coded.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from data.universal_image_dataset import UniversalImageDataset
from data.preprocessing import get_eval_transform
from evidence_extraction import EvidenceExtractor
from detectors.beatrix_detector import BeatrixDetector
from utils import load_model_from_checkpoint


# ============================================================
# DATASET
# ============================================================

def prepare_dataset(dataset_path, image_size=224):
    """
    Load an arbitrary ImageFolder-style image dataset.

    No class names or number of classes are hard-coded.
    """

    dataset = UniversalImageDataset(
        root_dir=dataset_path,
        transform=get_eval_transform(image_size=image_size)
    )

    return dataset


# ============================================================
# DETECTOR RESULT FORMAT
# ============================================================

def standardize_result(
    detector_name,
    score,
    decision,
    confidence=None
):
    """
    Convert detector output into a common format.
    """

    result = {
        "detector": detector_name,
        "score": float(score),
        "decision": decision
    }

    if confidence is not None:
        result["confidence"] = float(
            confidence
        )

    return result


# ============================================================
# ADAPTIVE ENSEMBLE
# ============================================================

def adaptive_fusion(results):
    """
    Temporary ensemble implementation.

    Only detectors that actually return a score
    are included.

    At the current project stage, only Beatrix
    is connected.
    """

    if not results:
        return {
            "final_score": 0.0,
            "final_decision": "INSUFFICIENT_EVIDENCE"
        }

    scores = [
        result["score"]
        for result in results
        if result.get("score") is not None
    ]

    if not scores:
        return {
            "final_score": 0.0,
            "final_decision": "INSUFFICIENT_EVIDENCE"
        }

    final_score = sum(scores) / len(scores)

    if final_score >= 0.5:
        decision = "BACKDOOR"
    else:
        decision = "CLEAN"

    return {
        "final_score": float(final_score),
        "final_decision": decision
    }


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # ARGUMENTS
    # --------------------------------------------------------

    parser = argparse.ArgumentParser(
        description=
        "Adaptive Multi-Detector Backdoor Detection"
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to arbitrary image dataset"
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained model checkpoint"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32
    )

    parser.add_argument(
        "--output",
        default="detection_result.json"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("=" * 70)
    print("ADAPTIVE BACKDOOR DETECTION FRAMEWORK")
    print("=" * 70)

    print(
        f"Device: {device}"
    )

    # ========================================================
    # MODEL
    # ========================================================

    print("\nLoading model...")

    model, arch_name, checkpoint_num_classes, input_size, checkpoint = load_model_from_checkpoint(
        args.model,
        device=device
    )

    print("Model loaded successfully.")
    print(f"Architecture: {arch_name}")
    print(f"Input resolution: {input_size}x{input_size}")
    print(f"Checkpoint number of classes: {checkpoint_num_classes}")

    # ========================================================
    # DATASET
    # ========================================================

    print("\nLoading dataset...")

    dataset = prepare_dataset(
        args.dataset,
        image_size=input_size
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    print(
        f"\nDataset: {args.dataset}"
    )

    print(
        f"Number of images: {len(dataset)}"
    )

    print(
        f"Number of classes: "
        f"{len(dataset.classes)}"
    )

    print(
        f"Classes: {dataset.classes}"
    )

    # --------------------------------------------------------
    # Verify dataset/model class count
    # --------------------------------------------------------

    if len(dataset.classes) != int(
        checkpoint_num_classes
    ):

        raise ValueError(
            "\nDataset/model class mismatch.\n"
            f"Dataset classes: {len(dataset.classes)}\n"
            f"Model classes: {checkpoint_num_classes}\n\n"
            "The input dataset must contain the same "
            "number of classes as the model was trained on."
        )

    # ========================================================
    # DETECTOR CONTEXT
    # ========================================================

    context = {

        "dataset": dataset,

        "dataloader": loader,

        "checkpoint": checkpoint,

        "model": model,

        "device": device,

        "dataset_path":
            str(
                Path(
                    args.dataset
                ).resolve()
            ),

        "model_path":
            str(
                Path(
                    args.model
                ).resolve()
            )
    }

    # ========================================================
    # DETECTORS
    # ========================================================

    detector_results = []

    print()
    print("=" * 70)
    print("RUNNING DETECTORS")
    print("=" * 70)

    # ========================================================
    # FLARE
    # ========================================================

    print("\n[1] FLARE")

    print(
        "FLARE implementation not connected yet."
    )

    # ========================================================
    # BEATRIX
    # ========================================================

    print("\n[2] BEATRIX")

    try:

        # ----------------------------------------------------
        # FEATURE EXTRACTION
        # ----------------------------------------------------

        print(
            "Extracting intermediate representations..."
        )

        # For ResNet-50, layer4 is the final convolutional
        # representation before global average pooling
        # and the classifier.
        #
        # Output:
        #     (N, 2048, 7, 7)
        #
        # This is the representation used by Beatrix.

        extractor = EvidenceExtractor(
            model=model,
            hook_layer_name="layer4",
            device=device
        )

        evidence = extractor.extract(
            loader
        )

        extractor.remove_hook()

        print(
            "Beatrix evidence extracted successfully."
        )

        print(
            f"Feature shape: "
            f"{tuple(evidence.features.shape)}"
        )

        print(
            f"Labels shape: "
            f"{evidence.labels.shape}"
        )

        print(
            f"Predictions shape: "
            f"{evidence.predictions.shape}"
        )

        print(
            f"Probabilities shape: "
            f"{evidence.probabilities.shape}"
        )

        # ----------------------------------------------------
        # BEATRIX DETECTOR
        # ----------------------------------------------------

        beatrix = BeatrixDetector(
            num_classes=len(dataset.classes)
        )

        # ----------------------------------------------------
        # CALIBRATION
        # ----------------------------------------------------

        print()
        print(
            "Calibrating Beatrix..."
        )

        beatrix.calibrate(
            clean_features=evidence.features,
            clean_labels=evidence.labels
        )

        # ----------------------------------------------------
        # DETECTION
        # ----------------------------------------------------

        print()
        print(
            "Running Beatrix detection..."
        )

        beatrix_result = beatrix.detect(
            features=evidence.features,
            labels=evidence.labels,
            predictions=evidence.predictions
        )

        # ----------------------------------------------------
        # RESULTS
        # ----------------------------------------------------

        flagged_count = int(
            beatrix_result.flags.sum()
        )

        total_count = len(
            beatrix_result.flags
        )

        backdoor_percentage = float(
            beatrix_result.backdoor_percentage
        )

        print()
        print(
            "Beatrix detection complete."
        )

        print(
            f"Samples flagged: "
            f"{flagged_count}/{total_count}"
        )

        print(
            f"Backdoor percentage: "
            f"{backdoor_percentage:.2f}%"
        )

        print(
            f"Threshold: "
            f"{beatrix_result.threshold:.6f}"
        )

        # ----------------------------------------------------
        # NORMALIZED SCORE
        # ----------------------------------------------------

        beatrix_score = float(beatrix_result.score)
        beatrix_decision = beatrix_result.decision

        print(
            f"Anomaly Index (J*): {beatrix_result.anomaly_index:.4f}"
        )
        if beatrix_result.is_backdoored:
            target_name = (
                dataset.classes[beatrix_result.suspected_target_class]
                if beatrix_result.suspected_target_class < len(dataset.classes)
                else "Unknown"
            )
            print(
                f"Suspected Target Class: Class {beatrix_result.suspected_target_class} ('{target_name}')"
            )
        print(
            f"Decision: {beatrix_decision} (Score: {beatrix_score:.4f})"
        )

        detector_results.append(
            standardize_result(
                detector_name="Beatrix",
                score=beatrix_score,
                decision=beatrix_decision
            )
        )

        # ----------------------------------------------------
        # ADD DETAILED BEATRIX INFORMATION
        # ----------------------------------------------------

        beatrix_details = {

            "samples": total_count,

            "flagged_samples":
                flagged_count,

            "backdoor_percentage":
                backdoor_percentage,

            "anomaly_index":
                float(
                    beatrix_result.anomaly_index
                ),

            "suspected_target_class":
                beatrix_result.suspected_target_class,

            "decision":
                beatrix_decision,

            "threshold":
                float(
                    beatrix_result.threshold
                ),

            "feature_shape":
                list(
                    evidence.features.shape
                )
        }

    except Exception as e:

        print()
        print(
            "Beatrix failed:"
        )

        print(
            str(e)
        )

        beatrix_details = {
            "error": str(e)
        }

    # ========================================================
    # NEURAL CLEANSE
    # ========================================================

    print("\n[3] NEURAL CLEANSE")

    print(
        "Neural Cleanse implementation not connected yet."
    )

    # ========================================================
    # IBD-PSC
    # ========================================================

    print("\n[4] IBD-PSC")

    print(
        "IBD-PSC implementation not connected yet."
    )

    # ========================================================
    # ADAPTIVE FUSION
    # ========================================================

    print()
    print("=" * 70)
    print("ADAPTIVE FUSION")
    print("=" * 70)

    final_result = adaptive_fusion(
        detector_results
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    output = {

        "dataset": args.dataset,

        "model": args.model,

        "num_images":
            len(dataset),

        "num_classes":
            len(dataset.classes),

        "classes":
            dataset.classes,

        "detectors":
            detector_results,

        "beatrix_details":
            beatrix_details,

        "ensemble":
            final_result
    }

    # --------------------------------------------------------
    # SAVE JSON
    # --------------------------------------------------------

    with open(
        args.output,
        "w"
    ) as f:

        json.dump(
            output,
            f,
            indent=4
        )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print(
        json.dumps(
            final_result,
            indent=4
        )
    )

    print()
    print(
        f"Result saved to: {args.output}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()