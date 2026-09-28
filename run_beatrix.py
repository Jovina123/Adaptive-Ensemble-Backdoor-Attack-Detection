"""
run_beatrix.py

Generic Beatrix Backdoor Detection Runner.

Supports two modes:

1. NORMAL MODE
   Loads the model and extracts intermediate representations.

2. CACHED FEATURE MODE
   Loads previously extracted Beatrix features from a .pt file.
   This avoids running the model and feature extraction again.

Example cached-feature usage:

python3 run_beatrix.py \
    --cached_features evidence/cifar10/all2all/target_0/feature_evidence.pt \
    --output results_beatrix_cached.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.universal_image_dataset import UniversalImageDataset
from data.preprocessing import get_eval_transform
from evidence_extraction import EvidenceExtractor
from detectors.beatrix_detector import BeatrixDetector
from utils import load_model_from_checkpoint


# ================================================================
# ARGUMENTS
# ================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Run Beatrix Backdoor Detection."
    )

    parser.add_argument(
        "--data_dir",
        required=False,
        default=None,
        help="Path to image dataset directory."
    )

    parser.add_argument(
        "--checkpoint",
        required=False,
        default=None,
        help="Path to trained model checkpoint."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for feature extraction."
    )

    parser.add_argument(
        "--hook_layer",
        type=str,
        default="layer4",
        help="Intermediate layer used for feature extraction."
    )

    parser.add_argument(
        "--clean_ratio",
        type=float,
        default=0.30,
        help="Percentage of each class used for clean calibration."
    )

    parser.add_argument(
        "--anomaly_threshold",
        type=float,
        default=2.0,
        help="Threshold used for Beatrix anomaly detection."
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results_beatrix.json",
        help="Output JSON report."
    )

    parser.add_argument(
        "--cached_features",
        type=str,
        default=None,
        help="Previously extracted Beatrix feature evidence (.pt)."
    )
    parser.add_argument(
    "--max_samples",
    type=int,
    default=None,
    help="Maximum number of cached samples to evaluate."
    )
    return parser.parse_args()


# ================================================================
# MAIN
# ================================================================

def main():

    args = parse_args()

    start_time = time.time()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print()
    print("=" * 70)
    print("BEATRIX BACKDOOR DETECTION")
    print("=" * 70)

    print("Device            :", device)
    print("Hook layer        :", args.hook_layer)
    print("Anomaly threshold :", args.anomaly_threshold)

    # ============================================================
    # VARIABLES
    # ============================================================

    model = None
    dataset = None
    arch_name = None
    input_size = None

    # ============================================================
    # MODE 1:
    # LOAD CACHED FEATURES
    # ============================================================

    if args.cached_features:

        print()
        print("[1] Loading Cached Beatrix Features...")

        cache_path = Path(args.cached_features)

        if not cache_path.exists():

            raise FileNotFoundError(
                f"\nCached feature file not found:\n{cache_path}"
            )

        evidence = torch.load(
            cache_path,
            map_location="cpu",
            weights_only=False
        )

        print("  Cache loaded successfully.")

        # --------------------------------------------------------
        # CHECK REQUIRED KEYS
        # --------------------------------------------------------

        required_keys = [
            "clean_features",
            "backdoor_features",
            "clean_labels",
            "backdoor_labels",
            "clean_predictions",
            "backdoor_predictions"
        ]

        for key in required_keys:

            if key not in evidence:

                raise KeyError(
                    f"Missing key in cached evidence: {key}"
                )

        # --------------------------------------------------------
        # DISPLAY FEATURE SHAPES
        # --------------------------------------------------------

        print(
            "  Clean features    :",
            tuple(evidence["clean_features"].shape)
        )

        print(
            "  Backdoor features :",
            tuple(evidence["backdoor_features"].shape)
        )

        print(
            "  Clean labels      :",
            tuple(evidence["clean_labels"].shape)
        )

        print(
            "  Backdoor labels   :",
            tuple(evidence["backdoor_labels"].shape)
        )

        # --------------------------------------------------------
        # COMBINE CLEAN + BACKDOOR
        # --------------------------------------------------------

        features = torch.cat(
            [
                evidence["clean_features"],
                evidence["backdoor_features"]
            ],
            dim=0
        )

        labels = torch.cat(
            [
                evidence["clean_labels"],
                evidence["backdoor_labels"]
            ],
            dim=0
        )

        predictions = torch.cat(
            [
                evidence["clean_predictions"],
                evidence["backdoor_predictions"]
            ],
            dim=0
        )

        # --------------------------------------------------------
        # NUMBER OF CLASSES
        # --------------------------------------------------------

        num_classes = int(
            max(
                labels.max().item(),
                predictions.max().item()
            ) + 1
        )

        arch_name = "Cached Model Features"

        input_size = features.shape[-1]

        print()
        print("  Total features    :", len(features))
        print("  Number of classes :", num_classes)
        print("  Feature shape     :", tuple(features.shape))

        # --------------------------------------------------------
        # CLASS NAMES
        # --------------------------------------------------------

        cifar10_names = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck"
        ]

        if num_classes == 10:

            class_names = cifar10_names

        else:

            class_names = [
                f"Class_{i}"
                for i in range(num_classes)
            ]

    # ============================================================
    # MODE 2:
    # LOAD MODEL AND EXTRACT FEATURES
    # ============================================================

    else:

        if args.data_dir is None:

            raise ValueError(
                "--data_dir is required when --cached_features is not used."
            )

        if args.checkpoint is None:

            raise ValueError(
                "--checkpoint is required when --cached_features is not used."
            )

        # --------------------------------------------------------
        # LOAD MODEL
        # --------------------------------------------------------

        print()
        print("[1] Loading Model Checkpoint...")

        model, arch_name, num_classes, input_size, ckpt = (
            load_model_from_checkpoint(
                args.checkpoint,
                device=device
            )
        )

        print("  Architecture     :", arch_name)
        print("  Number of classes :", num_classes)
        print(
            "  Input resolution :",
            f"{input_size}x{input_size}"
        )

        # --------------------------------------------------------
        # LOAD DATASET
        # --------------------------------------------------------

        print()
        print("[2] Preparing Dataset...")

        transform = get_eval_transform(
            image_size=input_size
        )

        dataset = UniversalImageDataset(
            root_dir=args.data_dir,
            transform=transform
        )

        print(
            "  Total images       :",
            len(dataset)
        )

        print(
            "  Discovered classes :",
            len(dataset.classes)
        )

        print(
            "  Class labels       :",
            dataset.classes
        )

        class_names = dataset.classes

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0
        )

        # --------------------------------------------------------
        # EXTRACT FEATURES
        # --------------------------------------------------------

        print()
        print(
            "[3] Extracting Intermediate Representations..."
        )

        print(
            "  Layer:",
            args.hook_layer
        )

        extractor = EvidenceExtractor(
            model=model,
            hook_layer_name=args.hook_layer,
            device=device
        )

        evidence = extractor.extract(
            loader
        )

        extractor.remove_hook()

        features = evidence.features
        labels = evidence.labels
        predictions = evidence.predictions

        print(
            "  Feature shape :",
            tuple(features.shape)
        )

        accuracy = np.mean(
            predictions.numpy() == labels.numpy()
        ) * 100

        print(
            f"  Model accuracy: {accuracy:.2f}%"
        )

    # ============================================================
    # BEATRIX DETECTOR
    # ============================================================

    print()
    print("[4] Initializing Beatrix Detector...")

    detector = BeatrixDetector(
        num_classes=num_classes,
        orders=(1, 2, 3),
        anomaly_threshold=args.anomaly_threshold
    )

    # ============================================================
    # SPLIT CALIBRATION / TEST DATA
    # ============================================================

    print()
    print("[5] Preparing Calibration Data...")

    clean_indices = []
    test_indices = []

    for c in range(num_classes):

        class_indices = np.where(
            labels.numpy() == c
        )[0]

        if len(class_indices) == 0:
            continue

        n_calib = max(
            int(len(class_indices) * args.clean_ratio),
            2
        )

        clean_indices.extend(
            class_indices[:n_calib]
        )

        test_indices.extend(
            class_indices
        )

    clean_indices = np.array(
        clean_indices,
        dtype=int
    )

    test_indices = np.array(
        test_indices,
        dtype=int
    )

    print(
        "  Calibration samples :",
        len(clean_indices)
    )

    print(
        "  Evaluation samples  :",
        len(test_indices)
    )

    # ============================================================
    # FIT BEATRIX
    # ============================================================

    print()
    print("[6] Fitting Beatrix Gram Matrix Profiles...")

    detector.fit(
        clean_features=features[clean_indices],
        clean_labels=labels[clean_indices]
    )

    print("  Calibration complete.")

    # ============================================================
    # DETECTION
    # ============================================================

    print()
    print("[7] Running Beatrix Detection...")

    result = detector.detect(
        features=features[test_indices],
        labels=labels[test_indices],
        predictions=predictions[test_indices]
    )

    elapsed_time = time.time() - start_time

    # ============================================================
    # RESULTS
    # ============================================================

    print()
    print("=" * 70)
    print("DETECTION RESULT SUMMARY")
    print("=" * 70)

    print(
        "Final Decision       :",
        result.decision
    )

    print(
        f"Standardized Score   : {result.score:.4f}"
    )

    print(
        f"Max Anomaly Index J* : "
        f"{result.anomaly_index:.4f}"
    )

    print(
        f"Threshold            : "
        f"{args.anomaly_threshold:.4f}"
    )

    if result.is_backdoored:

        target_class = result.suspected_target_class

        if (
            target_class is not None
            and target_class < len(class_names)
        ):

            target_name = class_names[
                target_class
            ]

        else:

            target_name = "Unknown"

        print(
            f"Suspected Target     : "
            f"Class {target_class} "
            f"('{target_name}')"
        )

    else:

        print(
            "Suspected Target     : None"
        )

    print(
        f"Flagged Samples      : "
        f"{result.flags.sum()} / "
        f"{len(result.flags)} "
        f"({result.backdoor_percentage:.2f}%)"
    )

    print(
        f"Processing Time      : "
        f"{elapsed_time:.2f} seconds"
    )

    # ============================================================
    # CLASS-WISE RESULTS
    # ============================================================

    print()
    print("Class-by-Class Anomaly Indices")
    print("-" * 55)

    for c in range(num_classes):

        if c < len(class_names):

            class_name = class_names[c]

        else:

            class_name = f"Class_{c}"

        j_value = result.class_anomaly_indices.get(
            c,
            0.0
        )

        if (
            result.is_backdoored
            and c == result.suspected_target_class
        ):

            marker = " <-- SUSPECTED TARGET"

        else:

            marker = ""

        print(
            f"  Class {c:2d} "
            f"({class_name:12s}) : "
            f"J* = {j_value:.4f}"
            f"{marker}"
        )

    # ============================================================
    # CREATE JSON REPORT
    # ============================================================

    report = {

        "detector": "Beatrix",

        "level": "Representation-Level",

        "mode": (
            "cached_features"
            if args.cached_features
            else "fresh_extraction"
        ),

        "model_path": args.checkpoint,

        "dataset_path": args.data_dir,

        "cached_feature_path": args.cached_features,

        "architecture": arch_name,

        "num_classes": num_classes,

        "classes": class_names,

        "decision": result.decision,

        "is_backdoored": bool(
            result.is_backdoored
        ),

        "score": float(
            result.score
        ),

        "anomaly_index": float(
            result.anomaly_index
        ),

        "anomaly_threshold": float(
            args.anomaly_threshold
        ),

        "suspected_target_class": (
            int(result.suspected_target_class)
            if result.suspected_target_class is not None
            else None
        ),

        "suspected_target_class_name": (
            class_names[
                result.suspected_target_class
            ]
            if (
                result.is_backdoored
                and result.suspected_target_class is not None
                and result.suspected_target_class < len(class_names)
            )
            else None
        ),

        "class_anomaly_indices": {
            str(k): float(v)
            for k, v in result.class_anomaly_indices.items()
        },

        "flagged_samples": int(
            result.flags.sum()
        ),

        "total_samples": int(
            len(result.flags)
        ),

        "backdoor_percentage": float(
            result.backdoor_percentage
        ),

        "average_threshold": float(
            result.threshold
        ),

        "elapsed_seconds": round(
            elapsed_time,
            2
        )
    }

    # ============================================================
    # SAVE REPORT
    # ============================================================

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w"
    ) as f:

        json.dump(
            report,
            f,
            indent=4
        )

    print()
    print("=" * 70)

    print(
        "Report saved to:",
        output_path.resolve()
    )

    print("=" * 70)
    print()

    return report


# ================================================================
# PROGRAM ENTRY POINT
# ================================================================

if __name__ == "__main__":

    main()