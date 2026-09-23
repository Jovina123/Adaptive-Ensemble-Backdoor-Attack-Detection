"""
run_beatrix.py

Generic Beatrix Backdoor Detection Runner.
Executes the Representation-Level Backdoor Detector based on:
    "The 'Beatrix' Resurrections: Robust Backdoor Detection via Gram Matrices"
    (NDSS 2023).

Works with arbitrary image datasets and supported model architectures
(CIFAR-10 / GTSRB PreActResNet, Torchvision ResNet-50, etc.).
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Beatrix Backdoor Detection on a model checkpoint and dataset."
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Path to image dataset directory (ImageFolder format)."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained model checkpoint file."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for evidence extraction."
    )
    parser.add_argument(
        "--hook_layer",
        type=str,
        default="layer4",
        help="Intermediate layer to hook for representation extraction (default: layer4)."
    )
    parser.add_argument(
        "--clean_ratio",
        type=float,
        default=0.3,
        help="Proportion of samples per class used for calibration reference (default: 0.3)."
    )
    parser.add_argument(
        "--anomaly_threshold",
        type=float,
        default=2.0,
        help="MAD anomaly index threshold for declaring a model backdoored (default: 2.0)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results_beatrix.json",
        help="Path to save output JSON report (default: results_beatrix.json)."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 70)
    print("BEATRIX BACKDOOR DETECTION (REPRESENTATION-LEVEL)")
    print("=" * 70)
    print(f"Device               : {device}")
    print(f"Dataset path         : {args.data_dir}")
    print(f"Checkpoint path      : {args.checkpoint}")
    print(f"Hook layer           : {args.hook_layer}")
    print(f"Anomaly threshold    : {args.anomaly_threshold}")

    start_time = time.time()

    # ------------------------------------------------------------------
    # 1. LOAD MODEL
    # ------------------------------------------------------------------
    print("\n[1] Loading Model Checkpoint...")
    model, arch_name, num_classes, input_size, ckpt = load_model_from_checkpoint(
        args.checkpoint, device=device
    )
    print(f"  Architecture       : {arch_name}")
    print(f"  Expected classes   : {num_classes}")
    print(f"  Input resolution   : {input_size}x{input_size}")

    # ------------------------------------------------------------------
    # 2. LOAD DATASET
    # ------------------------------------------------------------------
    print("\n[2] Preparing Dataset Loader...")
    transform = get_eval_transform(image_size=input_size)
    dataset = UniversalImageDataset(
        root_dir=args.data_dir,
        transform=transform
    )
    print(f"  Total images       : {len(dataset)}")
    print(f"  Discovered classes : {len(dataset.classes)}")
    print(f"  Class labels       : {dataset.classes}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # ------------------------------------------------------------------
    # 3. EVIDENCE EXTRACTION (Module 1)
    # ------------------------------------------------------------------
    print(f"\n[3] Extracting Intermediate Representations from '{args.hook_layer}'...")
    extractor = EvidenceExtractor(
        model=model,
        hook_layer_name=args.hook_layer,
        device=device
    )
    evidence = extractor.extract(loader)
    extractor.remove_hook()

    print(f"  Feature shape      : {tuple(evidence.features.shape)}")
    print(f"  Model Accuracy     : {np.mean(evidence.predictions == evidence.labels) * 100:.2f}%")

    # ------------------------------------------------------------------
    # 4. BEATRIX CALIBRATION & DETECTION (Module 4)
    # ------------------------------------------------------------------
    print("\n[4] Initializing Beatrix Detector...")
    detector = BeatrixDetector(
        num_classes=num_classes,
        orders=(1, 2, 3),
        anomaly_threshold=args.anomaly_threshold
    )

    # Split clean calibration data vs detection test data
    # (Uses clean_ratio per class for calibration baseline)
    clean_indices = []
    test_indices = []

    for c in range(num_classes):
        c_idx = np.where(evidence.labels == c)[0]
        if len(c_idx) == 0:
            continue
        n_calib = max(int(len(c_idx) * args.clean_ratio), 2)
        clean_indices.extend(c_idx[:n_calib])
        test_indices.extend(c_idx)

    print(f"  Calibration samples: {len(clean_indices)}")
    print(f"  Evaluation samples : {len(test_indices)}")

    print("\n  Fitting class-conditional Gram matrix profiles...")
    detector.fit(
        clean_features=evidence.features[clean_indices],
        clean_labels=evidence.labels[clean_indices]
    )

    print("  Running Gramian deviation analysis & model-level anomaly test...")
    result = detector.detect(
        features=evidence.features[test_indices],
        labels=evidence.labels[test_indices],
        predictions=evidence.predictions[test_indices]
    )

    elapsed_time = time.time() - start_time

    # ------------------------------------------------------------------
    # 5. PRESENT DETECTION RESULTS
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DETECTION RESULT SUMMARY")
    print("=" * 70)
    print(f"Final Decision       : {result.decision}")
    print(f"Standardized Score   : {result.score:.4f} ({result.score * 100:.1f}%)")
    print(f"Max Anomaly Index J* : {result.anomaly_index:.4f} (Threshold: {args.anomaly_threshold})")

    if result.is_backdoored:
        target_name = dataset.classes[result.suspected_target_class] if result.suspected_target_class < len(dataset.classes) else "Unknown"
        print(f"Suspected Target     : Class {result.suspected_target_class} ('{target_name}')")
    else:
        print("Suspected Target     : None (Model classified as CLEAN)")

    print(f"Flagged Samples      : {result.flags.sum()} / {len(result.flags)} ({result.backdoor_percentage:.2f}%)")
    print(f"Processing Time      : {elapsed_time:.2f} seconds")

    print("\nClass-by-Class Anomaly Indices:")
    print("-" * 50)
    for c in range(num_classes):
        c_name = dataset.classes[c] if c < len(dataset.classes) else f"Class_{c}"
        j_val = result.class_anomaly_indices.get(c, 0.0)
        marker = " <-- SUSPECTED TARGET" if (result.is_backdoored and c == result.suspected_target_class) else ""
        print(f"  Class {c:2d} ({c_name:12s}) : J* = {j_val:.4f}{marker}")

    # ------------------------------------------------------------------
    # 6. SAVE REPORT JSON
    # ------------------------------------------------------------------
    report = {
        "detector": "Beatrix",
        "level": "Representation-Level",
        "model_path": args.checkpoint,
        "dataset_path": args.data_dir,
        "architecture": arch_name,
        "num_classes": num_classes,
        "classes": dataset.classes,
        "decision": result.decision,
        "is_backdoored": result.is_backdoored,
        "score": float(result.score),
        "anomaly_index": float(result.anomaly_index),
        "anomaly_threshold": float(args.anomaly_threshold),
        "suspected_target_class": result.suspected_target_class,
        "suspected_target_class_name": (
            dataset.classes[result.suspected_target_class]
            if (result.is_backdoored and result.suspected_target_class < len(dataset.classes))
            else None
        ),
        "class_anomaly_indices": result.class_anomaly_indices,
        "flagged_samples": int(result.flags.sum()),
        "total_samples": len(result.flags),
        "backdoor_percentage": float(result.backdoor_percentage),
        "average_threshold": float(result.threshold),
        "elapsed_seconds": round(elapsed_time, 2)
    }

    out_path = Path(args.output)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=4)

    print("\n" + "=" * 70)
    print(f"Report saved to: {out_path.resolve()}")
    print("=" * 70 + "\n")

    return report


if __name__ == "__main__":
    main()