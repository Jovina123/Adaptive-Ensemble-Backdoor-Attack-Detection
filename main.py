"""
Entry point for Adaptive-Ensemble-Backdoor-Attack-Detection.

Usage:
    python main.py --dataset cifar10 --gpu 0 --target_label 0

This loads a trained (possibly backdoored) classifier + trigger
generator from a checkpoint, extracts intermediate-layer activations
for clean and triggered inputs, then runs them through the ensemble
detector (currently just Beatrix — see ensemble.py to add more).
"""

import os

from defenses.Beatrix.config import get_argument
from defenses.Beatrix.Beatrix import train as load_features
from ensemble import EnsembleDetector


def main():
    opt = get_argument().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

    if opt.dataset in ("mnist", "cifar10"):
        opt.num_classes = 10
    elif opt.dataset == "gtsrb":
        opt.num_classes = 43
    else:
        raise ValueError(f"Unsupported dataset: {opt.dataset}")

    if opt.dataset in ("cifar10", "gtsrb"):
        opt.input_height, opt.input_width, opt.input_channel = 32, 32, 3
    elif opt.dataset == "mnist":
        opt.input_height, opt.input_width, opt.input_channel = 28, 28, 1

    import torch
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loads/trains the classifier + trigger generator, then runs a forward
    # pass over clean and backdoor-triggered inputs, returning the hooked
    # intermediate-layer features (this is defined in Beatrix.py).
    print(f"Loading model and extracting features for target_label={opt.target_label} ...")
    data = load_features(opt)

    clean_features = data["clean_feature"]
    bd_features = data["bd_feature"]
    ori_labels = data["ori_label"].cpu().numpy()
    bd_labels = data["bd_label"].cpu().numpy()

    ensemble = EnsembleDetector(num_classes=opt.num_classes)

    # Calibrate on clean data only, per class.
    print("Fitting detectors on clean activations ...")
    ensemble.fit(clean_features, ori_labels)

    # Score the backdoor-triggered activations.
    print("Scoring backdoor-triggered activations ...")
    scores = ensemble.score(bd_features, bd_labels)
    flags = ensemble.flagged(bd_features, bd_labels, percentile="p95")

    flagged_rate = flags.mean() * 100
    print(f"\nFlagged {flagged_rate:.1f}% of triggered samples as backdoored "
          f"(mean anomaly score: {scores.mean():.4f})")


if __name__ == "__main__":
    main()
