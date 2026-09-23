"""
CIFAR-10 experiment for the FLARE backdoor detector.

This experiment:
1. Loads CIFAR-10.
2. Creates a BadNets-poisoned training dataset.
3. Trains a ResNet-18 backdoored model using the official
   BackdoorBox CIFAR-10 BadNets training configuration.
4. Builds ground-truth poison labels from the known poisoned indices.
5. Runs FLARE.
6. Evaluates FLARE against the known poison labels.

The heavy computation is intended to run in a CUDA-enabled
environment such as Google Colab.
"""

import os
import sys

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

# BackdoorBox is kept outside the project repository.
BACKDOORBOX_ROOT = os.environ.get("BACKDOORBOX_ROOT")

if BACKDOORBOX_ROOT is None:
    BACKDOORBOX_ROOT = os.path.abspath(
        os.path.join(PROJECT_ROOT, "..", "BackdoorBox")
    )

if not os.path.isdir(BACKDOORBOX_ROOT):
    raise FileNotFoundError(
        f"BackdoorBox directory was not found: {BACKDOORBOX_ROOT}\n"
        "Set the BACKDOORBOX_ROOT environment variable to the "
        "location of your BackdoorBox clone."
    )

if BACKDOORBOX_ROOT not in sys.path:
    sys.path.insert(0, BACKDOORBOX_ROOT)


# ---------------------------------------------------------------------------
# Experiment configuration
#
# These values follow the CIFAR-10 BadNets configuration in the
# BackdoorBox test_BadNets.py file.
# ---------------------------------------------------------------------------

SEED = 666
DETERMINISTIC = True

POISONING_RATE = 0.05
TARGET_LABEL = 1

BATCH_SIZE = 1024
NUM_WORKERS = 4

LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
GAMMA = 0.1
LR_SCHEDULE = [150, 180]

EPOCHS = 200

LOG_ITERATION_INTERVAL = 100
TEST_EPOCH_INTERVAL = 10
SAVE_EPOCH_INTERVAL = 10

SAVE_DIR = "experiments"
EXPERIMENT_NAME = "ResNet-18_CIFAR-10_BadNets"


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("FLARE CIFAR-10 Experiment")
    print("=" * 70)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. "
            "The current BackdoorBox FLARE implementation uses CUDA directly. "
            "Run this experiment in a CUDA-enabled environment such as "
            "Google Colab with a GPU runtime."
        )

    # Import BackdoorBox only after its path has been configured.
    import core

    # -----------------------------------------------------------------------
    # Reproducibility
    # -----------------------------------------------------------------------

    torch.manual_seed(SEED)

    # -----------------------------------------------------------------------
    # CIFAR-10
    # -----------------------------------------------------------------------

    print("\nLoading CIFAR-10...")

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        transform=transform_train,
        download=True,
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        transform=transform_test,
        download=True,
    )

    print(f"Training samples: {len(trainset)}")
    print(f"Test samples: {len(testset)}")

    # -----------------------------------------------------------------------
    # BadNets trigger
    #
    # This follows the CIFAR-10 BadNets configuration in BackdoorBox:
    # a 3x3 white square in the bottom-right corner.
    # -----------------------------------------------------------------------

    pattern = torch.zeros((32, 32), dtype=torch.uint8)
    pattern[-3:, -3:] = 255

    weight = torch.zeros((32, 32), dtype=torch.float32)
    weight[-3:, -3:] = 1.0

    # -----------------------------------------------------------------------
    # Create BadNets experiment
    # -----------------------------------------------------------------------

    print("\nCreating BadNets poisoned dataset...")

    badnets = core.BadNets(
        train_dataset=trainset,
        test_dataset=testset,
        model=core.models.ResNet(18),
        loss=nn.CrossEntropyLoss(),
        y_target=TARGET_LABEL,
        poisoned_rate=POISONING_RATE,
        pattern=pattern,
        weight=weight,
        seed=SEED,
        deterministic=DETERMINISTIC,
    )

    poisoned_trainset, poisoned_testset = badnets.get_poisoned_dataset()

    # Known poisoned indices are our experimental ground truth.
    poison_indices = poisoned_trainset.poisoned_set

    y_true = torch.zeros(len(trainset), dtype=torch.long)
    y_true[list(poison_indices)] = 1

    print(f"Poisoning rate: {POISONING_RATE}")
    print(f"Target label: {TARGET_LABEL}")
    print(f"Known poisoned samples: {len(poison_indices)}")

    # -----------------------------------------------------------------------
    # Train the backdoored model
    #
    # This follows the ResNet-18 CIFAR-10 BadNets schedule in
    # BackdoorBox's test_BadNets.py.
    # -----------------------------------------------------------------------

    schedule = {
        "device": "GPU",
        "CUDA_SELECTED_DEVICES": "0",

        "benign_training": False,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,

        "lr": LEARNING_RATE,
        "momentum": MOMENTUM,
        "weight_decay": WEIGHT_DECAY,
        "gamma": GAMMA,
        "schedule": LR_SCHEDULE,

        "epochs": EPOCHS,

        "log_iteration_interval": LOG_ITERATION_INTERVAL,
        "test_epoch_interval": TEST_EPOCH_INTERVAL,
        "save_epoch_interval": SAVE_EPOCH_INTERVAL,

        "save_dir": SAVE_DIR,
        "experiment_name": EXPERIMENT_NAME,
    }

    print("\nTraining backdoored ResNet-18...")
    print(f"Training epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")

    badnets.train(schedule)

    # The trained model is returned by the BackdoorBox attack object.
    badnet_model = badnets.get_model()

    print("\nBackdoored model training completed.")

    # -----------------------------------------------------------------------
    # FLARE
    # -----------------------------------------------------------------------

    from detectors.flare.flare_detector import FLAREDetector

    print("\nCreating FLARE detector...")

    detector = FLAREDetector(
        model=badnet_model,
        xi=0.02,
        seed=SEED,
        deterministic=DETERMINISTIC,
    )

    print("\nRunning FLARE...")
    print("This stage performs likelihood extraction and clustering.")

    detector.detect(
        poisoned_trainset=poisoned_trainset,
        y_true=y_true,
    )

    print("\nFLARE experiment completed.")


if __name__ == "__main__":
    main()