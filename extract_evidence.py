import os
import sys
import torch
import numpy as np
from pathlib import Path

# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = Path(
    "/Users/evaelizabeth/Desktop/Adaptive-Ensemble-Backdoor-Attack-Detection"
)

BEATRIX_DIR = PROJECT_ROOT / "defenses" / "Beatrix"

sys.path.insert(0, str(BEATRIX_DIR))
sys.path.insert(1, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

import config

from dataloader import get_dataloader
from networks.models import Generator
from classifier_models import (
    PreActResNet18,
    ResNet18,
    PreActResNet34
)

from Beatrix import create_bd


# ============================================================
# CONFIGURATION
# ============================================================

DATASET = "cifar10"
ATTACK_MODE = "all2all"
TARGET_LABEL = 0

DEVICE = "cpu"

BATCH_SIZE = 256

MAX_SAMPLES = 10000

OUTPUT_DIR = (
    PROJECT_ROOT
    / "evidence"
    / DATASET
    / ATTACK_MODE
    / f"target_{TARGET_LABEL}"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# BUILD ARGUMENTS
# ============================================================

args = [
    "--dataset",
    DATASET,

    "--attack_mode",
    ATTACK_MODE,

    "--target_label",
    str(TARGET_LABEL),

    "--device",
    DEVICE,

    "--gpu",
    "0",

    "--batchsize",
    str(BATCH_SIZE),
    "--num_workers",
    "0",
    "--data_root",
    str(PROJECT_ROOT / "data"),

    "--checkpoints",
    str(PROJECT_ROOT / "checkpoints"),

    "--temps",
    str(PROJECT_ROOT / "temps")
]

opt = config.get_arguments().parse_args(args)


# ============================================================
# DATASET SETTINGS
# ============================================================

if opt.dataset == "cifar10":

    opt.num_classes = 10
    opt.input_height = 32
    opt.input_width = 32
    opt.input_channel = 3

elif opt.dataset == "gtsrb":

    opt.num_classes = 43
    opt.input_height = 32
    opt.input_width = 32
    opt.input_channel = 3

elif opt.dataset == "mnist":

    opt.num_classes = 10
    opt.input_height = 28
    opt.input_width = 28
    opt.input_channel = 1

else:

    raise ValueError(
        "Unsupported dataset: "
        + opt.dataset
    )


# ============================================================
# CHECKPOINT
# ============================================================

checkpoint_path = (
    PROJECT_ROOT
    / "checkpoints"
    / DATASET
    / ATTACK_MODE
    / f"target_{TARGET_LABEL}"
    / f"{ATTACK_MODE}_{DATASET}_ckpt.pth.tar"
)

print()
print("=" * 70)
print("EVIDENCE EXTRACTION")
print("=" * 70)

print("Dataset       :", DATASET)
print("Attack mode   :", ATTACK_MODE)
print("Target label  :", TARGET_LABEL)
print("Device        :", DEVICE)
print("Checkpoint    :", checkpoint_path)
print()


if not checkpoint_path.exists():

    raise FileNotFoundError(
        f"""
Checkpoint not found:

{checkpoint_path}

Train the corresponding backdoor model first.
"""
    )


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("Loading checkpoint...")

state_dict = torch.load(
    checkpoint_path,
    map_location=DEVICE,
    weights_only=False
)

print("Checkpoint loaded.")
print()


# ============================================================
# CLASSIFIER
# ============================================================

print("Loading classifier...")

if DATASET == "cifar10":

    netC = PreActResNet18(
        num_classes=10
    ).to(DEVICE)

elif DATASET == "gtsrb":

    netC = PreActResNet18(
        num_classes=43
    ).to(DEVICE)

elif DATASET == "mnist":

    from networks.models import NetC_MNIST

    netC = NetC_MNIST().to(DEVICE)

else:

    raise ValueError(
        "Unsupported dataset"
    )


netC.load_state_dict(
    state_dict["netC"]
)

netC.eval()

for param in netC.parameters():

    param.requires_grad = False


print("Classifier loaded.")


# ============================================================
# GENERATOR
# ============================================================

print("Loading backdoor generator...")

netG = Generator(
    opt
)

netG.load_state_dict(
    state_dict["netG"]
)

netG.to(DEVICE)
netG.eval()

for param in netG.parameters():

    param.requires_grad = False


print("Generator loaded.")


# ============================================================
# MASK GENERATOR
# ============================================================

print("Loading mask generator...")

netM = Generator(
    opt,
    out_channels=1
)

netM.load_state_dict(
    state_dict["netM"]
)

netM.to(DEVICE)
netM.eval()

for param in netM.parameters():

    param.requires_grad = False


print("Mask generator loaded.")


# ============================================================
# DATALOADER
# ============================================================

print()
print("Loading CIFAR-10 test dataset...")

test_loader = get_dataloader(
    opt,
    train=False
)

print(
    "Number of batches:",
    len(test_loader)
)

print()


# ============================================================
# FEATURE EXTRACTION
# ============================================================

clean_features = []
backdoor_features = []

clean_labels = []
backdoor_labels = []

clean_predictions = []
backdoor_predictions = []

clean_images = []
backdoor_images = []

batch_counter = 0
sample_counter = 0


# ------------------------------------------------------------
# Feature hook
# ------------------------------------------------------------

feature_storage = {}


def hook_fn(module, input_data, output_data):

    feature_storage["features"] = (
        output_data
        .detach()
        .cpu()
    )


# ------------------------------------------------------------
# Hook layer4
# ------------------------------------------------------------

hook_handle = (
    netC.layer4.register_forward_hook(
        hook_fn
    )
)


print("=" * 70)
print("EXTRACTING CLEAN + BACKDOOR EVIDENCE")
print("=" * 70)
print()


with torch.no_grad():

    for images, labels in test_loader:

        if sample_counter >= MAX_SAMPLES:

            break


        # ----------------------------------------------------
        # Move clean images
        # ----------------------------------------------------

        images = images.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )


        # ----------------------------------------------------
        # CLEAN FORWARD PASS
        # ----------------------------------------------------

        feature_storage.clear()

        clean_output = netC(
            images
        )

        clean_feature = (
            feature_storage["features"]
        )

        clean_prediction = torch.argmax(
            clean_output,
            dim=1
        )


        # ----------------------------------------------------
        # CREATE BACKDOOR INPUT
        # ----------------------------------------------------

        (
            bd_images,
            bd_targets,
            patterns,
            masks
        ) = create_bd(
            images,
            labels,
            netG,
            netM,
            opt
        )


        # ----------------------------------------------------
        # BACKDOOR FORWARD PASS
        # ----------------------------------------------------

        feature_storage.clear()

        bd_output = netC(
            bd_images
        )

        bd_feature = (
            feature_storage["features"]
        )

        bd_prediction = torch.argmax(
            bd_output,
            dim=1
        )


        # ----------------------------------------------------
        # STORE FEATURES
        # ----------------------------------------------------

        clean_features.append(
            clean_feature
        )

        backdoor_features.append(
            bd_feature
        )


        # ----------------------------------------------------
        # STORE LABELS
        # ----------------------------------------------------

        clean_labels.append(
            labels.cpu()
        )

        backdoor_labels.append(
            bd_targets.cpu()
        )


        # ----------------------------------------------------
        # STORE PREDICTIONS
        # ----------------------------------------------------

        clean_predictions.append(
            clean_prediction.cpu()
        )

        backdoor_predictions.append(
            bd_prediction.cpu()
        )


        # ----------------------------------------------------
        # STORE IMAGES
        # ----------------------------------------------------

        clean_images.append(
            images.cpu()
        )

        backdoor_images.append(
            bd_images.cpu()
        )


        sample_counter += images.shape[0]

        batch_counter += 1


        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        print(
            f"Batch {batch_counter:03d} | "
            f"Samples extracted: "
            f"{min(sample_counter, MAX_SAMPLES)}"
        )


# ============================================================
# REMOVE HOOK
# ============================================================

hook_handle.remove()


# ============================================================
# CONCATENATE
# ============================================================

clean_features = torch.cat(
    clean_features,
    dim=0
)

backdoor_features = torch.cat(
    backdoor_features,
    dim=0
)

clean_labels = torch.cat(
    clean_labels,
    dim=0
)

backdoor_labels = torch.cat(
    backdoor_labels,
    dim=0
)

clean_predictions = torch.cat(
    clean_predictions,
    dim=0
)

backdoor_predictions = torch.cat(
    backdoor_predictions,
    dim=0
)

clean_images = torch.cat(
    clean_images,
    dim=0
)

backdoor_images = torch.cat(
    backdoor_images,
    dim=0
)


# ============================================================
# LIMIT TO MAX SAMPLES
# ============================================================

clean_features = clean_features[:MAX_SAMPLES]
backdoor_features = backdoor_features[:MAX_SAMPLES]

clean_labels = clean_labels[:MAX_SAMPLES]
backdoor_labels = backdoor_labels[:MAX_SAMPLES]

clean_predictions = clean_predictions[:MAX_SAMPLES]
backdoor_predictions = backdoor_predictions[:MAX_SAMPLES]

clean_images = clean_images[:MAX_SAMPLES]
backdoor_images = backdoor_images[:MAX_SAMPLES]


# ============================================================
# PRINT SHAPES
# ============================================================

print()
print("=" * 70)
print("EVIDENCE EXTRACTION COMPLETED")
print("=" * 70)

print()
print("CLEAN CIFAR-10")
print("----------------------------")
print(
    "Images shape      :",
    tuple(clean_images.shape)
)
print(
    "Labels shape      :",
    tuple(clean_labels.shape)
)
print(
    "Features shape    :",
    tuple(clean_features.shape)
)
print(
    "Predictions shape :",
    tuple(clean_predictions.shape)
)

print()
print("BACKDOOR CIFAR-10")
print("----------------------------")
print(
    "Images shape      :",
    tuple(backdoor_images.shape)
)
print(
    "Labels shape      :",
    tuple(backdoor_labels.shape)
)
print(
    "Features shape    :",
    tuple(backdoor_features.shape)
)
print(
    "Predictions shape :",
    tuple(backdoor_predictions.shape)
)


# ============================================================
# CLEAN ACCURACY
# ============================================================

clean_accuracy = (
    (
        clean_predictions
        == clean_labels
    )
    .float()
    .mean()
    .item()
    * 100
)


# ============================================================
# BACKDOOR ATTACK SUCCESS RATE
# ============================================================

if ATTACK_MODE == "all2one":

    attack_success = (
        (
            backdoor_predictions
            == TARGET_LABEL
        )
        .float()
        .mean()
        .item()
        * 100
    )

else:

    attack_success = (
        (
            backdoor_predictions
            == backdoor_labels
        )
        .float()
        .mean()
        .item()
        * 100
    )


print()
print("CLEAN ACCURACY       : {:.2f}%".format(
    clean_accuracy
))

print(
    "BACKDOOR SUCCESS     : {:.2f}%".format(
        attack_success
    )
)


# ============================================================
# SAVE COMPLETE EVIDENCE
# ============================================================

evidence_path = (
    OUTPUT_DIR
    / "cifar10_evidence.pt"
)


evidence = {

    "dataset":
        DATASET,

    "attack_mode":
        ATTACK_MODE,

    "target_label":
        TARGET_LABEL,

    "clean_images":
        clean_images,

    "backdoor_images":
        backdoor_images,

    "clean_labels":
        clean_labels,

    "backdoor_labels":
        backdoor_labels,

    "clean_predictions":
        clean_predictions,

    "backdoor_predictions":
        backdoor_predictions,

    "clean_features":
        clean_features,

    "backdoor_features":
        backdoor_features,

    "clean_accuracy":
        clean_accuracy,

    "backdoor_success":
        attack_success,

    "feature_layer":
        "layer4"
}


torch.save(
    evidence,
    evidence_path
)


# ============================================================
# SAVE FEATURE-ONLY DATA
# ============================================================

feature_path = (
    OUTPUT_DIR
    / "feature_evidence.pt"
)


feature_evidence = {

    "clean_features":
        clean_features,

    "backdoor_features":
        backdoor_features,

    "clean_labels":
        clean_labels,

    "backdoor_labels":
        backdoor_labels,

    "clean_predictions":
        clean_predictions,

    "backdoor_predictions":
        backdoor_predictions
}


torch.save(
    feature_evidence,
    feature_path
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_path = (
    OUTPUT_DIR
    / "evidence_summary.txt"
)


with open(
    summary_path,
    "w"
) as f:

    f.write(
        "CIFAR-10 BACKDOOR EVIDENCE EXTRACTION\n"
    )

    f.write(
        "======================================\n\n"
    )

    f.write(
        f"Dataset: {DATASET}\n"
    )

    f.write(
        f"Attack mode: {ATTACK_MODE}\n"
    )

    f.write(
        f"Target label: {TARGET_LABEL}\n"
    )

    f.write(
        f"Number of samples: {len(clean_labels)}\n\n"
    )

    f.write(
        "CLEAN DATA\n"
    )

    f.write(
        f"Images shape: {tuple(clean_images.shape)}\n"
    )

    f.write(
        f"Features shape: {tuple(clean_features.shape)}\n"
    )

    f.write(
        f"Clean accuracy: {clean_accuracy:.2f}%\n\n"
    )

    f.write(
        "BACKDOOR DATA\n"
    )

    f.write(
        f"Images shape: {tuple(backdoor_images.shape)}\n"
    )

    f.write(
        f"Features shape: {tuple(backdoor_features.shape)}\n"
    )

    f.write(
        f"Backdoor success: {attack_success:.2f}%\n\n"
    )

    f.write(
        "FEATURE EXTRACTION LAYER\n"
    )

    f.write(
        "layer4\n"
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("FILES SAVED")
print("=" * 70)

print()
print(
    "Complete evidence:"
)

print(
    evidence_path
)

print()
print(
    "Feature evidence:"
)

print(
    feature_path
)

print()
print(
    "Summary:"
)

print(
    summary_path
)

print()
print("=" * 70)
print("DONE")
print("=" * 70)
