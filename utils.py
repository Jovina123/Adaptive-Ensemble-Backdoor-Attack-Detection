"""Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
"""
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.init as init

import platform

def get_mean_and_std(dataset):
    """Compute the mean and std value of dataset."""
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print("==> Computing mean and std..")
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    """Init layer parameters."""
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode="fan_out")
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

if platform.system() == 'Windows':
    term_width = 100
else:
    _, term_width = os.popen("stty size", "r").read().split()
    term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.0
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(" [")
    for i in range(cur_len):
        sys.stdout.write("=")
    sys.stdout.write(">")
    for i in range(rest_len):
        sys.stdout.write(".")
    sys.stdout.write("]")

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    if msg:
        L.append(" | " + msg)

    msg = "".join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(" ")

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write("\b")
    sys.stdout.write(" %d/%d " % (current + 1, total))

    if current < total - 1:
        sys.stdout.write("\r")
    else:
        sys.stdout.write("\n")
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ""
    i = 1
    if days > 0:
        f += str(days) + "D"
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + "h"
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + "m"
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + "s"
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + "ms"
        i += 1
    if f == "":
        f = "0ms"
    return f


def load_model_from_checkpoint(checkpoint_path, device="cpu", num_classes=None):
    """
    Universally load a classifier model from a checkpoint.
    Supports:
        - PreActResNet18 / CIFAR-10 / GTSRB checkpoints containing 'netC'.
        - Torchvision ResNet-50 checkpoints containing 'state_dict'.
        - Generic PyTorch checkpoints.
    Returns:
        (model, arch_name, num_classes, input_size, checkpoint_dict)
    """
    from pathlib import Path
    import torchvision.models as models
    from classifier_models import PreActResNet18

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint must be a dictionary.")

    if "netC" in ckpt:
        opt = ckpt.get("opt", None)
        n_classes = num_classes or (getattr(opt, "num_classes", 10) if opt else 10)
        dataset_name = getattr(opt, "dataset", "cifar10") if opt else "cifar10"
        input_size = 28 if dataset_name == "mnist" else 32

        model = PreActResNet18(num_classes=n_classes)
        model.load_state_dict(ckpt["netC"], strict=True)
        arch = "PreActResNet18"

    elif "state_dict" in ckpt:
        n_classes = int(ckpt.get("num_classes", num_classes or 2))
        arch = ckpt.get("architecture", "resnet50")
        input_size = 224

        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
        model.load_state_dict(ckpt["state_dict"], strict=True)

    else:
        raise ValueError("Unrecognized checkpoint format (neither 'netC' nor 'state_dict' found).")

    model = model.to(device).eval()
    return model, arch, n_classes, input_size, ckpt
