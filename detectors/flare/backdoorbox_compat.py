"""
Compatibility helpers for loading only the BackdoorBox components
required by this project.

BackdoorBox's top-level package eagerly imports all attack modules.
Some of those modules depend on APIs removed from modern PyTorch.
This loader avoids those unrelated imports and loads only the
components required by our experiments.
"""

import importlib
import os
import sys
import types


def setup_backdoorbox(backdoorbox_root):
    """
    Prepare a minimal BackdoorBox package context without importing
    BackdoorBox's top-level package.
    """

    backdoorbox_root = os.path.abspath(backdoorbox_root)

    if not os.path.isdir(backdoorbox_root):
        raise FileNotFoundError(
            f"BackdoorBox directory not found: {backdoorbox_root}"
        )

    core_root = os.path.join(backdoorbox_root, "core")

    if not os.path.isdir(core_root):
        raise FileNotFoundError(
            f"BackdoorBox core directory not found: {core_root}"
        )

    if backdoorbox_root not in sys.path:
        sys.path.insert(0, backdoorbox_root)

    package_paths = {
        "core": core_root,
        "core.attacks": os.path.join(core_root, "attacks"),
        "core.defenses": os.path.join(core_root, "defenses"),
        "core.utils": os.path.join(core_root, "utils"),
        "core.models": os.path.join(core_root, "models"),
    }

    for package_name, package_path in package_paths.items():
        if not os.path.isdir(package_path):
            raise FileNotFoundError(
                f"BackdoorBox package directory not found: {package_path}"
            )

        if package_name not in sys.modules:
            package = types.ModuleType(package_name)
            package.__path__ = [package_path]
            sys.modules[package_name] = package

    # BackdoorBox's attack base class imports Log directly from
    # core.utils. Load that symbol explicitly without executing
    # core.utils.__init__, which would import unrelated modules.
    log_module = importlib.import_module("core.utils.log")
    sys.modules["core.utils"].Log = log_module.Log

    # FLARE imports the test utility directly from core.utils.
    # Load the actual function and expose it on the package shell.
    test_module = importlib.import_module("core.utils.test")
    sys.modules["core.utils"].test = test_module.test


def load_badnets(backdoorbox_root):
    """Load the BackdoorBox BadNets class."""

    setup_backdoorbox(backdoorbox_root)

    from core.attacks.BadNets import BadNets

    return BadNets


def load_resnet(backdoorbox_root):
    """Load the BackdoorBox ResNet model."""

    setup_backdoorbox(backdoorbox_root)

    from core.models.resnet import ResNet

    return ResNet


def load_flare(backdoorbox_root):
    """Load the BackdoorBox FLARE defense."""

    setup_backdoorbox(backdoorbox_root)

    from core.defenses.FLARE import FLARE

    return FLARE