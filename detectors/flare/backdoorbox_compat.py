"""
Compatibility helpers for loading only the BackdoorBox components
required by this project.

BackdoorBox's top-level package eagerly imports all attack modules.
Some of those modules depend on APIs removed from modern PyTorch.
This loader avoids those unrelated imports and loads only the
components required by our experiments.
"""

import os
import sys
import types


def setup_backdoorbox(backdoorbox_root):
    """
    Prepare a minimal BackdoorBox package context.

    Parameters
    ----------
    backdoorbox_root : str
        Path to the external BackdoorBox repository.
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

    # Add BackdoorBox to sys.path so its modules can be resolved.
    if backdoorbox_root not in sys.path:
        sys.path.insert(0, backdoorbox_root)

    # Create minimal package shells.
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


def load_badnets(backdoorbox_root):
    """Load the BackdoorBox BadNets class."""
    setup_backdoorbox(backdoorbox_root)

    from core.attacks.BadNets import BadNets

    return BadNets


def load_resnet(backdoorbox_root):
    """Load the BackdoorBox ResNet class."""
    setup_backdoorbox(backdoorbox_root)

    from core.models.resnet import ResNet

    return ResNet


def load_flare(backdoorbox_root):
    """Load the BackdoorBox FLARE class."""
    setup_backdoorbox(backdoorbox_root)

    from core.defenses.FLARE import FLARE

    return FLARE