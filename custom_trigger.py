"""
custom_trigger.py

Generic backdoor trigger functions.
Works with any RGB image classification dataset.
"""

import torch


def solid_patch_trigger(
    image,
    patch_size=16,
    color=(1.0, 0.0, 0.0),
    position="bottom_right"
):
    """
    Applies a solid-color square trigger.

    image:
        Tensor with shape (C, H, W), values in [0, 1]

    patch_size:
        Size of the square trigger.

    color:
        RGB color tuple.

    position:
        bottom_right, bottom_left, top_right, or top_left
    """

    triggered = image.clone()

    _, h, w = triggered.shape

    # Make sure the trigger cannot exceed the image dimensions
    patch_size = min(patch_size, h, w)

    if position == "bottom_right":
        y0 = h - patch_size
        x0 = w - patch_size

    elif position == "bottom_left":
        y0 = h - patch_size
        x0 = 0

    elif position == "top_right":
        y0 = 0
        x0 = w - patch_size

    else:
        # top_left
        y0 = 0
        x0 = 0

    # Apply RGB trigger
    for c in range(min(3, triggered.shape[0])):
        triggered[
            c,
            y0:y0 + patch_size,
            x0:x0 + patch_size
        ] = color[c]

    return triggered


def checkerboard_trigger(
    image,
    patch_size=16,
    position="bottom_right"
):
    """
    Applies a black/white checkerboard trigger.
    """

    triggered = image.clone()

    _, h, w = triggered.shape

    patch_size = min(patch_size, h, w)

    if position == "bottom_right":
        y0 = h - patch_size
        x0 = w - patch_size

    elif position == "bottom_left":
        y0 = h - patch_size
        x0 = 0

    elif position == "top_right":
        y0 = 0
        x0 = w - patch_size

    else:
        y0 = 0
        x0 = 0

    checker = torch.zeros(
        patch_size,
        patch_size,
        dtype=triggered.dtype,
        device=triggered.device
    )

    checker[::2, ::2] = 1.0
    checker[1::2, 1::2] = 1.0

    for c in range(triggered.shape[0]):
        triggered[
            c,
            y0:y0 + patch_size,
            x0:x0 + patch_size
        ] = checker

    return triggered


def blended_watermark_trigger(
    image,
    alpha=0.15,
    pattern_seed=42
):
    """
    Applies a faint fixed noise pattern across the image.
    """

    generator = torch.Generator()

    generator.manual_seed(pattern_seed)

    pattern = torch.rand(
        image.shape,
        generator=generator,
        dtype=image.dtype
    )

    pattern = pattern.to(image.device)

    triggered = (
        (1 - alpha) * image
        + alpha * pattern
    )

    return triggered.clamp(0, 1)


# Default trigger used by the training and dataset-generation scripts.
DEFAULT_TRIGGER = solid_patch_trigger


def apply_trigger(
    image,
    trigger_fn=None,
    **kwargs
):
    """
    Generic trigger wrapper.
    """

    fn = trigger_fn or DEFAULT_TRIGGER

    return fn(image, **kwargs)