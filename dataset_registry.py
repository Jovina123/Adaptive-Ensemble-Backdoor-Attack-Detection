"""
dataset_registry.py

Central place to register datasets. Instead of if/elif chains for
--dataset scattered across main.py, train.py, and dataloader.py, every
dataset is registered here once with everything needed to use it:
number of classes, input shape, and which classifier architecture to
use (must be a ResNet-family model with a `layer4` submodule -- or
whatever layer you point the hook at -- for Beatrix's feature hook to
work; see HOOK_LAYER below).

To add a new dataset:
    register_dataset(
        name="my_dataset",
        num_classes=20,
        input_shape=(3, 32, 32),
        model_name="PreActResNet18",
    )

Then --dataset my_dataset works everywhere without touching main.py,
train.py, or dataloader.py's branching logic.
"""

DATASET_REGISTRY = {}


def register_dataset(name, num_classes, input_shape, model_name="PreActResNet18", hook_layer="layer4"):
    """
    input_shape: (channels, height, width)
    model_name: must be a class importable from classifier_models
                (PreActResNet18, ResNet18, PreActResNet34, or your own
                addition -- as long as it has a submodule named
                hook_layer for the feature hook to attach to).
    hook_layer: name of the submodule Beatrix's LayerActivations hooks
                into. Defaults to "layer4" (matches all the ResNet
                variants already in classifier_models/). Override this
                per-dataset if you add an architecture with a
                differently-named final block.
    """
    DATASET_REGISTRY[name] = {
        "num_classes": num_classes,
        "input_channel": input_shape[0],
        "input_height": input_shape[1],
        "input_width": input_shape[2],
        "model_name": model_name,
        "hook_layer": hook_layer,
    }


def get_dataset_config(name):
    if name not in DATASET_REGISTRY:
        available = ", ".join(DATASET_REGISTRY.keys())
        raise ValueError(
            f"Unknown dataset '{name}'. Registered datasets: {available}. "
            f"Add yours with register_dataset() in dataset_registry.py."
        )
    return DATASET_REGISTRY[name]


def apply_dataset_config(opt):
    """
    Fills in opt.num_classes, opt.input_channel/height/width, and
    opt.hook_layer from the registry based on opt.dataset. Call this
    once near the top of main.py / train.py instead of hand-written
    if/elif blocks.
    """
    cfg = get_dataset_config(opt.dataset)
    opt.num_classes = cfg["num_classes"]
    opt.input_channel = cfg["input_channel"]
    opt.input_height = cfg["input_height"]
    opt.input_width = cfg["input_width"]
    opt.model_name = cfg["model_name"]
    opt.hook_layer = cfg["hook_layer"]
    return opt


# Register the datasets already supported by this project.
register_dataset("mnist", num_classes=10, input_shape=(1, 28, 28), model_name="NetC_MNIST", hook_layer=None)
register_dataset("cifar10", num_classes=10, input_shape=(3, 32, 32), model_name="PreActResNet18")
register_dataset("gtsrb", num_classes=43, input_shape=(3, 32, 32), model_name="PreActResNet18")

# hook_layer=None for MNIST is intentional: NetC_MNIST has no layer4
# (or equivalent), so Beatrix's representation-level detector cannot
# run on it as-is. main.py should check for this and skip/warn rather
# than crash -- see the updated main.py.
