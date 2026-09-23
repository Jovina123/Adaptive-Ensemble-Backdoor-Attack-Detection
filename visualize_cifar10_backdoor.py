import os
import csv
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

from networks.models import Generator


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

CHECKPOINT = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "cifar10",
    "all2one",
    "target_0",
    "all2one_cifar10_ckpt.pth.tar"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "cifar10_beatrix"
)

CLEAN_DIR = os.path.join(OUTPUT_DIR, "clean")
BACKDOORED_DIR = os.path.join(OUTPUT_DIR, "backdoored")
COMPARISON_DIR = os.path.join(OUTPUT_DIR, "comparisons")

os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(BACKDOORED_DIR, exist_ok=True)
os.makedirs(COMPARISON_DIR, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)
print("Checkpoint:", CHECKPOINT)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
    weights_only=False
)

opt = checkpoint["opt"]

print("\nAttack configuration:")
print("Dataset:", opt.dataset)
print("Attack mode:", opt.attack_mode)
print("Target label:", opt.target_label)
print("Image size:", opt.input_height, "x", opt.input_width)
print("Mask density:", opt.mask_density)


# ============================================================
# LOAD TRIGGER GENERATOR
# ============================================================

print("\nLoading netG...")

netG = Generator(opt)

netG.load_state_dict(checkpoint["netG"])
netG.to(device)
netG.eval()


# ============================================================
# LOAD MASK GENERATOR
# ============================================================

print("Loading netM...")

netM = Generator(opt, out_channels=1)

netM.load_state_dict(checkpoint["netM"])
netM.to(device)
netM.eval()


# ============================================================
# CIFAR-10 DATASET
# ============================================================

print("\nLoading CIFAR-10...")

transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = torchvision.datasets.CIFAR10(
    root=os.path.join(PROJECT_ROOT, "data"),
    train=False,
    download=True,
    transform=transform
)


class_names = [
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


# ============================================================
# FUNCTION: CREATE BACKDOOR IMAGE
# ============================================================

def create_backdoor_image(image):

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():

        # Generate trigger pattern
        pattern = netG(image)

        pattern = netG.normalize_pattern(
            pattern
        )

        # Generate mask
        mask = netM(image)

        mask = netM.threshold(
            mask
        )

        # Apply trigger
        backdoored = (
            (1 - mask) * image
            + mask * pattern
        )

        backdoored = torch.clamp(
            backdoored,
            0,
            1
        )

    return (
        image.squeeze(0).cpu(),
        backdoored.squeeze(0).cpu()
    )


# ============================================================
# SAVE TENSOR AS IMAGE
# ============================================================

def save_tensor_image(tensor, path):

    image = tensor.permute(
        1, 2, 0
    ).numpy()

    image = (image * 255).astype("uint8")

    Image.fromarray(image).save(path)


# ============================================================
# GENERATE DATA
# ============================================================

NUM_IMAGES = 100

print(
    f"\nGenerating {NUM_IMAGES} clean/backdoored image pairs..."
)


metadata_path = os.path.join(
    OUTPUT_DIR,
    "metadata.csv"
)

with open(
    metadata_path,
    "w",
    newline=""
) as csvfile:

    writer = csv.writer(csvfile)

    writer.writerow([
        "sample_id",
        "original_label",
        "original_class",
        "target_label",
        "target_class",
        "clean_image",
        "backdoored_image"
    ])

    for i in range(NUM_IMAGES):

        image, label = dataset[i]

        clean, backdoored = create_backdoor_image(
            image
        )

        clean_name = f"clean_{i:04d}.png"
        backdoor_name = f"backdoor_{i:04d}.png"

        clean_path = os.path.join(
            CLEAN_DIR,
            clean_name
        )

        backdoor_path = os.path.join(
            BACKDOORED_DIR,
            backdoor_name
        )

        save_tensor_image(
            clean,
            clean_path
        )

        save_tensor_image(
            backdoored,
            backdoor_path
        )

        # Create side-by-side comparison
        clean_pil = Image.open(
            clean_path
        ).resize((256, 256))

        backdoor_pil = Image.open(
            backdoor_path
        ).resize((256, 256))

        comparison = Image.new(
            "RGB",
            (512, 256)
        )

        comparison.paste(
            clean_pil,
            (0, 0)
        )

        comparison.paste(
            backdoor_pil,
            (256, 0)
        )

        comparison.save(
            os.path.join(
                COMPARISON_DIR,
                f"comparison_{i:04d}.png"
            )
        )

        writer.writerow([
            i,
            label,
            class_names[label],
            opt.target_label,
            class_names[opt.target_label],
            clean_name,
            backdoor_name
        ])

        if (i + 1) % 10 == 0:
            print(
                f"Generated {i + 1}/{NUM_IMAGES}"
            )


# ============================================================
# COMPLETE
# ============================================================

print("\n========================================")
print("CIFAR-10 BACKDOOR DATASET GENERATED")
print("========================================")

print("\nClean images:")
print(CLEAN_DIR)

print("\nBackdoored images:")
print(BACKDOORED_DIR)

print("\nComparisons:")
print(COMPARISON_DIR)

print("\nMetadata:")
print(metadata_path)

print("\nTarget class:", opt.target_label)
print("Target class name:", class_names[opt.target_label])
