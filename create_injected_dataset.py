"""
create_injected_dataset.py

Creates a poisoned evaluation dataset from the clean dataset.

Input:
    pet_dataset/train

Output:
    pet_dataset/train_injected_new
    pet_dataset/poison_manifest.csv
"""

import argparse
import csv
import os
import random
import shutil

from PIL import Image
import torchvision.transforms.functional as TF

from custom_trigger import apply_trigger


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        type=str,
        default="pet_dataset/train"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="pet_dataset/train_injected_new"
    )

    parser.add_argument(
        "--poison_ratio",
        type=float,
        default=0.15
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    source = args.source
    output = args.output

    print("=" * 60)
    print("CREATING INJECTED DATASET")
    print("=" * 60)

    print(f"Source        : {source}")
    print(f"Output        : {output}")
    print(f"Poison ratio  : {args.poison_ratio}")
    print(f"Random seed   : {args.seed}")
    print()

    if not os.path.exists(source):
        raise FileNotFoundError(
            f"Source dataset not found: {source}"
        )

    # ---------------------------------------------------------
    # Find classes
    # ---------------------------------------------------------

    classes = sorted(
        [
            d
            for d in os.listdir(source)
            if os.path.isdir(
                os.path.join(source, d)
            )
        ]
    )

    class_to_idx = {
        class_name: idx
        for idx, class_name in enumerate(classes)
    }

    # ---------------------------------------------------------
    # Collect images
    # ---------------------------------------------------------

    samples = []

    for class_name in classes:

        class_dir = os.path.join(
            source,
            class_name
        )

        for filename in sorted(
            os.listdir(class_dir)
        ):

            filepath = os.path.join(
                class_dir,
                filename
            )

            if not os.path.isfile(filepath):
                continue

            extension = os.path.splitext(
                filename
            )[1].lower()

            if extension not in [
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".webp"
            ]:
                continue

            samples.append(
                {
                    "source_path": filepath,
                    "filename": filename,
                    "class_name": class_name,
                    "class_id": class_to_idx[class_name]
                }
            )

    print(
        f"Found {len(samples)} images "
        f"across {len(classes)} classes."
    )

    if len(samples) == 0:
        raise RuntimeError(
            "No images were found in the source dataset."
        )

    # ---------------------------------------------------------
    # Select poisoned samples
    # ---------------------------------------------------------

    rng = random.Random(args.seed)

    indices = list(
        range(len(samples))
    )

    rng.shuffle(indices)

    n_poison = int(
        len(samples) * args.poison_ratio
    )

    poison_indices = set(
        indices[:n_poison]
    )

    print(
        f"Poisoning {n_poison} "
        f"of {len(samples)} images."
    )

    # ---------------------------------------------------------
    # Remove old output if it exists
    # ---------------------------------------------------------

    if os.path.exists(output):

        print(
            f"Removing existing output: {output}"
        )

        shutil.rmtree(output)

    os.makedirs(
        output,
        exist_ok=True
    )

    # ---------------------------------------------------------
    # Create class folders
    # ---------------------------------------------------------

    for class_name in classes:

        os.makedirs(
            os.path.join(
                output,
                class_name
            ),
            exist_ok=True
        )

    # ---------------------------------------------------------
    # Manifest
    # ---------------------------------------------------------

    manifest_path = os.path.join(
        os.path.dirname(output),
        "poison_manifest.csv"
    )

    manifest_rows = []

    # ---------------------------------------------------------
    # Process every image
    # ---------------------------------------------------------

    for idx, sample in enumerate(samples):

        source_path = sample[
            "source_path"
        ]

        filename = sample[
            "filename"
        ]

        class_name = sample[
            "class_name"
        ]

        class_id = sample[
            "class_id"
        ]

        is_poisoned = (
            idx in poison_indices
        )

        output_path = os.path.join(
            output,
            class_name,
            filename
        )

        # -----------------------------------------------------
        # Poisoned image
        # -----------------------------------------------------

        if is_poisoned:

            image = Image.open(
                source_path
            ).convert("RGB")

            tensor = TF.to_tensor(
                image
            )

            triggered = apply_trigger(
                tensor
            )

            triggered = triggered.clamp(
                0,
                1
            )

            triggered = TF.to_pil_image(
                triggered
            )

            triggered.save(
                output_path
            )

        # -----------------------------------------------------
        # Clean image
        # -----------------------------------------------------

        else:

            shutil.copy2(
                source_path,
                output_path
            )

        # -----------------------------------------------------
        # Save ground truth
        # -----------------------------------------------------

        manifest_rows.append(
            {
                "index": idx,
                "filename": filename,
                "relative_path":
                    f"{class_name}/{filename}",
                "class": class_id,
                "poisoned":
                    int(is_poisoned)
            }
        )

    # ---------------------------------------------------------
    # Save manifest
    # ---------------------------------------------------------

    with open(
        manifest_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "filename",
                "relative_path",
                "class",
                "poisoned"
            ]
        )

        writer.writeheader()

        writer.writerows(
            manifest_rows
        )

    # ---------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------

    clean_count = (
        len(samples) - n_poison
    )

    print()
    print("=" * 60)
    print("INJECTED DATASET CREATED")
    print("=" * 60)

    print(
        f"Total images       : {len(samples)}"
    )

    print(
        f"Poisoned images    : {n_poison}"
    )

    print(
        f"Clean images       : {clean_count}"
    )

    print(
        f"Actual ratio       : "
        f"{n_poison / len(samples):.2%}"
    )

    print(
        f"Dataset            : {output}"
    )

    print(
        f"Manifest           : {manifest_path}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()