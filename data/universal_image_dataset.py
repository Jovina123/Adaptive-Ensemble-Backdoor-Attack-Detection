"""
Universal Image Dataset Loader

Purpose:
    Load arbitrary image classification datasets without hard-coding
    dataset names, class names, number of classes, or image paths.

Supported structure:

dataset/
    class_a/
        image1.jpg
        image2.jpg
    class_b/
        image3.jpg
        image4.jpg

Also supports common image extensions.
"""

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torch


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp"
}


class UniversalImageDataset(Dataset):

    def __init__(
        self,
        root_dir,
        transform=None,
        class_to_idx=None
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory does not exist: {self.root_dir}"
            )

        # Discover class folders automatically
        if class_to_idx is None:
            self.classes = sorted([
                folder.name
                for folder in self.root_dir.iterdir()
                if folder.is_dir()
            ])

            self.class_to_idx = {
                class_name: idx
                for idx, class_name in enumerate(self.classes)
            }

        else:
            self.class_to_idx = class_to_idx
            self.classes = [
                name for name, _ in sorted(
                    class_to_idx.items(),
                    key=lambda x: x[1]
                )
            ]

        if len(self.classes) == 0:
            raise ValueError(
                f"No class folders found in {self.root_dir}"
            )

        self.samples = []

        for class_name in self.classes:

            class_dir = self.root_dir / class_name

            if not class_dir.exists():
                continue

            class_index = self.class_to_idx[class_name]

            for image_path in class_dir.rglob("*"):

                if (
                    image_path.is_file()
                    and image_path.suffix.lower()
                    in SUPPORTED_EXTENSIONS
                ):
                    self.samples.append(
                        (str(image_path), class_index)
                    )

        if len(self.samples) == 0:
            raise ValueError(
                f"No supported images found in {self.root_dir}"
            )

        print("\n========================================")
        print("UNIVERSAL DATASET")
        print("========================================")
        print(f"Dataset       : {self.root_dir}")
        print(f"Classes       : {len(self.classes)}")
        print(f"Images        : {len(self.samples)}")
        print("Class mapping :")

        for class_name, index in self.class_to_idx.items():
            count = sum(
                1 for _, label in self.samples
                if label == index
            )

            print(
                f"  {index:3d} -> "
                f"{class_name:<25} "
                f"({count} images)"
            )

        print("========================================\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path, label = self.samples[index]

        try:
            image = Image.open(image_path).convert("RGB")

        except Exception as e:
            raise RuntimeError(
                f"Could not read image: {image_path}\n"
                f"Error: {e}"
            )

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def get_class_names(self):
        return self.classes

    def get_class_to_idx(self):
        return self.class_to_idx

    def get_image_paths(self):
        return [path for path, _ in self.samples]