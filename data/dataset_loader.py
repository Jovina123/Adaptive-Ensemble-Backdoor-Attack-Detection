import os
from pathlib import Path
from PIL import Image
import numpy as np


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp"
}


class UniversalImageDataset:

    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.dataset_path}"
            )

        self.samples = []
        self.classes = []

        self._discover_dataset()

    def _discover_dataset(self):

        # --------------------------------------------------
        # CASE 1:
        # Dataset contains class folders
        #
        # dataset/
        #     cat/
        #     dog/
        #     car/
        # --------------------------------------------------

        directories = [
            p for p in self.dataset_path.iterdir()
            if p.is_dir()
        ]

        if directories:

            self.classes = sorted(
                [p.name for p in directories]
            )

            for class_index, class_dir in enumerate(
                sorted(directories)
            ):

                for image_path in class_dir.rglob("*"):

                    if (
                        image_path.is_file()
                        and image_path.suffix.lower()
                        in SUPPORTED_EXTENSIONS
                    ):

                        self.samples.append({
                            "path": str(image_path),
                            "class_name": class_dir.name,
                            "class_index": class_index
                        })

        # --------------------------------------------------
        # CASE 2:
        # Flat dataset
        #
        # dataset/
        #     image1.jpg
        #     image2.jpg
        #     image3.png
        #
        # No class labels assumed.
        # --------------------------------------------------

        else:

            for image_path in self.dataset_path.rglob("*"):

                if (
                    image_path.is_file()
                    and image_path.suffix.lower()
                    in SUPPORTED_EXTENSIONS
                ):

                    self.samples.append({
                        "path": str(image_path),
                        "class_name": None,
                        "class_index": None
                    })

    def __len__(self):
        return len(self.samples)

    def get_paths(self):
        return [
            sample["path"]
            for sample in self.samples
        ]

    def get_labels(self):

        return [
            sample["class_index"]
            for sample in self.samples
        ]

    def get_class_names(self):

        return self.classes

    def load_images(
        self,
        image_size=(224, 224)
    ):

        images = []

        for sample in self.samples:

            try:

                image = Image.open(
                    sample["path"]
                ).convert("RGB")

                image = image.resize(
                    image_size
                )

                image = np.asarray(
                    image,
                    dtype=np.float32
                )

                images.append(image)

            except Exception as error:

                print(
                    f"Could not load "
                    f"{sample['path']}: {error}"
                )

        if not images:
            raise RuntimeError(
                "No valid images were found."
            )

        return np.array(images)

    def summary(self):

        print("\n========== DATASET ==========")

        print(
            f"Dataset path : {self.dataset_path}"
        )

        print(
            f"Images       : {len(self.samples)}"
        )

        print(
            f"Classes      : "
            f"{len(self.classes) if self.classes else 'Unknown'}"
        )

        if self.classes:

            print(
                "Class names  : "
                + ", ".join(self.classes)
            )

        else:

            print(
                "Class names  : Not available"
            )

        print("=============================\n")