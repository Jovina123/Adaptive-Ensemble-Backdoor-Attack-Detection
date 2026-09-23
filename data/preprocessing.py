"""
Generic preprocessing for arbitrary image datasets.
"""

from torchvision import transforms


DEFAULT_IMAGE_SIZE = 224


def get_train_transform(image_size=DEFAULT_IMAGE_SIZE):

    return transforms.Compose([
        transforms.Resize(
            (image_size, image_size)
        ),

        transforms.RandomHorizontalFlip(),

        transforms.RandomRotation(10),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_eval_transform(image_size=DEFAULT_IMAGE_SIZE):

    return transforms.Compose([
        transforms.Resize(
            (image_size, image_size)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])