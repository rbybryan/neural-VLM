"""Image transforms for the visual encoder.

ConvNeXt-V2 was pretrained on ImageNet-22k with standard ImageNet
normalization. We use the same statistics. Higher input resolution (e.g.,
448x448) may improve performance on text-heavy benchmarks (DocVQA, ChartQA)
and is a v2 lever; v1 uses 224 to keep compute tight.
"""

from __future__ import annotations

from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_eval_transform(image_size: int = 224):
    return build_train_transform(image_size)
