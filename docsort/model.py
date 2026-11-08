"""The image classifier: an ImageNet-pretrained CNN with a new 7-class head (transfer learning)."""
from __future__ import annotations

import torch
from torch import nn
from torchvision import models, transforms

IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def build_model(num_classes: int, arch: str = "efficientnet_b0", pretrained: bool = True) -> nn.Module:
    if arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif arch == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    else:
        raise ValueError(f"unknown arch {arch}")
    return model


def head_parameters(model: nn.Module) -> list[nn.Parameter]:
    return list(model.classifier.parameters())


def eval_transform(size: int) -> transforms.Compose:
    # The whole page matters (layout is the strongest clue), so we resize instead of cropping.
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((size, size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def train_transform(size: int) -> transforms.Compose:
    # Small, realistic changes only: slight tilt / shift / zoom and lighting. No flips — documents aren't mirrored.
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((size, size), antialias=True),
        transforms.RandomAffine(degrees=3, translate=(0.03, 0.03), scale=(0.95, 1.05), fill=255),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_checkpoint(path, device: str = "cpu") -> tuple[nn.Module, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(len(ckpt["classes"]), ckpt["arch"], pretrained=False)
    model.load_state_dict({k: v.float() for k, v in ckpt["state_dict"].items()})
    model.eval()
    return model, ckpt
