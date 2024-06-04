#!/usr/bin/env python
"""_summary_.

Returns
-------
    _type_: _description_

"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms
from torchvision.models.resnet import ResNet18_Weights


class ImageData:
    """_summary_."""

    def __init__(self, dirr: str) -> None:
        """_summary_.

        Args:
        ----
            dirr (None): _description_

        """
        self.D = dirr

    def load_images(self) -> list[Image.Image]:
        """_summary_.

        Returns
        -------
            _type_: _description_

        """
        return [Image.open(Path(self.D) / Path(f))
                for f in os.listdir(self.D)
                if f.endswith((".jpg", ".png"))]


class ImgProcess:
    """_summary_."""

    def __init__(self, size: int) -> None:
        """_summary_.

        Args:
        ----
            size (None): _description_

        """
        self.s = size

    def resize_and_gray(self, img_list: list[Image.Image]) -> list[transforms.Compose]:
        """_summary_.

        Args:
        ----
            img_list (None): _description_

        Returns:
        -------
            _type_: _description_

        """
        p_images = []
        for img in img_list:
            t = transforms.Compose([
                transforms.Resize((self.s, self.s)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                ),
            ])
            p_images.append(t(img))
        return p_images


class Predictor:
    """_summary_."""

    def __init__(self) -> None:
        """_summary."""
        self.mdl = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.mdl.eval()

    def predict_img(self, processed_images: list[transforms.Compose]) -> list[int | float]:  # noqa: E501
        """_summary.

        Args:
        ----
            processed_images (None): _description_

        Returns:
        -------
            _type_: _description_

        """
        results = []
        for img_tensor in processed_images:
            pred = self.mdl(img_tensor.unsqueeze(0))
            results.append(torch.argmax(pred, dim=1).item())
        return results


if __name__ == "__main__":
    loader = ImageData("../images/")
    images = loader.load_images()

    processor = ImgProcess(256)
    processed_images = processor.resize_and_gray(images)

    pred = Predictor()
    results = pred.predict_img(processed_images)
