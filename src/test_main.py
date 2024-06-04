"""_summary_."""

from __future__ import annotations

import PIL
import torch

from src import main


def test_load_images() -> None:
    """_summary_."""
    loader = main.ImageData("../images/")
    images = loader.load_images()

    assert len(images) > 0

    for image in images:
        assert type(image) == PIL.PngImagePlugin.PngImageFile


def test_process_images() -> None:
    """_summary_."""
    loader = main.ImageData("../images/")
    images = loader.load_images()

    processor = main.ImgProcess(256)
    processed_images = processor.resize_and_gray(images)

    assert len(processed_images) > 0

    for image in processed_images:
        assert type(image) == torch.Tensor


def test_main() -> None:
    """_summary_."""
    loader = main.ImageData("../images/")
    images = loader.load_images()

    processor = main.ImgProcess(256)
    processed_images = processor.resize_and_gray(images)

    pred = main.Predictor()
    results = pred.predict_img(processed_images)

    assert results == [285]
