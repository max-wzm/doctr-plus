import json
import math
import os
import random
import warnings
from os.path import join as pjoin
from random import randint

import cv2
import h5py as h5
import numpy as np
import torch

from data_process import GRID_SIZE, IMG_SIZE
from data_process.base import BaseDataset, get_geometric_transform
from data_process.utils import ImageInfo, do_tight_crop, numpy_unwarping, resize_bm


def trigger(prob):
    return random.uniform(0, 1) < prob


def get_bound_crop():
    bound = (0, 448, 0, 448)
    if trigger(0.4):
        return bound

    h, w = randint(2, 3), randint(2, 3)
    top, left = randint(0, 4 - h) * 112, randint(0, 4 - w) * 112
    bound = (top, top + h * 112, left, left + w * 112)
    if trigger(0.3):
        return bound

    h = 4
    if trigger(0.5):
        h, w = w, h
    top, left = randint(0, 4 - h) * 112, randint(0, 4 - w) * 112
    bound = (top, top + h * 112, left, left + w * 112)
    if trigger(0.5):
        return bound

    top, left = randint(30, 224), randint(30, 224)
    btm, right = randint(top + 100, 400), randint(left + 100, 400)
    bound = (top, btm, left, right)
    return bound


class UVDocDataset(BaseDataset):
    """
    Torch dataset class for the UVDoc dataset.
    """

    def __init__(
        self,
        uv_data_path=["./data/UVdoc", "./data/doc3"],
        syn_suffix=["png", "png"],
        appearance_augmentation=[],
        geometric_augmentations=[],
        grid_size=GRID_SIZE,
        split="train",
        train_ratio=0.8,
        total_num=[150000, 150000],
    ) -> None:
        super().__init__(
            appearance_augmentation=appearance_augmentation,
            img_size=IMG_SIZE,
            grid_size=grid_size,
        )
        self.original_grid_size = (89, 61)  # size of the captured data
        self.geometric_transform = get_geometric_transform(
            geometric_augmentations, gridsize=self.original_grid_size
        )

        self.all_samples = ImageInfo.read_from_dataroots(
            uv_data_path, syn_suffix, total_num
        )
        train_ends = int(len(self.all_samples) * train_ratio)
        assert split in ["train", "val"]
        if split == "train":
            self.all_samples = self.all_samples[:train_ends]
        if split == "val":
            self.all_samples = self.all_samples[train_ends:]

    def transform_image(self, img_RGB, grid2D):
        """
        Apply transform toward image and its grid2D

        Args: img_RGB (H, W, C) and grid2D (2, H, W)

        Returns: img_RGB (H, W, C) and grid2D (H, W, 2)

        """
        # Pixel-wise augmentation
        img_RGB = self.appearance_transform(image=img_RGB)["image"]
        # Geometric Augmentations
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            transformed = self.geometric_transform(
                image=img_RGB,
                keypoints=grid2D.transpose(1, 2, 0).reshape(-1, 2),
            )
            img_RGB = transformed["image"]

            grid2D = np.array(transformed["keypoints"]).reshape(
                *self.original_grid_size, 2
            )
        return img_RGB, grid2D

    def __getitem__(self, index):
        try:
            # Get all paths
            image_info = self.all_samples[index]
            img_path = image_info.img_path
            # Load 2D grid, 3D grid and image. Normalize 3D grid
            grid2D = image_info.grid2D
            # print(grid2D_.shape) # 2, 89, 61

            img_RGB = cv2.imread(img_path)
            if not isinstance(img_RGB, np.ndarray):
                with open("requirements.txt", "w") as file:
                    # Write the string to the file
                    string_to_write = f"img from {img_path} is {type(img_RGB)}"
                    file.write(string_to_write)
                return self.__getitem__(randint(0, self.__len__() - 1))
            img_RGB, grid2D = self.transform_image(img_RGB, grid2D)

            h, w, _ = img_RGB.shape
            grid2D = grid2D * 448 / np.array([w, h])
            grid2D = resize_bm(grid2D, (448, 448))
            img_RGB = cv2.resize(img_RGB, (448, 448))

            bnd = get_bound_crop()
            img_RGB, grid2D = do_tight_crop(img_RGB, grid2D, bnd)

            img_RGB = cv2.resize(img_RGB, (288, 288))
            img_RGB = img_RGB.transpose(2, 0, 1) / 255.0

            grid2D = resize_bm(grid2D, (288, 288))
            grid2D = ((grid2D.transpose(2, 0, 1) / 448.0) - 0.5) * 2
            img_RGB_unwarped = numpy_unwarping(img_RGB, grid2D)

            # return as (c, h, w) and range [0, 1] / [-1, 1]
            return (
                img_RGB.astype(np.float32),
                img_RGB_unwarped.astype(np.float32),
                grid2D.astype(np.float32),
            )
        except cv2.error:
            with open("requirements.txt", "w") as file:
                # Write the string to the file
                string_to_write = f"img from {img_path} is error"
                file.write(string_to_write)
            return self.__getitem__(randint(0, self.__len__() - 1))
