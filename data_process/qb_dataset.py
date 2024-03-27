import os
import random
import warnings
from os.path import join as pjoin
from random import randint

import cv2
import hdf5storage as h5
import numpy as np

from data_process import GRID_SIZE, IMG_SIZE
from data_process.base import BaseDataset, get_geometric_transform
from data_process.utils import do_tight_crop, numpy_unwarping, resize_bm


def trigger(prob):
    return random.uniform(0, 1) < prob


def get_bound_crop():
    bound = (0, 448, 0, 448)
    # return bound
    if trigger(0.7):
        return bound

    h, w = randint(2, 3), randint(2, 3)
    top, left = randint(0, 4 - h) * 112, randint(0, 4 - w) * 112
    bound = (top, top + h * 112, left, left + w * 112)
    if trigger(0.4):
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


class QbDataset(BaseDataset):
    """
    Torch dataset class for the QBDoc dataset.
    """

    def __init__(
        self,
        data_path="./data/QBdoc",
        appearance_augmentation=[],
        geometric_augmentations=[],
        grid_size=GRID_SIZE,
        split="train",
        train_ratio=0.5,
    ) -> None:
        super().__init__(
            data_path=data_path,
            appearance_augmentation=appearance_augmentation,
            img_size=IMG_SIZE,
            grid_size=grid_size,
        )
        self.original_grid_size = (448, 448)  # size of the captured data
        self.geometric_transform = get_geometric_transform(
            geometric_augmentations, gridsize=self.original_grid_size
        )

        self.all_samples = [
            x[:-4]
            for x in os.listdir(pjoin(self.dataroot, "img"))
            if x.endswith(".jpg")
        ]
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
        # Get all paths
        sample_id = self.all_samples[index]
        img_path = pjoin(self.dataroot, "img", f"{sample_id}.jpg")
        bm_path = pjoin(self.dataroot, "bm", f"{sample_id}.mat")
        img_RGB = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)[:, :, ::-1]
        bm = h5.loadmat(bm_path)["bm"].transpose(2, 0, 1)

        img_RGB, bm = self.transform_image(img_RGB, bm)
        h, w, _ = img_RGB.shape
        bm = bm * 448 / np.array([w, h])
        bm = resize_bm(bm, (448, 448))
        img_RGB = cv2.resize(img_RGB, (448, 448))

        bnd = get_bound_crop()
        img_RGB, bm = do_tight_crop(img_RGB, bm, bnd)

        img_RGB = cv2.resize(img_RGB, (288, 288))
        img_RGB = img_RGB.transpose(2, 0, 1) / 255.0

        bm = resize_bm(bm, (288, 288))
        bm = ((bm.transpose(2, 0, 1) / 448.0) - 0.5) * 2
        img_RGB_unwarped = numpy_unwarping(img_RGB, bm)

        # return as (c, h, w) and range [0, 1] / [-1, 1]
        return (
            img_RGB.astype(np.float32),
            img_RGB_unwarped.astype(np.float32),
            bm.astype(np.float32),
        )
