import json
import os
import random
from os.path import join as pjoin

import cv2
import hdf5storage as h5
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

RANGE_H = (0, 288)
RANGE_W = (0, 288)


class ImageInfo:
    def __init__(self, dataroot, sample_id, suffix) -> None:
        self.dataroot = dataroot
        self.sample_id = sample_id
        self.suffix = suffix
        pass

    @staticmethod
    def read_from_dataroots(dataroots, suffixes, total_num):
        res = []
        for i, dataroot in enumerate(dataroots):
            suffix = suffixes[i]
            num = total_num[i]
            all_samples = [
                ImageInfo(dataroot, x[:-4], suffix)
                for x in os.listdir(pjoin(dataroot, "img"))
                if x.endswith(suffix)
            ]
            print("len of ", dataroot, len(all_samples), os.listdir(pjoin(dataroot, "img"))
            random.shuffle(all_samples)
            all_samples = all_samples[:num]
            res.extend(all_samples)
        print("read from ",dataroots, len(res))
        return res

    @property
    def img_path(self):
        return pjoin(self.dataroot, "img", f"{self.sample_id}.{self.suffix}")

    @property
    def bm_path(self):
        return pjoin(self.dataroot, "bm", f"{self.sample_id}.mat")

    @property
    def grid2D_path(self):
        if not os.path.exists(pjoin(self.dataroot, "metadata_sample")):
            return self.bm_path
        with open(
            pjoin(self.dataroot, "metadata_sample", f"{self.sample_id}.json"), "r"
        ) as f:
            sample_name = json.load(f)["geom_name"]
        grid2D_path = pjoin(self.dataroot, "grid2d", f"{sample_name}.mat")
        return grid2D_path

    @property
    def grid2D(self):
        if not os.path.exists(pjoin(self.dataroot, "metadata_sample")):
            bm = h5.loadmat(self.bm_path)["grid2D"]
            bm = resize_bm(bm, (89, 61)).transpose(2, 0, 1)
            return bm
        with open(
            pjoin(self.dataroot, "metadata_sample", f"{self.sample_id}.json"), "r"
        ) as f:
            sample_name = json.load(f)["geom_name"]
        grid2D_path = pjoin(self.dataroot, "grid2d", f"{sample_name}.mat")
        return h5.loadmat(grid2D_path)["grid2d"].transpose(2, 0, 1)


def new_h_map(old_h, warped_box, cropped_box):
    w_top, w_btm, w_left, w_right = warped_box
    cr_top, cr_btm, cr_left, cr_right = cropped_box
    # return mapping(mapping(old_h - w_top, 0, w_btm-w_top, 0, cr_btm-cr_top), 0, cr_btm-cr_top, 0, 288)
    return mapping(old_h - w_top, 0, w_btm - w_top, 0, 448)


def new_w_map(old_h, warped_box, cropped_box):
    w_top, w_btm, w_left, w_right = warped_box
    cr_top, cr_btm, cr_left, cr_right = cropped_box
    # return mapping(mapping(old_h - w_left, 0, w_right-w_left, 0, cr_right-cr_left), 0, cr_right-cr_left, 0, 288)
    return mapping(old_h - w_left, 0, w_right - w_left, 0, 448)


def load_bm(path):
    bm_path2 = path
    bm2 = h5.loadmat(bm_path2)["bm"]
    return bm2


def new_crop_bm(bm, cropped_box, warped_box):
    cr_top, cr_btm, cr_left, cr_right = cropped_box
    new_bm = bm.copy()[cr_top:cr_btm, cr_left:cr_right, :]
    new_bm[:, :, 0] = new_w_map(new_bm[:, :, 0], warped_box, cropped_box)
    new_bm[:, :, 1] = new_h_map(new_bm[:, :, 1], warped_box, cropped_box)
    return new_bm


def crop_image_tight(img, grid2D, cropped_box):
    """
    Crops the image tightly around the keypoints in grid2D.
    This function creates a tight crop around the document in the image.

    Returns warped_box
    """
    grid2D = grid2D.copy()
    size = img.shape

    cr_top, cr_btm, cr_left, cr_right = cropped_box
    grid2D = grid2D[cr_top:cr_btm, cr_left:cr_right, :]
    minx = np.floor(np.amin(grid2D[:, :, 0])).astype(int)
    maxx = np.ceil(np.amax(grid2D[:, :, 0])).astype(int)
    miny = np.floor(np.amin(grid2D[:, :, 1])).astype(int)
    maxy = np.ceil(np.amax(grid2D[:, :, 1])).astype(int)

    s = 5
    s = min(
        min(s, minx), miny
    )  # s shouldn't be smaller than actually available natural padding is
    s = min(min(s, size[1] - 1 - maxx), size[0] - 1 - maxy)

    cx1 = random.randint(0, max(s - 3, 1))
    cx2 = random.randint(0, max(s - 3, 1)) + 1
    cy1 = random.randint(0, max(s - 3, 1))
    cy2 = random.randint(0, max(s - 3, 1)) + 1

    top = max(0, miny + random.randint(-10, 10))
    bot = min(448, maxy + random.randint(-10, 10))
    left = max(0, minx + random.randint(-10, 10))
    right = min(448, maxx + random.randint(-10, 10))
    return (top, bot, left, right)


def mapping(value, a, b, c, d):
    # mapping from [a, b] to [c, d]
    mapped_value = (value - a) * (d - c) / (b - a) + c
    return mapped_value


def get_unwarp(alb, bm):
    """
    get unwarped image

    input image must be resized to (288, 288)
    return float image

    require input: h, w, 3
    output: h, w, 3
    """
    alb = alb.copy()
    bm = bm.copy()

    assert alb.shape[2] == 3
    assert bm.shape[2] == 2

    h, w, c = alb.shape
    # scale bm to -1.0 to 1.0
    bm_ = bm / np.array([w, h])
    bm_ = (bm_ - 0.5) * 2
    bm_ = resize_bm(bm_, (h, w))
    bm_ = np.reshape(bm_, (1, h, w, 2))
    bm_ = torch.from_numpy(bm_).float()
    img_ = alb.transpose((2, 0, 1)).astype(np.float32) / 255.0
    img_ = np.expand_dims(img_, 0)
    img_ = torch.from_numpy(img_)
    uw = F.grid_sample(img_, bm_)
    uw = uw[0].numpy().transpose((1, 2, 0))
    return uw


def tensor_unwarping(warped_imgs, bms, size=(288, 288)):
    """
    Utility function that unwarps an image.
    Unwarp warped_img based on the 2D grid point_positions with a size img_size.
    Args:
        warped_img  :       torch.Tensor of shape BxCxHxW (dtype float)
        point_positions:    torch.Tensor of shape Bx2xGhxGw (dtype float)
        img_size:           tuple of int [w, h]
    """
    upsampled_grid = F.interpolate(bms, size=size, mode="bilinear", align_corners=True)
    unwarped_img = F.grid_sample(
        warped_imgs, upsampled_grid.transpose(1, 2).transpose(2, 3), align_corners=True
    )

    return unwarped_img


def numpy_unwarping(warped_img, bm, size=(288, 288)):
    warped_tensor = torch.from_numpy(warped_img).float().unsqueeze(0)
    bm_tensor = torch.from_numpy(bm).float().unsqueeze(0)
    unwarped_tensor = tensor_unwarping(warped_tensor, bm_tensor, size)
    return unwarped_tensor[0].numpy()


def resize_bm(bm2, target_size):
    """
    expand bm to a target size
    bm shape = (H, W, D)
    bm[h, w] = [w, h]
    """
    h, w, _ = bm2.shape
    if (h, w) == target_size:
        return bm2
    bm2 = bm2.transpose(2, 0, 1)
    grid2D = torch.from_numpy(bm2).float()
    upsampled_grid = F.interpolate(
        grid2D.unsqueeze(0), size=target_size, mode="bilinear", align_corners=True
    )
    upsampled_grid = upsampled_grid.squeeze(0).numpy()
    upsampled_grid = upsampled_grid.transpose(1, 2, 0)
    return upsampled_grid


def do_tight_crop(img, bm, cropped_box):
    warped_box = crop_image_tight(img, bm, cropped_box)
    w_top, w_btm, w_left, w_right = warped_box
    img = img[w_top:w_btm, w_left:w_right]
    img = cv2.resize(img, (448, 448))
    bm = new_crop_bm(bm, cropped_box, warped_box)
    # bm = cv2.resize(img, (288, 288))
    return img, bm


if __name__ == "__main__":
    alb = cv2.imread("doc3d/alb/1/998_2-vc_Page_007-gaI0001.png")
    alb_cp = alb.copy()
    bm = load_bm("doc3d/bm/1/998_2-vc_Page_007-gaI0001.mat")
    print(bm.shape)
    bm_cp = bm.copy()
    cropped_box = (224, 288, 224, 288)  # top, btm, left, right
    warped_box = crop_image_tight(alb, bm, cropped_box)
    print(warped_box)
    cr_top, cr_btm, cr_left, cr_right = cropped_box
    w_top, w_btm, w_left, w_right = warped_box

    alb = alb[w_top:w_btm, w_left:w_right]
    alb = cv2.resize(alb, (288, 288))
    print(alb.shape)
    bm = new_crop_bm(bm, cropped_box, warped_box)

    points = [(223, 0), (223, 447), (447, 0), (447, 447)]
    for h, w in points:
        print(h, w, bm_cp[h, w, :])
    _, ax = plt.subplots(2, 2)
    ax[0][0].imshow(alb)
    ax[0][1].imshow(get_unwarp(alb, bm))
    ax[1][0].imshow(alb_cp)
    ax[1][1].imshow(get_unwarp(alb_cp, bm_cp)[224:288, 0:288])
    print(bm_cp[212, 333, :])
    plt.show()
