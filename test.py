import cv2
import numpy as np
from torch.utils.data import DataLoader

from data_process.mixed_dataset import MixedDataset
from data_process.uvdoc_dataset import UVDocDataset


def setup_data():
    t_UVDoc_data = UVDocDataset(
        appearance_augmentation=["visual", "noise", "color"],
        geometric_augmentations=["rotate", "flip", "perspective"],
    )
    train_loader = DataLoader(
        dataset=t_UVDoc_data,
        batch_size=2,
        num_workers=1,
        shuffle=True,
        pin_memory=True,
    )
    return train_loader


ds = MixedDataset()
print(ds.qb_dataset.__len__())
img, uw, bm = ds.__getitem__(27)
print(img.shape, uw.shape)
img = img.transpose(1, 2, 0)
uw = uw.transpose(1, 2, 0)
img = (img * 255.0).astype(np.uint8)
uw = (uw * 255.0).astype(np.uint8)
print(img.shape, uw.shape, bm.shape)
print(bm.max(), bm.min())
cv2.imwrite("i.png", img)
cv2.imwrite("iuw.png", uw.astype(np.uint8))
