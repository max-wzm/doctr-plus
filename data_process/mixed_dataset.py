import torch

from data_process import GRID_SIZE, IMG_SIZE
from data_process.qb_dataset import QbDataset
from data_process.uvdoc_dataset import UVDocDataset


class MixedDataset(torch.utils.data.Dataset):
    """
    Torch dataset class for the QBDoc dataset.
    """

    def __init__(
        self,
        qb_data_path="./data/QBdoc",
        uv_data_path="./data/UVdoc",
        appearance_augmentation=[],
        geometric_augmentations=[],
        grid_size=GRID_SIZE,
        split="train",
    ) -> None:
        super().__init__()
        self.qb_dataset = QbDataset(
            qb_data_path,
            appearance_augmentation=appearance_augmentation,
            geometric_augmentations=[],
            grid_size=grid_size,
            split=split,
        )
        self.uv_dataset = UVDocDataset(
            uv_data_path,
            appearance_augmentation=appearance_augmentation,
            geometric_augmentations=geometric_augmentations,
            grid_size=grid_size,
            split=split,
        )

    def __len__(self):
        return len(self.qb_dataset) + len(self.uv_dataset)

    def __getitem__(self, index):
        if index < len(self.qb_dataset):
            return self.qb_dataset.__getitem__(index)
        return self.uv_dataset.__getitem__(index - len(self.qb_dataset))
