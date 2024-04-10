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
        qb_data_path=["./data/QBdoc2", "./data/QBdoc3"],
        real_suffix=["jpg", "jpg"],
        uv_data_path=["./data/UVdoc", "./data/doc3"],
        syn_suffix=["png", "png"],
        appearance_augmentation=[],
        geometric_augmentations=[],
        grid_size=GRID_SIZE,
        split="train",
    ) -> None:
        super().__init__()
        self.qb_dataset = QbDataset(
            qb_data_path,
            real_suffix,
            appearance_augmentation=appearance_augmentation,
            geometric_augmentations=[],
            grid_size=grid_size,
            split=split,
            total_num=[20000, 30000],
        )
        self.uv_dataset = UVDocDataset(
            uv_data_path,
            syn_suffix,
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


class MixedSeperateDataset(MixedDataset):
    """
    Torch dataset class for the QBDoc dataset.
    """

    def __init__(
        self,
        appearance_augmentation=[],
        geometric_augmentations=[],
        grid_size=GRID_SIZE,
        split="train",
    ) -> None:
        super().__init__()
        self.qb_dataset = QbDataset(
            appearance_augmentation=appearance_augmentation,
            geometric_augmentations=[],
            grid_size=grid_size,
            split=split,
            total_num=[30000, 20000],
        )
        self.uv_dataset = MixedDataset(
            appearance_augmentation=appearance_augmentation,
            geometric_augmentations=geometric_augmentations,
            grid_size=grid_size,
            split=split,
        )
        self.maxlen = max(len(self.qb_dataset), len(self.uv_dataset))
        self.minlen = min(len(self.qb_dataset), len(self.uv_dataset))

    def __len__(self):
        return self.maxlen

    def __getitem__(self, index):
        index_shortest = index % self.minlen
        uv_index = index_shortest if len(self.uv_dataset) == self.minlen else index
        qb_index = index + index_shortest - uv_index
        return self.uv_dataset[uv_index], self.qb_dataset[qb_index]
