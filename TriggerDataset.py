import torch
from torch.utils.data import Dataset

from typing import Callable, Mapping, Union


class TriggerDataset(Dataset):
    def __init__(self,
                 trigger_path: str,
                 trigger_labels: Mapping = None,
                 transform: Callable = None,
                 target_transform: Callable = None):
        self.trigger_path = trigger_path
        self.trigger_labels = trigger_labels
        self.transform = transform
        self.target_transform = target_transform

        self.trigger = torch.load(self.trigger_path)

    def __len__(self):
        return self.trigger.shape[0]

    def __getitem__(self, idx: Union[int, torch.Tensor]):
        if isinstance(idx, int):
            image = self.trigger[idx] / 255.0
            label = self.trigger_labels[idx] if self.trigger_labels else idx % 10
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)
            return image, label
        if isinstance(idx, torch.Tensor):
            image = self.trigger[idx] / 255.0
            label = self.trigger_labels[idx] if self.trigger_labels else idx % 10
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)
            return image, label
        raise RuntimeError(f'unrecognized type {type(idx)}')
