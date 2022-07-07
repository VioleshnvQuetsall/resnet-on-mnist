import numpy as np
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

        self.trigger = torch.load(self.trigger_path).squeeze().numpy().astype(np.uint8)
        # print(self.trigger.shape)

    def __len__(self):
        return self.trigger.shape[0]

    def __getitem__(self, idx: int):
        image = self.trigger[idx]
        label = self.trigger_labels[idx] if self.trigger_labels else idx % 10
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def random_choices(self, count: int):
        choices = np.random.randint(len(self), size=(count,))
        images = ((self.trigger[choices] / 255.0) - 0.1307) / 0.3081
        labels = choices % 10
        return torch.from_numpy(images.reshape(count, 1, 28, 28)).type(torch.float), torch.from_numpy(labels)
