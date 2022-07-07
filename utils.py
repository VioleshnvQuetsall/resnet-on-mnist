from typing import Optional

import torch
from torch.utils.data import DataLoader
from torch import nn

import matplotlib.pyplot as plt

import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize

from logger import get_logger
from TriggerDataset import TriggerDataset


def train(train_dataloader: DataLoader,
          model: nn.Module,
          loss_fn: nn.CrossEntropyLoss,
          optimizer: torch.optim.Optimizer,
          with_trigger: bool,
          trigger_dataset: TriggerDataset,
          count: int,
          device: str):
    logger = get_logger()
    model.train()
    size = len(train_dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(train_dataloader):
        if with_trigger:
            trigger_X, trigger_y = trigger_dataset.random_choices(count)
            X, y = torch.cat((X, trigger_X), 0), torch.cat((y, trigger_y), 0)
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            logger.debug(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader: DataLoader,
         model: nn.Module,
         loss_fn: nn.CrossEntropyLoss,
         device: str) -> tuple[float, float]:
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    return test_loss, correct


def display(model: nn.Module,
            ax: plt.Axes,
            train_dataloader: DataLoader,
            device: str,
            colors: bool = None,
            weights: bool = False,
            boundary: bool = True,
            connect_line: bool = False,
            data: bool = True):
    colors = colors or ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    model.eval()
    with torch.no_grad():
        length = 1.0

        if data:
            length = -float('inf')
            for batch, (X, y) in enumerate(train_dataloader):
                t = model.forward_tensor(X.to(device))
                c = model.fc(t).argmax(1).cpu().numpy()
                t = t.cpu().numpy()
                for i in range(10):
                    points = t[c == i]
                    ax.scatter(points[:, 0], points[:, 1], s=10, alpha=0.3, marker='o', color=colors[i])
                length = max(length, np.max(np.sqrt(np.sum(t ** 2, axis=1))))
                if batch == 300:
                    break

        w = model.fc.weight.cpu().numpy()

        if weights:
            max_w = np.max(np.sqrt(np.sum(w ** 2, axis=1)))
            for i in range(10):
                x, y = [[0, w[i][k] / max_w * length] for k in (0, 1)]
                ax.plot(x, y, linestyle='-', linewidth=2, color=colors[i], label=f'{i}')

        if boundary:
            # get weights ordered by angles
            tensor_order = np.arctan2(w[:, 1], w[:, 0]).argsort()
            zero_index = np.where(tensor_order == 0)[0][0]
            tensor_order = np.concatenate((
                tensor_order[:zero_index],
                tensor_order[zero_index:]
            ))
            tensor_in_order = w[tensor_order]
            diff = np.diff(tensor_in_order, axis=0, prepend=tensor_in_order[-1:, :])
            # rotate -90 degrees
            diff = diff @ np.array([[0, 1], [-1, 0]]).T
            # scaling to same length
            diff = diff / np.sqrt(np.sum(diff ** 2, axis=1)).reshape(-1, 1) * length
            for i in range(10):
                x, y = [[0, diff[i][k]] for k in (0, 1)]
                ax.plot(x, y, linestyle='--', linewidth=2, color=colors[tensor_order[i]], alpha=0.7, label=f'{i}')

        if connect_line:
            tensor_in_order = w[tensor_order] * 50.0
            for i in range(10):
                x, y = [[tensor_in_order[j, k] for j in ((i + 9) % 10, i)] for k in (0, 1)]
                ax.plot(x, y, linestyle='-', linewidth=2, color=colors[tensor_order[i]], alpha=0.3)

        # unused due to failure to confirm colors
        # x_min, x_max = -45, 45
        # y_min, y_max = -45, 45

        # a = np.linspace(x_min, x_max, 500)
        # b = np.linspace(y_min, y_max, 500)

        # A, B = np.meshgrid(a, b)
        # AB = np.hstack((A.reshape(-1, 1), B.reshape(-1, 1)))
        # AB = torch.from_numpy(AB).to(torch.float32)

        # with torch.no_grad():
        #     # predict AB
        #     ax.contourf(A, B, model.fc(AB).argmax(1).numpy().reshape(A.shape), alpha=0.2)
        #     # ax.scatter(X[:,0], X[:,1], c=y, alpha=0.5)

        # ax.xlim(x_min, x_max)
        # ax.ylim(y_min, y_max)

        ax.legend()


def dataloaders(mnist_dir: str, trigger_path: str):
    transform_train = Compose([
        ToTensor(),
        Normalize(mean=[0.1307], std=[0.3081])
    ])

    training_data = datasets.MNIST(
        root=mnist_dir,
        train=True,
        download=True,
        transform=transform_train
    )
    test_data = datasets.MNIST(
        root=mnist_dir,
        train=False,
        download=True,
        transform=transform_train
    )
    trigger_data = TriggerDataset(
        trigger_path=trigger_path,
        trigger_labels=None,
        transform=transform_train,
        target_transform=None
    )

    train_dataloader = DataLoader(training_data, batch_size=32, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_data, batch_size=32, shuffle=True, num_workers=4)
    trigger_dataloader = DataLoader(trigger_data, batch_size=4, shuffle=True)

    return train_dataloader, test_dataloader, trigger_dataloader
