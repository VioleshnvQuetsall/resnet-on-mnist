from typing import Optional

import torch
from torch import nn
from torchvision import models

import os

import matplotlib.pyplot as plt

import logging
import logging.handlers

from logger import get_logger
from utils import train, test, train_with_trigger, display, dataloaders


def train_model(model: nn.Module,
                save_path: Optional[str],
                epochs: int,
                lr: float,
                with_trigger: bool,
                count: Optional[int] = 4):
    if epochs == 0:
        return model
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    test_loss, test_correct, trigger_loss, trigger_correct = [[] for _ in range(4)]

    for t in range(1, epochs + 1):
        print(f"-------------------------------\nEpoch {t}")

        if with_trigger:
            train_with_trigger(train_dataloader, trigger_dataloader.dataset,
                               count, model, loss_fn, optimizer, device)
        else:
            train(train_dataloader, model, loss_fn, optimizer, device)

        loss, correct = test(test_dataloader, model, loss_fn, device)
        test_loss.append(loss)
        test_correct.append(correct)
        logger.debug(f"Test Error: Accuracy: {(100 * correct):>0.1f}%, Avg loss: {loss:>8f}")

        loss, correct = test(trigger_dataloader, model, loss_fn, device)
        trigger_loss.append(loss)
        trigger_correct.append(correct)
        logger.debug(f"Trigger Error: Accuracy: {(100 * correct):>0.1f}%, Avg loss: {loss:>8f}")

        if save_path and t % 5 == 0:
            torch.save(model.state_dict(), save_path)
            logger.debug(f"Saved PyTorch Model State to {save_path}")

    logger.info(f'  test_loss: {test_loss}')
    logger.info(f'  test_correct: {test_correct}')
    logger.info(f'  trigger_loss: {trigger_loss}')
    logger.info(f'  trigger_correct: {trigger_correct}')
    logger.debug(f"{'Trigger' if with_trigger else 'Prepare'} Done!")

    return model


def display_weights(
        model: nn.Module,
        save_path: Optional[str] = None,
        show: bool = True):
    if not (save_path or show):
        return

    fig, axs = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [2, 2]})

    fig.tight_layout()

    parameters = {
        'model': model,
        'train_dataloader': train_dataloader,
        'device': device,
        'colors': None,
        'connect_line': False,
        'data': True
    }

    display(
        ax=axs[0],
        weights=True,
        boundary=False,
        **parameters
    )

    display(
        ax=axs[1],
        weights=False,
        boundary=True,
        **parameters
    )

    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()


def trigger_helper(model: nn.Module, task: str, name: str, dir_path: str):
    model_name = f'{name}_{task}'
    logger.info(f'####################### {model_name} ############################')
    logger.info(f'{model_name}:')

    parameters = {
        'train': (
            [
                f'{dir_path}/{name}.pth',
                f'{dir_path}/{name}.png',
                None,
                None
            ], [40, 0]
        ),
        'trigger': (
            [
                f'{dir_path}/{name}.pth',
                f'{dir_path}/{name}.png',
                f'{dir_path}/{model_name}.pth',
                f'{dir_path}/{model_name}.png'
<<<<<<< HEAD
            ], [160, 160]
=======
            ], [80, 80]
>>>>>>> c6134415177272f135573267e792e9d81268b7a0
        ),
        'scratch_trigger': (
            [
                None,
                None,
                f'{dir_path}/{model_name}.pth',
                f'{dir_path}/{model_name}.png'
<<<<<<< HEAD
            ], [0, 320]
=======
            ], [0, 160]
>>>>>>> c6134415177272f135573267e792e9d81268b7a0
        ),
        'finetune': (
            [
                f'{dir_path}/{model_name}.pth',
                f'{dir_path}/{model_name}.png',
                None,
                None
            ], [40, 0]
        )
    }

    save_path, epochs = parameters[task]

    train_model(
        model=model,
        save_path=save_path[0],
        epochs=epochs[0],
        lr=1e-3 if task != 'finetune' else 1e-4,
        with_trigger=False
    )
    display_weights(
        model=model,
        save_path=save_path[1],
        show=False
    )
    train_model(
        model=model,
        save_path=save_path[2],
        epochs=epochs[1],
        lr=1e-4,
        with_trigger=True,
        count=16
    )
    display_weights(
        model=model,
        save_path=save_path[3],
        show=False
    )
    return model


def main():
    for name, task in zip(
            [f'model{i}' for i in range(26, 30)],
            ['trigger'] * 2 + ['scratch_trigger'] * 2
    ):
        dir_path = f'files/{name}'
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)

        handler = logging.FileHandler(f'{dir_path}/log', 'w')
        handler.setLevel(logging.INFO)
        logger.handlers.append(handler)

        model = models.resnet18(weights=None, num_classes=10)
        model.to(device)
        trigger_helper(model, task, name, dir_path)
        trigger_helper(model, 'finetune', name, dir_path)

        logger.handlers.pop()


if __name__ == '__main__':
    logger = get_logger()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataloader, test_dataloader, trigger_dataloader = dataloaders(
        mnist_dir='mnist',
        trigger_path='files/trigger.pth'
    )
    main()
