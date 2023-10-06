import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import typing
from tqdm.auto import tqdm
from math import ceil
import matplotlib.pyplot as plt
from torchvision import datasets


def accuracy_fn(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    acc = torch.eq(y_pred, y_true).sum().item()
    return (acc / len(y_true)) * 100


def train_step(
        model: nn.Module,
        data_loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: Optimizer,
        accuracy_fn: typing.Callable[[torch.Tensor, torch.Tensor], float],
        device: torch.device
) -> float:
    """
    Performs one training step with model trying to learn on data_loader
    :param model: Model
    :param data_loader: Train Data Loader
    :param loss_fn: Loss function
    :param optimizer: Optimizer
    :param accuracy_fn: Accuracy measure
    :param device: Destination device
    :return: Model accuracy
    """
    train_loss, train_acc = 0, 0
    model.train()

    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)

        y_predictions = model(X)

        loss = loss_fn(y_predictions, y)
        train_loss += loss
        train_acc += accuracy_fn(y_predictions.argmax(dim=1), y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)

    print(f'Train loss = {train_loss} | Train accuracy = {train_acc:.2f}%')
    return train_acc


def test_step(
        model: nn.Module,
        data_loader: DataLoader,
        loss_fn: nn.Module,
        accuracy_fn: typing.Callable[[torch.Tensor, torch.Tensor], float],
        device: torch.device
) -> float:
    """
    Perform test step on model going over dataloader
    :param model: Model to test
    :param data_loader: Dataset
    :param loss_fn: Loss Function
    :param accuracy_fn: Accuracy function
    :param device: Device
    :return: Test accuracy
    """
    test_loss, test_acc = 0, 0
    model.eval()

    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            y_predictions = model(X)
            test_loss += loss_fn(y_predictions.argmax(dim=1), y)
            test_acc += accuracy_fn(y_predictions, y)
    test_loss /= len(data_loader)
    test_acc /= len(data_loader)

    print(f'Test loss = {test_loss} | Test accuracy = {test_acc:.2f}%')
    return test_acc


def eval_model(
        model: nn.Module,
        data_loader: DataLoader,
        accuracy_fn: typing.Callable[[torch.Tensor, torch.Tensor], float],
        loss_fn: nn.Module,
        device: torch.device
) -> dict:
    """
    Eval model
    :param model:
    :param data_loader:
    :param accuracy_fn:
    :param loss_fn:
    :param device:
    :return: Dictionary with following keys:
        model_name, model_loss, model_acc
    """
    model_loss, model_acc = 0, 0
    with torch.inference_mode():
        for X, y in tqdm(data_loader):
            X, y = X.to(device), y.to(device)
            y_predictions = model(X)
            model_loss += loss_fn(y_predictions.argmax(dim=1), y)
            model_acc += accuracy_fn(y_predictions)
        model_acc /= len(data_loader)
        model_loss /= len(data_loader)

    return {
        'model_name': model.__class__.__name__,
        'model_acc': model_acc,
        'model_loss': model_loss
        }


def plot_sample_predictions(
        model: nn.Module,
        dataset: datasets,
        labels: dict,
        no_figs: int
):
    cols = 3
    rows = ceil(no_figs / cols)
    correct_predictions = 0

    figure = plt.figure(figsize=(7, 7))
    for i in range(1, rows * cols + 1):
        img_idx = torch.randint(len(dataset), size=[1]).item()
        img, label = dataset[img_idx]

        predicted_label = model(img.unsqueeze(dim=0)).argmax(dim=1).item()
        if predicted_label == label:
            correct_predictions += 1
            text_color = 'g'
        else:
            text_color = 'r'

        figure.add_subplot(rows, cols, i)
        plt.imshow(np.transpose(img, (1, 2, 0)))
        plt.title(f'{labels[label]} ({labels[predicted_label]})', c=text_color)
        plt.axis('off')

    print(f'Model accuracy: {(correct_predictions / no_figs) * 100:.2f}%')