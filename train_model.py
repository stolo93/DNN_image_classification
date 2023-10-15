"""
Load model from checkpoint (if exists), train it and create another checkpoint
"""

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import CIFAR10
from pathlib import Path
from datetime import datetime
import os
from tqdm.auto import tqdm
import time

from Utils import train_step, test_step, accuracy_fn
from TinyVGG import TinyVGGModel


def load_checkpoint(path: Path = Path('checkpoints')):
    """
    Load latest checkpoint from path or none
    :param path: Path to checkpoint dir
    :return: Latest save or none
    """
    if not os.path.isdir(path):
        return None
    archives = os.listdir(path)
    if archives:
        latest = sorted(archives)[-1]
        cp_path = Path(path / latest)
        print(f'Checkpoint found, loading {cp_path}')
        return torch.load(cp_path)


def save_checkpoint(
        path: Path,
        epochs: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_stats: list,
        test_stats: list
):
    """
    Create and save checkpoint of the current training state
    :param path: checkpoint directory
    :param epochs: total num of epochs
    :param model: model in training
    :param optimizer: optimizer used in training
    :param train_stats: Train statistics in a list
    :param test_stats: Test statistics in a list
    :return: None
    """
    time_str = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    try:
        orig_mask = os.umask(0)
        os.makedirs(path, 0o777, exist_ok=True)
    finally:
        os.umask(orig_mask)
    tiny_vgg_cp = Path('tiny_vgg_' + time_str + '.tar')
    cp_path = path / tiny_vgg_cp

    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_stats': train_stats,
        'test_stats': test_stats
    }, cp_path)
    print(f'Checkpoint saved to: {cp_path}')


def get_dataloaders(path: Path = Path('data')):
    """
    Load data to path and create dataloaders and labels
    :param path: Path to dataset
    :return: train, test, labels
    """
    data_path = Path('data')
    try:
        orig_mask = os.umask(0)
        os.makedirs(data_path, 0o777, True)
    finally:
        os.umask(orig_mask)
    train_data = CIFAR10(root=str(data_path), train=True, download=True, transform=ToTensor())
    test_data = CIFAR10(root=str(data_path), train=False, download=True, transform=ToTensor())

    train_dataloader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(dataset=test_data, batch_size=64, shuffle=False)

    labels = {idx: cls for cls, idx in train_data.class_to_idx.items()}
    return train_dataloader, test_dataloader, labels


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_dataloader, test_dataloader, labels = get_dataloaders()

    # Create model, optimizer and loss function
    torch.manual_seed(42)
    model_tvgg = TinyVGGModel(input_shape=3, hidden_units=10, output_shape=len(labels))
    optimizer = torch.optim.SGD(model_tvgg.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    total_epochs = 0
    train_stats = []
    test_stats = []

    # Try to load checkpoint
    checkpoint = load_checkpoint()
    if checkpoint is not None:
        model_tvgg.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        train_stats = checkpoint['train_stats']
        test_stats = checkpoint['test_stats']
        total_epochs = checkpoint['epoch']

    # Move everything to the desired device
    model_tvgg = model_tvgg.to(device)
    loss_fn = loss_fn.to(device)

    total_time = 0
    # Training
    while True:
        print(f'\nCurrent accuracy: Train {train_stats[-1][1]:.3f} || Test {test_stats[-1][1]:.3f}')
        epochs = int(input(f'Insert number of epochs\n(Total epochs {total_epochs}):\n'))

        time_start = time.time()
        for epoch in tqdm(range(1, epochs + 1)):
            print(f'Epoch: {epoch} / {epochs} || Total: {total_epochs}')
            train_loss_acc = train_step(
                model=model_tvgg,
                data_loader=train_dataloader,
                accuracy_fn=accuracy_fn,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device
            )
            test_loss_acc = test_step(
                model=model_tvgg,
                data_loader=test_dataloader,
                loss_fn=loss_fn,
                accuracy_fn=accuracy_fn,
                device=device
            )
            train_stats.append(train_loss_acc)
            test_stats.append(test_loss_acc)
            total_epochs += 1

        time_end = time.time()
        total_time += (time_end - time_start)

        if input('Continue [y/n]') != 'y':
            break

    print(f'Total training time: {total_time:.3f}')

    # Save checkpoint
    save_checkpoint(
        path=Path('checkpoints'),
        epochs=total_epochs,
        model=model_tvgg,
        optimizer=optimizer,
        train_stats=train_stats,
        test_stats=test_stats
    )


if __name__ == '__main__':
    main()
