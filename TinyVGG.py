import torch
from torch import nn


class TinyVGGModel(nn.Module):
    """
    Smaller version of VGGnet architecture for image classification.
    Inspired by: CNN101 @ https://dl.acm.org/doi/pdf/10.1145/3334480.3382899
    """

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_1_1_block = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=output_shape,
                kernel_size=3,
            ),
            nn.ReLU()
        )
        self.conv_1_2_block = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_2_1_block = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3
            ),
            nn.ReLU()
        )
        self.conv_2_2_block = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=hidden_units * 5 * 5,
                out_features=output_shape
            )
        )

    def forward(self, x: torch.Tensor):
        x = self.conv_1_1_block(x)
        x = self.conv_1_2_block(x)
        x = self.conv_2_1_block(x)
        x = self.conv_2_2_block(x)
        return self.classifier(x)
