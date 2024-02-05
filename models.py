from torch import nn
import torch


# -------------------------------------------------------------------
class SimpleCNNModule(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        self.cnn_block1 = self.cnn_block(64)
        self.cnn_block2 = self.cnn_block(64)
        self.cnn_block3 = self.cnn_block(64)
        self.flat = nn.Flatten()
        self.dense_block1 = self.dense_block(100)
        self.dense_block2 = self.dense_block(100)
        self.last = nn.Linear(100, n_classes)

    def cnn_block(self, out_channels):
        return nn.Sequential(
            nn.LazyConv1d(out_channels, 3, padding="same"),
            nn.BatchNorm1d(out_channels, track_running_stats=False),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

    def dense_block(self, dim_out):
        return nn.Sequential(
            nn.LazyLinear(dim_out),
            nn.BatchNorm1d(dim_out, track_running_stats=False),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.cnn_block1(x)
        x = self.cnn_block2(x)
        x = self.cnn_block3(x)

        x = self.flat(x)
        x = self.dense_block1(x)
        emb = self.dense_block2(x)
        x = self.last(emb)

        return x


# -------------------------------------------------------------------
class ResNetBaseline(nn.Module):
    """A PyTorch implementation of the ResNet Baseline
    From https://arxiv.org/abs/1909.04939

    Attributes
    ----------
    sequence_length:
        The size of the input sequence
    mid_channels:
        The 3 residual blocks will have as output channels:
        [mid_channels, mid_channels * 2, mid_channels * 2]
    num_pred_classes:
        The number of output classes
    """

    def __init__(self, mid_channels: int = 64,
                 n_classes: int = 1) -> None:
        super().__init__()

        # in_channel=1 doesn't do anything here but it just refer to the
        self.layers = nn.Sequential(*[
            ResNetBlock(in_channels=1, out_channels=mid_channels),
            ResNetBlock(in_channels=mid_channels, out_channels=mid_channels * 2),
            ResNetBlock(in_channels=mid_channels * 2, out_channels=mid_channels * 2),

        ])
        self.final = nn.Linear(mid_channels * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = self.layers(x)
        return self.final(x.mean(dim=-1))


class ResNetBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        channels = [out_channels, out_channels, out_channels]
        kernel_sizes = [8, 5, 3]

        self.layers = nn.Sequential(*[
            ConvBlock(out_channels=channels[i], kernel_size=kernel_sizes[i]) for i in range(len(kernel_sizes))
        ])

        self.match_channels = False
        if in_channels != out_channels:
            self.match_channels = True
            self.residual = nn.Sequential(*[
                ConvBlock(out_channels=out_channels, kernel_size=1),
                nn.BatchNorm1d(num_features=out_channels)
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore

        if self.match_channels:
            return self.layers(x) + self.residual(x)
        return self.layers(x)


# -------------------------------------------------------------------
class ConvBlock(nn.Module):

    def __init__(self, out_channels: int, kernel_size: int) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.LazyConv1d(out_channels, kernel_size, padding="same"),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore

        return self.layers(x)

