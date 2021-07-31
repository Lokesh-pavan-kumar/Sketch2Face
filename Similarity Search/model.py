import torch
import torch.nn as nn
from typing import Union


class SiameseNetwork(nn.Module):
    def __init__(self, encoder_network: nn.Module = None, emb_dim: int = 512,
                 freeze: bool = False, rate: float = 0.6) -> None:
        """
        :param encoder_network: pretrained network that is used in the siamese model
        :param emb_dim: dimensions of the image embeddings
        :param freeze: boolean stating whether or not to freeze the pretrained network
        :param rate: dropout rate used in the network
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.rate = rate
        if encoder_network is None:
            self.siamese_net = self.get_network()
        else:
            # TODO: Currently configured for Resnet50 like architectures
            # Only the final child in the network is a linear layer and the rest of the network can be used as is
            encoder_network.requires_grad_(not freeze)  # Freeze the params?
            self.encoder_layers = list(encoder_network.children())
            self.siamese_net = nn.Sequential(
                *self.encoder_layers[:-1],  # Output = [2048, 1, 1]
                nn.Dropout2d(self.rate),
                nn.Conv2d(2048, emb_dim * 2, (1, 1)),
                nn.Flatten(),
                nn.Linear(emb_dim * 2, emb_dim)
            )

    def forward(self, images: torch.Tensor):
        return self.siamese_net(images)

    def encode_samples(self, images: torch.Tensor):
        """
        :param images: image tensors to be encoded
        :return: the embeddings of the images as a tensor
        """
        if images.ndim == 3:
            images = images.unsqueeze(0)
        return self(images)

    def get_network(self):
        """
        :return: A custom network for images of size (3, 256, 256)
        """
        return nn.Sequential(
            ConvBlock(3, 32, (3, 3)),
            nn.MaxPool2d((2, 2)),
            ConvBlock(32, 64, (3, 3), stride=(2, 2)),
            ConvBlock(64, 128, (5, 5)),
            nn.MaxPool2d((2, 2)),
            ConvBlock(128, 256, (3, 3), stride=(2, 2)),
            ConvBlock(256, 512, (3, 3)),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(512, self.emb_dim, (1, 1)),  # Change the channel dimensions
            nn.MaxPool2d((2, 2)),
            nn.Flatten(),  # Output shape (N, emb_dim)
            # Fully connected layers
            nn.Linear(self.emb_dim * 3 * 3, self.emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.rate),
            nn.Linear(self.emb_dim, self.emb_dim),
        )


class Lambda(nn.Module):
    def __init__(self, func) -> None:
        """
        A lambda layer that runs a given function.
        Can be used in nn Sequential
        :param func: function to be applied onto the input
        """
        super(Lambda, self).__init__()
        self.func = func

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.func(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, k_size: Union[int, tuple], **kwargs) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, k_size, bias=False, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x) -> torch.Tensor:
        return self.block(x)

# # Testing the implementation
# if __name__ == '__main__':
#     input_ = torch.randn(5, 3, 256, 256)
#     network = SiameseNetwork(None, 512)
#     output = network(input_)
#     print(output.shape)
#     assert output.shape == torch.Size([5, 512])
