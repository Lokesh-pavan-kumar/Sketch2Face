import torch
import torchvision.transforms as T
from pathlib import Path
from PIL import Image
from typing import Union, List
import torch.nn as nn


class SiameseNetwork(nn.Module):
    def __init__(self, encoder_network: nn.Module = None, emb_dim: int = 512,
                 freeze: bool = False) -> None:
        """
        :param encoder_network: pretrained network that is used in the siamese model
        :param emb_dim: dimensions of the image embeddings
        :param freeze: boolean stating whether or not to freeze the pretrained network
        """
        super().__init__()
        self.emb_dim = emb_dim
        if encoder_network is None:
            self.siamese_net = self.get_network()
        else:
            # TODO: Currently configured for Resnet50 like architectures
            # Only the final child in the network is a linear layer and the rest of the network can be used as is
            encoder_network.requires_grad_(not freeze)  # Freeze the params?
            self.encoder_layers = list(encoder_network.children())
            self.siamese_net = nn.Sequential(
                *self.encoder_layers[:-1],  # Output = [2048, 1, 1]
                nn.Flatten(),
                nn.Linear(2048, emb_dim)
            )

    def forward(self, images: torch.Tensor):  # Before running make sure the network is in train mode
        return self.siamese_net(images)

    @torch.no_grad()  # No need to calculate any gradients
    def encode_samples(self, images: torch.Tensor):
        """
        :param images: image tensors to be encoded
        :return: the embeddings of the images as a tensor
        """
        self.eval()  # Change to eval mode
        if images.ndim == 3:
            images = images.unsqueeze(0)
        assert self.training is False
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
            nn.Dropout(0.6),
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


def save_embeddings(network: torch.nn.Module, in_path: str,
                    out_file: str, rename: bool = False, start_from: int = None):
    in_path = Path(in_path)
    if not in_path.exists():
        print("Provided path does not exist")
        return -1
    if rename:  # Rename the files
        print("Renaming files")
        rename_files(in_path, "mugshot", start_from if start_from is not None else 1)
    transform = T.Compose([
        T.Resize(257),
        T.CenterCrop(256),
        T.ToTensor(),
    ])
    images = [Image.open(f_name) for f_name in sorted(in_path.iterdir(), key=lambda x: int(x.stem.split("_")[1]))]
    write_embeddings(network, images, transform, out_file)


def write_embeddings(network: torch.nn.Module, images: List, transform: T, out_file: str, device: str = "cpu"):
    tensor_images = torch.stack([transform(image) for image in images])
    network = network.to(device)
    network.eval()
    embeddings = network.encode_samples(tensor_images).squeeze()
    assert embeddings.ndim <= 2
    torch.save(embeddings, out_file + ".pt")
    return


def rename_files(path: Union[str, Path], prefix: str = None, start_from: int = 1) -> int:
    path = Path(path) if isinstance(path, str) else path
    if not path.exists():  # If path does not exist
        print("Provided path does not exist")
        return -1  # Not successful
    curr_id = start_from  # Current name of the file
    for file_name in path.iterdir():
        if file_name.is_file():
            directory = file_name.parent  # Get the parent dir
            extension = file_name.suffix  # Get the file ext
            new_name = str(curr_id) + extension
            if prefix is not None:
                new_name = prefix + "_" + new_name
            file_name.rename(Path(directory, new_name))  # Directory + New_Name, rename and save
            curr_id += 1
    return curr_id  # If successful returns the id of the last renamed file obj


def load_network(filename: str, network: torch.nn.Module, optimizer: torch.optim.Optimizer = None, lr: float = None,
                 **kwargs):
    checkpoint = torch.load(filename, map_location="cpu")
    network.load_state_dict(checkpoint['network'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    meta_data = {}
    for param in kwargs:
        if checkpoint.get(param, None) is not None:
            meta_data[param] = checkpoint[param]
    print(f'Loaded model from {filename}')

    return meta_data
