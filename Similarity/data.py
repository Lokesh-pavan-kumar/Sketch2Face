import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from typing import Union


class SiameseDataset(Dataset):
    def __init__(self, anchors: Union[Path, str], positives: Union[Path, str], transform=None):
        anc_path = Path(anchors) if isinstance(anchors, str) else anchors
        pos_path = Path(positives) if isinstance(positives, str) else positives

        if not anc_path.exists() or not pos_path.exists():  # Need paired data, same number of samples
            print("Paths do not exist")
            exit(-1)
        self.anchors = sorted(image for image in anc_path.iterdir())
        self.positives = sorted([image for image in pos_path.iterdir()])
        assert len(self.anchors) == len(self.positives)
        self.transform = transform

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, idx: int):
        anchor, positive = self.anchors[idx], self.positives[idx]
        anchor, positive = Image.open(anchor), Image.open(positive)
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
        return anchor, positive


class Grayscale:
    def __init__(self):
        pass

    def __call__(self, sample: torch.Tensor):
        n_channels, *_ = sample.shape  # Get the number of channels
        if n_channels == 1:
            sample = sample.squeeze()
            sample = torch.stack([sample, sample, sample], 0)  # 3 Channeled grayscale image
        return sample

# # Testing the implementation
# if __name__ == '__main__':
#     input_ = torch.randn(1, 100, 100)
#     transform = Grayscale()
#     output = transform(input_)
#     print("Shape should be the same as input but with three channels")
#     print(f"Shape: {output.shape}")
