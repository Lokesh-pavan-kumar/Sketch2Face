import torch
import torchvision.transforms as T
from pathlib import Path
from PIL import Image
from typing import Union, List


def save_embeddings(network: torch.nn.Module, transform: T, in_path: str,
                    out_file: str, rename: bool = False, start_from: int = None):
    in_path = Path(in_path)
    if not in_path.exists():
        print("Provided path does not exist")
        return -1
    if rename:  # Rename the files
        print("Renaming files")
        rename_files(in_path, "mugshot", start_from if start_from is not None else 0)
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


def rename_files(path: Union[str, Path], prefix: str = None, start_from: int = 0) -> int:
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
