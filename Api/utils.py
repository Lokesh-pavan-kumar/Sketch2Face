import torch
from pathlib import Path
from PIL import Image


def save_embeddings(network: torch.nn.Module, transform, in_path: str, out_path: str, device: str = "cpu") -> None:
    network = network.to(device)
    in_path = Path(in_path)
    if not in_path.exists():
        print("Input path does not exist")
        return
    # Load, transform and puts on the device
    samples = transform([Image.open(img_path) for img_path in sorted(in_path.iterdir())]).to(device)
    embeddings = network.encode_samples(samples).squeeze()  # Calculate embeddings
    torch.save(embeddings, out_path + ".pt")  # Save the tensors
    with open(out_path + ".txt", "w") as lookup_file:
        for img_path in sorted(in_path.iterdir()):
            lookup_file.write(img_path.name + "\n")
    return
