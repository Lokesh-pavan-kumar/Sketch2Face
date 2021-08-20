import torch
from PIL import Image
from utils import load_network
from generator import Generator
import torchvision.transforms as T
from torchvision.utils import save_image
from pathlib import Path
from tqdm.auto import tqdm

generator = Generator(3)
choice = int(input("Enter\n1 - Sketch to Photo\n2 - Photo to Sketch\n"))

if choice == 1:
    load_network("GeneratorS2F.pth.tar", generator, None, None)
    transform = T.Compose([
        T.Resize(257),
        T.CenterCrop(256),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
    ])
else:
    load_network("GeneratorF2S.pth.tar", generator, None, None)
    transform = T.Compose([
        T.Resize(257),
        T.CenterCrop(256),
        T.ToTensor()
    ])

generator.eval()
img_path = Path("./Data/test/photos")
destination_path = Path("./Data/test/fake-sketches")

if not destination_path.exists():
    destination_path.mkdir()

with torch.no_grad():
    for path in tqdm(sorted(img_path.iterdir())):
        sample = transform(Image.open(path))
        if sample.ndim <= 3:
            sample = sample.unsqueeze(0)
        generated_sample = generator(sample).squeeze(0)
        save_image(generated_sample, destination_path / path.name)
