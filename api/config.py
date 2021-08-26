import torch
import torchvision.transforms as T
from torchvision.models import resnet50
from model import Grayscale, SiameseNetwork, load_network
from typing import Union
from generator import Generator
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Global use

backbone = resnet50(False)  # will load the entire siamese later, pretrained weights can be ignored

# Generation networks - F2S, S2F
generator_s2f = Generator(3)  # Generator Sketch to Face
load_network("./generator_models/GeneratorS2F.pth.tar", generator_s2f)

generator_f2s = Generator(3)  # Generator Face to Sketch
load_network("./generator_models/GeneratorF2S.pth.tar", generator_f2s)

# Siamese networks - Traditional, S2S and F2F
siamese_normal = SiameseNetwork(backbone, 2048)  # Similarity b/w photo and sketch
load_network("./siamese_models/siamese-traditional-300.pth.tar", siamese_normal)

siamese_sketch = SiameseNetwork(backbone, 2048)  # Similarity b/w sketch and sketch
load_network("./siamese_models/siamese-S2S-300.pth.tar", siamese_sketch)

siamese_photo = SiameseNetwork(backbone, 2048)  # Similarity b/w photo and photo
load_network("./siamese_models/siamese-F2F-300.pth.tar", siamese_photo)

transform = T.Compose([
    T.Resize(257),
    T.CenterCrop(256),
    T.ToTensor(),
    Grayscale()
])

# mugshot and embedding file details
mugshot_path = Path("./Mugshot")
emb_loc = Path("./Mugshot/embeddings.pt")  # location of the embedding file

# misc
image_exts = [".png", ".jpg", ".jpeg", ".tif", ".bmp"]


def preprocess(image, mode: Union[int, str]):
    """
    Takes in an input image and converts it to a different domain using the appropriate generator network
    :param image: input image
    :param mode: an integer denoting the conversion process
    0 - Sketch2Face
    1 - Face2Sketch
    else - resize, crop and to tensor
    :return: converted, preprocessed image tensor
    """
    img_tensor = transform(image)  # PIL to torch.Tensor
    if img_tensor.ndim == 3:
        img_tensor = img_tensor.unsqueeze(0)  # Add fake batch-dim
    if mode == 0 or mode == "siamese-face":  # Sketch to Face
        generator_s2f.eval()
        processed = generator_s2f(img_tensor).squeeze()
    elif mode == 1 or mode == "siamese-sketch":  # Face to Sketch
        generator_f2s.eval()
        processed = generator_f2s(img_tensor).squeeze()
    else:
        processed = transform(image).squeeze()
    assert processed.ndim == 3
    return processed


def get_embedding(image, transform_image: bool = False, transformation: int = None):
    if transform_image:
        dst = preprocess(image, transformation)
        if dst.ndim == 3:
            dst = dst.unsqueeze(0)
        embedding = siamese_sketch.encode_samples(dst) if transformation == 0 else siamese_photo.encode_samples(dst)
        embedding = embedding.squeeze()
    else:  # No transformation required
        processed = preprocess(image, -1)
        if processed.ndim == 3:
            processed = processed.unsqueeze(0)
        embedding = siamese_normal.encode_samples(processed).squeeze()
    return embedding / torch.norm(embedding, dim=-1, keepdim=True)
