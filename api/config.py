import torch
import torchvision.transforms as T
from torchvision.models import resnet50
from model import Grayscale, SiameseNetwork, load_network
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
load_network("./siamese-traditional-300.pth.tar", siamese_normal)

siamese_sketch = SiameseNetwork(backbone, 2048)  # Similarity b/w sketch and sketch
load_network("./siamese-S2S-300.pth.tar", siamese_sketch)

siamese_photo = SiameseNetwork(backbone, 2048)  # Similarity b/w photo and photo
load_network("./siamese-F2F-300.pth.tar", siamese_photo)

transform = T.Compose([
    T.Resize(257),
    T.CenterCrop(256),
    T.ToTensor(),
    Grayscale()
])

# mugshot and embedding file details
mugshot_path = Path("./Mugshot")
# names of the images in the mugshot - mugshot_idx.ext
image_names = [f_name for f_name in sorted(mugshot_path.iterdir(), key=lambda x: int(x.stem.split("_")[1]))]
emb_loc = Path("./Mugshot/embeddings.pt")  # location of the embedding file

# misc
image_exts = [".png", ".jpg", ".jpeg", ".tif", ".bmp"]


def preprocess(image, mode: int):
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
    if mode == 0:  # Sketch to Face
        generator_s2f.eval()
        processed = generator_s2f(img_tensor).squeeze()
    elif mode == 1:  # Face to Sketch
        generator_f2s.eval()
        processed = generator_f2s(img_tensor).squeeze()
    else:
        processed = transform(image).squeeze()
    assert processed.ndim == 3
    return processed


def get_embedding(image, transform_image: bool = False, transformation: int = None):
    if transform_image and transformation == 0:  # Transform into face from sketch
        sketch_image = preprocess(image, transformation)
        if sketch_image.ndim == 3:
            sketch_image = sketch_image.unsqueeze(0)
        embedding = siamese_sketch(sketch_image).squeeze()
    elif transform_image and transformation == 1:  # Transform into sketch from face
        digital_image = preprocess(image, transformation)
        if digital_image.ndim == 3:
            digital_image = digital_image.unsqueeze(0)
        embedding = siamese_photo(digital_image).squeeze()
    else:  # No transformation required
        processed = preprocess(image, -1)
        if processed.ndim == 3:
            processed = processed.unsqueeze(0)
        embedding = siamese_normal(processed)
    return embedding
