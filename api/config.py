import torch
import torchvision.transforms as T
from torchvision.models import resnet50
from duplicates import Grayscale, SiameseNetwork, load_network
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Global use

backbone = resnet50(False)  # will load the entire siamese later, pretrained weights can be ignored
siamese_network = SiameseNetwork(backbone, 2048)
load_network("./siamese-traditional-300.pth.tar", siamese_network)

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
# names of the images in the mugshot - mugshot_idx.ext
