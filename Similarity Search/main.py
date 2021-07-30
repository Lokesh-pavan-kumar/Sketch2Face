import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet50  # Pretrained network
import torchvision.transforms as T
from model import SiameseNetwork
from loss import ContrastiveLoss
from data import SiameseDataset, Grayscale
from utils import train_fn
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")
# training, network configuration
transform = T.Compose([
    T.Resize(257),
    T.CenterCrop(256),
    T.ToTensor(),
    Grayscale()
])

# Directory structure
# root
# | -> Samples-1
# | -> Samples-2
root = Path("")  # point towards the path of the dataset

emb_dim = 1024
bs = 16
n_epochs = 5
lr = 1e-3
alpha = 0.25
freeze = True

siamese_ds = SiameseDataset(root / "sketches", root / "photos", transform=transform)
siamese_dl = DataLoader(siamese_ds, bs, shuffle=True)

encoder_network = resnet50(True)
network = SiameseNetwork(encoder_network=encoder_network, emb_dim=emb_dim, rate=0.6, freeze=freeze).to(device)
optimizer = optim.Adam(network.parameters(), lr=lr)
loss_fn = ContrastiveLoss(alpha=alpha, device=device)

# # Uncomment to run the network
# losses = []
# for _ in range(n_epochs):
#     loss = train_fn(network, loss_fn, optimizer, siamese_dl, device)
#     losses.append(loss)
