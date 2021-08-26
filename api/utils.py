import torch
from pathlib import Path
from PIL import Image
from typing import Union, List
from config import transform


def save_embeddings(network: torch.nn.Module, in_path: str,
                    out_file: str, rename: bool = False, start_from: int = None):
    in_path = Path(in_path)
    if not in_path.exists():
        print("Provided path does not exist")
        return -1
    if rename:  # Rename the files
        print("Renaming files")
        rename_files(in_path, "mugshot", start_from if start_from is not None else 1)
    images = [Image.open(f_name) for f_name in sorted(in_path.iterdir(), key=lambda x: int(x.stem.split("_")[1]))]
    write_embeddings(network, images, out_file)


def add_embedding(network: torch.nn.Module, img_path, out_file: str, device: str = "cpu"):
    network = network.to(device)
    network.eval()
    image = Image.open(img_path)
    sample = transform(image)
    if sample.ndim == 3:
        sample = sample.unsqueeze(0)
    emb = network.encode_samples(sample).squeeze()
    emb = emb / torch.norm(emb, dim=-1)
    if emb.ndim == 1:
        emb = emb.unsqueeze(0)
    embeddings = torch.load(out_file)
    embeddings = torch.cat([embeddings, emb], dim=0)
    torch.save(embeddings, out_file)


def write_embeddings(network: torch.nn.Module, images: List, out_file: str, device: str = "cpu"):
    tensor_images = torch.stack([transform(image) for image in images])
    network = network.to(device)
    network.eval()
    embeddings = network.encode_samples(tensor_images).squeeze()
    embeddings = embeddings / torch.norm(embeddings, dim=-1, keepdim=True)
    if embeddings.ndim == 1:
        embeddings = embeddings.unsqueeze(0)
    assert embeddings.ndim == 2  # in the file embeddings should be of shape (n, emb_dim)
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


def calc_similarity(target_embedding: torch.Tensor, src_file: str, device: str = "cpu"):
    embeddings = torch.load(src_file, map_location=device)  # Fetch embeddings from file
    assert embeddings.ndim == 2  # Shape => (n, emb_dim)
    target_embedding = target_embedding.squeeze()
    if target_embedding.ndim == 1:  # torch.mm accepts 2-d tensors
        target_embedding = target_embedding.unsqueeze(-1)
    similarities = torch.mm(embeddings, target_embedding).squeeze()  # Calculate the similarities
    assert similarities.ndim == 1
    values, indices = torch.sort(similarities, descending=True)
    return values.tolist(), indices.tolist()

# if __name__ == '__main__':
#     from torchvision.models import resnet50
#
#     sample = Image.open("sample.jpg")
#     tsfm = T.Compose([
#         T.Resize(257),
#         T.CenterCrop(256),
#         T.ToTensor(),
#         Grayscale()
#     ])
#     sample = tsfm(sample).unsqueeze(0)
#     backbone = resnet50(False)
#     siamese_network = SiameseNetwork(backbone, 2048)
#     load_network("./siamese-traditional-300.pth.tar", siamese_network, None, None)
#     if not Path("embeddings.pt").exists():
#         save_embeddings(siamese_network, "./Mugshot", "embeddings", True)
#     sample_embedding = siamese_network.encode_samples(sample)
#     sample_embedding = sample_embedding / torch.norm(sample_embedding)
#     sims, indexes = calc_similarity(sample_embedding, "embeddings.pt")
#     print(sims)
#     for sim, idx in zip(sims[:5], indexes[:5]):
#         image = cv2.imread(f"./Mugshot/mugshot_{idx+1}.jpg", cv2.IMREAD_UNCHANGED)
#         cv2.imshow(f"Sim-{sim}", image)
#
#         if cv2.waitKey(0) == 27:
#             continue
#     cv2.destroyAllWindows()
