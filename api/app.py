import aiofiles
from fastapi import FastAPI, BackgroundTasks
from fastapi import File, UploadFile
from io import BytesIO
from PIL import Image

from utils import save_embeddings, add_embedding, calc_similarity
from config import transform, siamese_network, image_names, mugshot_path, emb_loc, image_exts

app = FastAPI(title="Sketch2Face",
              description="API Endpoints for Sketch2Face")


@app.get("/")
def index():
    """
    Root endpoint for the system
    """
    return {"title": "sketch2face",
            "message": "contains the api endpoints for the sketch2face application, for"
                       " more information visit the /docs section which contains the openapi spec for the endpoints"}


# TODO: Add support for multiple/ensemble models
@app.post("/get-similar/")
def similarity_search(image: UploadFile = File(...), num_matches: int = 5):
    """
    Takes in an input image (forensic sketch) and the number of matches to return.  
    Computes and returns **num_matches** most similar images, similarity value for the input image
    """
    image = transform(Image.open(BytesIO(image.file.read())))  # To torch.Tensor
    if image.ndim == 3:
        image = image.unsqueeze(0)  # Add fake batch-dim
    embedding = siamese_network.encode_samples(image).squeeze()  # Get the embedding
    assert embedding.ndim == 1
    embedding = embedding / embedding.norm()  # Normalize the embedding
    similarities, ids = calc_similarity(embedding, "./embeddings.pt")  # Calculate similarities
    similarities, ids = similarities[:num_matches], ids[:num_matches]
    result = []  # contains the final output json
    for idx in range(num_matches):
        curr_img = image_names[ids[idx]].absolute()
        result.append([similarities[idx], curr_img])
    return result


@app.post("/upload-image/")
async def add_image(background_task: BackgroundTasks, image: UploadFile = File(...)):
    """
    Takes in an input image and saves it in the mugshot database
    """
    curr_count = len([f_name for f_name in mugshot_path.iterdir() if
                      f_name.is_file() and f_name.suffix in image_exts])  # Current count of samples in the db
    extension = image.filename.split(".")[1]  # get file extension
    img_name = f"./Mugshot/mugshot_{curr_count}.{extension}"
    async with aiofiles.open(img_name, "wb") as out_image:
        content = await image.read()  # async read
        await out_image.write(content)  # async write
    if not emb_loc.exists():  # embedding file not present, rename, calc and save embeddings
        background_task.add_task(save_embeddings, network=siamese_network, in_path=str(mugshot_path),
                                 out_file="./Mugshot/embeddings", rename=True, start_from=1)
    else:  # embedding file present, append to the embedding
        background_task.add_task(add_embedding, network=siamese_network, img_path=img_name, out_file=str(emb_loc))
    return {"result": "Success", "filename": f"mugshot_{curr_count + 1}.{extension}"}


@app.get("/details/")
def mugshot_details():
    """
    Returns the number of images in the mugshot database
    """
    img_count = len([f_name for f_name in mugshot_path.iterdir() if f_name.is_file() and f_name.suffix in image_exts])
    return {"image_count": img_count}
