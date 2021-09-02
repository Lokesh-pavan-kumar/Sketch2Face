import aiofiles
from fastapi import FastAPI, BackgroundTasks
from fastapi import File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
from enum import Enum

from utils import save_embeddings, add_embedding, calc_similarity
from config import siamese_normal, mugshot_path, emb_loc, image_exts, get_embedding

app = FastAPI(title="Sketch2Face",
              description="API Endpoints for Sketch2Face")


origins = [
    "http://localhost",
    "http://127.0.0.1:8000",
    "http://localhost:4200",
    "http://localhost:4000",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Strategy(str, Enum):
    normal = "normal"
    face2face = "siamese-face"
    sketch2sketch = "siamese-sketch"
    ensemble = "siamese-ensemble"


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
def similarity_search(*, image: UploadFile = File(...), strategy: Strategy, num_matches: int = 5):
    """
    Takes in an input image (forensic sketch) and the number of matches to return.  
    Computes and returns **num_matches** most similar images, similarity value for the input image
    """
    image_names = [f_name for f_name in mugshot_path.iterdir() if f_name.is_file() and f_name.suffix in image_exts]
    image_names = sorted(image_names, key=lambda x: int(x.stem.split("_")[1]))

    if len(image_names) == 0:  # If there are no images in the mugshot db
        return {"message": "no images in the mugshot db"}

    image = Image.open(BytesIO(image.file.read()))
    transform = strategy.value != "normal"
    embedding = get_embedding(image, transform, strategy.value).squeeze()  # Fetch the normalized embedding
    assert embedding.ndim == 1
    similarities, ids = calc_similarity(embedding, "./Mugshot/embeddings.pt")  # Calculate similarities
    similarities, ids = similarities[:num_matches], ids[:num_matches]
    result = []  # contains the final output json
    for idx in range(min(num_matches, len(image_names))):
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
        background_task.add_task(save_embeddings, network=siamese_normal, in_path=str(mugshot_path),
                                 out_file="./Mugshot/embeddings")
    else:  # embedding file present, append to the embedding
        background_task.add_task(add_embedding, network=siamese_normal, img_path=img_name, out_file=str(emb_loc))
    return {"result": "Success", "filename": f"mugshot_{curr_count}.{extension}"}


@app.get("/details/")
def mugshot_details():
    """
    Returns the number of images in the mugshot database
    """
    img_count = len([f_name for f_name in mugshot_path.iterdir() if f_name.is_file() and f_name.suffix in image_exts])
    return {"image_count": img_count}
