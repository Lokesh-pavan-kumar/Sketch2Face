from fastapi import FastAPI, BackgroundTasks
from fastapi import File, UploadFile
import aiofiles
from pathlib import Path
from utils import save_embeddings
from torchvision.models import resnet50
from utils import SiameseNetwork, load_network

backbone_net = resnet50(False)
siamese_network = SiameseNetwork(backbone_net, 2048)
load_network("./siamese-traditional-300.pth.tar", siamese_network)

app = FastAPI(title="Sketch2Face",
              description="API Endpoints for Sketch2Face")
mugshot_path = Path("./Mugshot")


@app.get("/")
def index():
    """
    Root endpoint for the system
    """
    return {"message": "endpoints for the sketch2face"}


@app.post("/upload-image/")
async def add_image(background_task: BackgroundTasks, image: UploadFile = File(...)):
    """
    Takes in an input image and saves it in the mugshot database
    """
    curr_count = len([f_name for f_name in mugshot_path.iterdir() if f_name.is_file()])
    extension = image.filename.split(".")[1]
    async with aiofiles.open(f"./Mugshot/mugshot_{curr_count}.{extension}", "wb") as out_image:
        content = await image.read()  # async read
        await out_image.write(content)  # async write
    background_task.add_task(save_embeddings, network=siamese_network, in_path=str(mugshot_path),
                             out_file="embeddings", rename=True, start_from=1)
    return {"result": "Success", "filename": f"mugshot_{curr_count + 1}.{extension}"}


@app.get("/details/")
def mugshot_details():
    img_count = 0
    emb_count = 0
    for f_name in mugshot_path.iterdir():
        if f_name.is_file():
            if f_name.suffix in [".png", ".jpg", ".jpeg"]:
                img_count += 1
            else:
                emb_count += 1
    return {"image_count": img_count, "embedding_count": emb_count}
