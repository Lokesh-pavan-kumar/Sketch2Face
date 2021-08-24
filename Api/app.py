from fastapi import FastAPI
from fastapi import File, UploadFile
import aiofiles

app = FastAPI(title="Sketch2Face",
              description="API Endpoints for Sketch2Face")


@app.post("/upload-image/")
async def add_image(image: UploadFile = File(...)):
    async with aiofiles.open(image.filename, "wb") as out_image:
        content = await image.read()  # async read
        await out_image.write(content)  # async write
    return {"result": "Success", "filename": image.filename}
