import os
import time
import uuid

from fastapi import FastAPI
from fastapi.exceptions import HTTPException
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel, Field

import torch
from diffusers import StableDiffusionPipeline
from torch import autocast
from datetime import datetime

with open('model.txt') as ifp:
    model = ifp.readline()

with open('float16.txt') as ifp:
    float16 = ifp.readline() == 'true'

with open('token.txt') as ifp:
    access_token = ifp.readline()

def dummy(images, **kwargs):
    return images, False

def load_pipeline(access_token, dummy, model_id='runwayml/stable-diffusion-v1-5', use_float16=false):
    device = "cuda"
    # use_auth_token=access_token
    if use_float16:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, revision="fp16")
    else:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    pipe.safety_checker = dummy
    pipe = pipe.to(device)

    return pipe

def write_log(dir_name, prompt):
    # Open the file in append & read mode ('a+')
    with open(dir_name + "/log.txt", "a+") as file_object:
        # Move read cursor to the start of file.
        file_object.seek(0)
        # If file is not empty then append '\n'
        data = file_object.read(100)
        if len(data) > 0 :
            file_object.write("\n")
        # Append text at the end of file
        file_object.write(prompt)
        file_object.close()

def generate_image(seed, w, h, gs, steps, pipe, prompt):
    generator = torch.Generator(device="cuda").manual_seed(seed)

    with autocast("cuda"):
        image = pipe(prompt.lower(), height=h, width=w, guidance_scale=gs, num_inference_steps=steps, generator=generator).images[0]

    return image

pipeline = load_pipeline(access_token, dummy, model, float16)
app = FastAPI(title="Stable Diffusion API")

class GenerationRequest(BaseModel):
    prompt: str = Field(..., title="Input prompt", description="Input prompt to be rendered")
    scale: float = Field(default=7.5, title="Scale", description="Unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))")
    steps: int = Field(default=50, title="Steps", description="Number of dim sampling steps")
    seed: int = Field(default=None, title="Seed", description="Optionally specify a seed for reproduceable results")
    width: int = Field(default=512, title="Width", description="Width of image being generated")
    height: int = Field(default=512, title="Height", description="Height of image being generated")

class GenerationResult(BaseModel):
    download_id: str = Field(..., title="Download ID", description="Identifier to download the generated image")
    time: float = Field(..., title="Time", description="Total duration of generating this image")

@app.get("/")
def home():
    return {"message": "See /docs for documentation"}

@app.post("/generate", response_model=GenerationResult)
def generate(req: GenerationRequest):
    start = time.time()
    image = generate_image(seed=req.seed, w=req.width, h=req.height, gs=req.scale, steps=req.steps, pipe=pipeline, prompt=req.prompt)
    now = datetime.now().strftime("%m%d%Y%H%M%S%f")
    folder = "/root/app/data"
    outfilename = os.path.join(folder, f"{now}.png")
    write_log(folder, now
              + " seed=" + str(req.seed)
              + " h=" + str(req.height)
              + " w=" + str(req.width)
              + " gs=" + str(req.scale)
              + " steps=" + str(req.steps)
              + " prompt=" + req.prompt)
    image.save(outfilename)
    alapsed = time.time() - start

    return GenerationResult(download_id=now, time=alapsed)

@app.get("/download/{id}", responses={200: {"description": "Image with provided ID", "content": {"image/png" : {"example": "No example available."}}}, 404: {"description": "Image not found"}})
async def download(id: str):
    path = os.path.join("/root/app/data", f"{id}.png")
    if os.path.exists(path):
        return FileResponse(path, media_type="image/png", filename=path.split(os.path.sep)[-1])
    else:
        raise HTTPException(404, detail="No such file")
