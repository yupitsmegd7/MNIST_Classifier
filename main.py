from pathlib import Path
from typing import Annotated

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torch import nn
from torchvision.transforms import ToTensor

app = FastAPI()

model = nn.Sequential(
    nn.Conv2d(1, 8, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(8, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(32 * 7 * 7, 10),
)

WEIGHTS_PATH = Path(__file__).with_name("digit_cnn_v3.pt")
weights = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
model.load_state_dict(weights)
model.eval()

to_tensor = ToTensor()


@app.post("/predict")
def predict(image: Annotated[UploadFile, File()]) -> dict[str, int]:
    try:
        with Image.open(image.file) as uploaded:
            digit_image = uploaded.convert("L").resize((28, 28))
    except OSError as error:
        raise HTTPException(status_code=400, detail="Upload a valid image.") from error

    # Scale pixels to 0–1 and add the batch dimension: [1, 1, 28, 28].
    batch = to_tensor(digit_image).unsqueeze(0)

    with torch.inference_mode():
        scores = model(batch)
        digit = scores.argmax(dim=1).item()

    return {"label": digit}

from fastapi.staticfiles import StaticFiles

app.mount(
    "/",
    StaticFiles(directory=Path(__file__).parent / "frontend", html=True),
    name="frontend",
)