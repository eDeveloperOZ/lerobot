import asyncio
import json
import logging
import os
from pathlib import Path
from typing import List

import uvicorn
from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch
from serial.tools import list_ports

from .auth import create_token, require_token
from .robot_adapter import InferenceLoop, RemoteCamera, SerialMotorsBus

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI()

ROOT = Path(__file__).resolve().parent.parent
app.mount("/", StaticFiles(directory=ROOT, html=True), name="static")

camera = RemoteCamera()
clients: List[WebSocket] = []
model = None
bus: SerialMotorsBus | None = None
loop: asyncio.Task | None = None


class PortInfo(BaseModel):
    usbVendorId: int
    usbProductId: int


@app.on_event("startup")
async def load_model() -> None:
    global model
    model_path = os.environ.get("MODEL_PATH")
    if not model_path:
        raise RuntimeError("MODEL_PATH env var not set")
    if Path(model_path).exists():
        model = torch.load(model_path, map_location="cpu")
    else:
        from huggingface_hub import hf_hub_download

        file = hf_hub_download(model_path)
        model = torch.load(file, map_location="cpu")
    logger.info("Model loaded from %s", model_path)


@app.get("/token")
async def token(device: str) -> dict:
    return {"token": create_token(device)}


@app.post("/connect_arm")
async def connect_arm(info: PortInfo, token: str = Depends(require_token)) -> dict:
    global bus, loop
    port_path = None
    for p in list_ports.comports():
        if p.vid == info.usbVendorId and p.pid == info.usbProductId:
            port_path = p.device
            break
    if not port_path:
        raise RuntimeError("port not found")
    bus = SerialMotorsBus(port_path)
    bus.connect()
    if loop is None:
        inference = InferenceLoop(camera, bus, model, clients)
        loop = asyncio.create_task(inference.run())
    return {"port": port_path}


@app.websocket("/video")
async def video_ws(websocket: WebSocket) -> None:
    token = websocket.query_params.get("t")
    require_token(token)
    await websocket.accept()
    clients.append(websocket)
    try:
        while True:
            data = await websocket.receive_bytes()
            camera.push_jpeg(data)
            await websocket.send_bytes(data)
    except WebSocketDisconnect:
        clients.remove(websocket)


if __name__ == "__main__":
    uvicorn.run("server.main:app", host="0.0.0.0", port=8000)

