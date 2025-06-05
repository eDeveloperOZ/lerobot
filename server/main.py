import asyncio
import os
import struct
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .auth import create_token, verify_token
from .robot_adapter import RemoteCamera, SerialMotorsBus, InferenceLoop

BASE_DIR = Path(__file__).resolve().parent.parent

app = FastAPI()
app.mount("/static", StaticFiles(directory=BASE_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (BASE_DIR / "index.html").read_text()


@app.get("/token")
def token(device: str) -> dict[str, str]:
    return {"token": create_token(device)}


@app.post("/connect_arm")
async def connect_arm(info: dict) -> dict:
    """Connect motors using USB vendor/product IDs from the browser."""
    vid = info.get("usbVendorId")
    pid = info.get("usbProductId")
    port = None
    try:
        import serial.tools.list_ports as lp

        for p in lp.comports():
            if p.vid == vid and p.pid == pid:
                port = p.device
                break
    except Exception:
        pass
    if not port:
        return {"status": "not_found"}
    app.state.bus.open(port)
    return {"status": "connected"}


@app.on_event("startup")
async def startup() -> None:
    model_path = os.environ.get("MODEL_PATH")
    if not model_path:
        raise RuntimeError("MODEL_PATH env var required")
    if os.path.exists(model_path):
        model = torch.load(model_path, map_location="cpu")
    else:
        from huggingface_hub import hf_hub_download

        model_file = hf_hub_download(model_path, filename="model.pth")
        model = torch.load(model_file, map_location="cpu")
    model.eval()
    camera = RemoteCamera()
    bus = SerialMotorsBus()
    loop = InferenceLoop(camera, bus, model)
    app.state.camera = camera
    app.state.bus = bus
    app.state.loop = loop
    asyncio.create_task(loop.run())


@app.websocket("/video")
async def video_ws(websocket: WebSocket, t: str) -> None:
    if not verify_token(t):
        await websocket.close(code=1008)
        return
    await websocket.accept()
    app.state.loop.register(websocket)
    try:
        while True:
            data = await websocket.receive_bytes()
            ts = struct.unpack("<d", data[:8])[0]
            frame_bytes = np.frombuffer(data[8:], dtype=np.uint8)
            frame = cv2.imdecode(frame_bytes, cv2.IMREAD_COLOR)
            app.state.camera.push(frame)
            await websocket.send_bytes(struct.pack("<d", ts))
    except WebSocketDisconnect:
        pass
    finally:
        app.state.loop.unregister(websocket)
