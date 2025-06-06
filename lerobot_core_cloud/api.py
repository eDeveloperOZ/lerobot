"""FastAPI server exposing the cloud API."""
from __future__ import annotations

from fastapi import FastAPI, File, UploadFile, WebSocket
from pydantic import BaseModel

from . import infer, train, upload_calibration

app = FastAPI(title="LeRobot Remote API")


@app.get("/env")
async def read_env():
    return {"mode": "cloud"}


@app.post("/api/upload_calibration")
async def api_upload_calibration(file: UploadFile = File(...)):
    data = await file.read()
    calibration_id = upload_calibration(data)
    return {"calibration_id": calibration_id}


class TrainRequest(BaseModel):
    policy: str
    dataset: str


@app.post("/api/train")
async def api_train(req: TrainRequest):
    return train(req.policy, req.dataset)


class InferRequest(BaseModel):
    observations: list[float] | None = None


@app.post("/api/infer")
async def api_infer(req: InferRequest):
    return {"actions": infer(req.observations)}


@app.websocket("/api/video")
async def api_video(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            await ws.receive_bytes()
            await ws.send_text("ack")
    except Exception:
        await ws.close()
