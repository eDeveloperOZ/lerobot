"""FastAPI server exposing teleoperation WebSocket endpoints."""

import os
import json
from typing import Dict, Optional

from pydantic import BaseModel

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from .robot_adapter import (
    RemoteCamera,
    WebSocketTeleoperator,
    SerialMotorsBus,
    load_model,
    clamp_action,
)


class ActionMsg(BaseModel):
    """Message format for actions sent over the motors WebSocket."""

    t: float
    action: Dict[str, float] = {}

app = FastAPI()

camera = RemoteCamera()
model = None
motors = None


@app.on_event("startup")
async def startup_event() -> None:
    global model
    path = os.getenv("MODEL_PATH")
    if path:
        model = load_model(path)


@app.get("/")
async def root() -> Dict[str, str]:
    return {"status": "ok"}


@app.websocket("/motors")
async def motors_ws(ws: WebSocket) -> None:
    await ws.accept()
    params = ws.query_params
    vid = int(params.get("vid", "0"))
    pid = int(params.get("pid", "0"))
    port = SerialMotorsBus.find_port(vid, pid)
    if port is None:
        await ws.close(code=4000)
        return
    global motors
    motors = SerialMotorsBus(port, {})
    motors.connect()
    teleop = WebSocketTeleoperator(ws)
    try:
        while True:
            # Receive an action message from the browser
            data = await ws.receive_text()
            msg = ActionMsg.model_validate_json(data)
            teleop.set_action(msg.action)
            # Apply the action with safety clamping
            if motors is not None:
                motors.sync_write(
                    "Goal_Position",
                    clamp_action(teleop.get_action()),
                    normalize=True,
                )
            # Echo timestamp back for RTT measurement
            await ws.send_json({"t": msg.t})
    except WebSocketDisconnect:
        pass
    finally:
        if motors is not None:
            motors.disconnect()
            motors = None


@app.websocket("/video")
async def video_ws(ws: WebSocket) -> None:
    await ws.accept()
    try:
        while True:
            data = await ws.receive_bytes()
            frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            camera.set_frame(frame)
            # When a model and motors are available, run inference and actuate
            if model is not None and motors is not None:
                action = clamp_action(model.policy(frame))
                motors.sync_write("Goal_Position", action, normalize=True)
    except WebSocketDisconnect:
        pass


