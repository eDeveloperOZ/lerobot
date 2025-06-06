import asyncio
import logging
from typing import Any, Dict, List

import cv2
import numpy as np
from fastapi import WebSocket

from lerobot.common.cameras import Camera
from lerobot.common.cameras.configs import CameraConfig

logger = logging.getLogger(__name__)


class RemoteCamera(Camera):
    """Camera that receives JPEG frames over a WebSocket."""

    def __init__(self) -> None:
        super().__init__(CameraConfig())
        self._queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=1)
        self.connected = False

    @property
    def is_connected(self) -> bool:
        return self.connected

    @staticmethod
    def find_cameras() -> List[Dict[str, Any]]:
        return []

    def connect(self, warmup: bool = True) -> None:  # noqa: D401
        self.connected = True

    def read(self) -> np.ndarray:
        return asyncio.run(self.async_read())

    async def async_read(self, timeout_ms: float = 1000) -> np.ndarray:
        frame = await asyncio.wait_for(self._queue.get(), timeout_ms / 1000)
        return frame

    def disconnect(self) -> None:
        self.connected = False

    def push_jpeg(self, data: bytes) -> None:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if not self._queue.empty():
            self._queue.get_nowait()
        self._queue.put_nowait(img)


class SerialMotorsBus:
    """Wrapper that detects servo type and writes actions."""

    def __init__(self, port: str) -> None:
        self.port = port
        self.bus = None
        self.motors: Dict[str, Any] = {}

    def connect(self) -> None:
        from lerobot.common.motors import Motor, MotorNormMode
        try:
            from lerobot.common.motors.dynamixel import DynamixelMotorsBus

            bus = DynamixelMotorsBus(self.port, {})
            bus.connect()
            ids_models = bus.broadcast_ping(num_retry=1) or {}
            if ids_models:
                motors = {
                    f"m{id_}": Motor(id_, bus._model_nb_to_model(model), MotorNormMode.DEGREES)
                    for id_, model in ids_models.items()
                }
                bus.motors = motors
                bus.set_baudrate(1_000_000)
                self.bus = bus
                self.motors = motors
                logger.info("Using Dynamixel bus with %s", motors)
                return
            bus.disconnect()
        except Exception:  # pragma: no cover - optional deps not available
            pass
        try:
            from lerobot.common.motors.feetech import FeetechMotorsBus

            bus = FeetechMotorsBus(self.port, {})
            bus.connect()
            ids_models = bus.broadcast_ping(num_retry=1) or {}
            if ids_models:
                motors = {
                    f"m{id_}": Motor(id_, bus._model_nb_to_model(model), MotorNormMode.DEGREES)
                    for id_, model in ids_models.items()
                }
                bus.motors = motors
                bus.set_baudrate(1_000_000)
                self.bus = bus
                self.motors = motors
                logger.info("Using Feetech bus with %s", motors)
                return
            bus.disconnect()
        except Exception:  # pragma: no cover
            pass
        logger.warning("Falling back to mock motors bus")
        self.bus = None

    def write(self, actions: Dict[str, float]) -> None:
        if not self.bus:
            logger.debug("Mock write: %s", actions)
            return
        try:
            self.bus.sync_write("Goal_Position", actions)
        except Exception as e:
            logger.error("Motor write failed: %s", e)


class InferenceLoop:
    """Continuously runs inference and drives the motors."""

    def __init__(self, camera: RemoteCamera, bus: SerialMotorsBus, model: Any, clients: List[WebSocket]):
        self.camera = camera
        self.bus = bus
        self.model = model
        self.clients = clients
        self.last_actions: Dict[str, float] = {}

    async def run(self) -> None:
        while True:
            frame = await self.camera.async_read()
            actions = self.model.policy({"camera": frame})
            clamped = {}
            for k, v in actions.items():
                last = self.last_actions.get(k, v)
                diff = max(-5.0, min(5.0, v - last))
                clamped[k] = last + diff
            self.last_actions = clamped
            self.bus.write(clamped)
            for ws in list(self.clients):
                try:
                    await ws.send_text(str(clamped))
                except Exception:
                    self.clients.remove(ws)

