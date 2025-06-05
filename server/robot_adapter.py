import asyncio
import json
import logging
from typing import Dict, Optional, Set

import cv2
import numpy as np
from fastapi import WebSocket

from lerobot.common.cameras.camera import Camera
from lerobot.common.cameras.configs import CameraConfig
from lerobot.common.motors.motors_bus import (
    Motor,
    MotorCalibration,
    MotorNormMode,
    MotorsBus,
)
from lerobot.common.motors.dynamixel.dynamixel import DynamixelMotorsBus
from lerobot.common.motors.feetech.feetech import FeetechMotorsBus

logger = logging.getLogger(__name__)


class RemoteCamera(Camera):
    """Camera that receives frames pushed over a WebSocket."""

    def __init__(self) -> None:
        super().__init__(CameraConfig(fps=None, width=None, height=None))
        self._queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=1)
        self._connected = True

    @property
    def is_connected(self) -> bool:
        return self._connected

    @staticmethod
    def find_cameras() -> list[dict]:  # pragma: no cover - unused
        return []

    def connect(self, warmup: bool = True) -> None:  # pragma: no cover - no-op
        self._connected = True

    def read(self, color_mode=None) -> np.ndarray:
        return asyncio.run(self.async_read())

    async def async_read(self, timeout_ms: float = 1000) -> np.ndarray:
        frame = await asyncio.wait_for(self._queue.get(), timeout_ms / 1000)
        return frame

    def disconnect(self) -> None:  # pragma: no cover - no-op
        self._connected = False

    def push(self, frame: np.ndarray) -> None:
        """Push a frame received from the client."""
        if self._queue.full():
            self._queue.get_nowait()
        self._queue.put_nowait(frame)


class SerialMotorsBus(MotorsBus):
    """Wrapper that auto-detects Dynamixel or Feetech motors."""

    def __init__(self) -> None:
        super().__init__("", {})
        self._bus: Optional[MotorsBus] = None

    @property
    def is_connected(self) -> bool:  # type: ignore[override]
        return self._bus is not None and self._bus.is_connected

    # ------------------------------------------------------------------
    # Detection and high level API
    # ------------------------------------------------------------------
    def open(self, port: str) -> None:
        """Open *port* and detect supported motors."""
        self.port = port
        for Bus in (DynamixelMotorsBus, FeetechMotorsBus):
            try:
                tmp = Bus(port, {})
                tmp.connect(handshake=False)
                ids_models = tmp.broadcast_ping()
                tmp.disconnect(disable_torque=False)
                if not ids_models:
                    continue
                motors = {
                    f"j{i}": Motor(id=id_, model=tmp._model_nb_to_model(mn), norm_mode=MotorNormMode.DEGREES)
                    for i, (id_, mn) in enumerate(ids_models.items())
                }
                bus = Bus(port, motors)
                bus.connect(handshake=False)
                bus.set_baudrate(1_000_000)
                self._bus = bus
                self.motors = motors
                logger.info("Connected %s on %s", Bus.__name__, port)
                return
            except Exception as e:  # pragma: no cover - serial hard to test
                logger.debug("Detection with %s failed: %s", Bus.__name__, e)
        logger.warning("No motors found on %s - using mock bus", port)
        self._bus = None

    def write(self, actions: Dict[str, float]) -> None:
        """Write goal positions in degrees."""
        if not self._bus:
            logger.info("MockBus actions: %s", actions)
            return
        try:
            self._bus.sync_write("Goal_Position", actions)
        except Exception as e:  # pragma: no cover - serial hard to test
            logger.error("Motor write failed: %s", e)

    # ------------------------------------------------------------------
    # Abstract method delegation
    # ------------------------------------------------------------------
    def _assert_protocol_is_compatible(self, instruction_name: str) -> None:
        if self._bus:
            self._bus._assert_protocol_is_compatible(instruction_name)

    def _handshake(self) -> None:  # pragma: no cover - delegated
        if self._bus:
            self._bus._handshake()

    def _find_single_motor(self, motor: str, initial_baudrate: int | None = None) -> tuple[int, int]:
        return self._bus._find_single_motor(motor, initial_baudrate) if self._bus else (0, 0)

    def configure_motors(self) -> None:
        if self._bus:
            self._bus.configure_motors()

    def disable_torque(self, motors=None, num_retry: int = 0) -> None:
        if self._bus:
            self._bus.disable_torque(motors, num_retry)

    def _disable_torque(self, motor: int, model: str, num_retry: int = 0) -> None:
        if self._bus:
            self._bus._disable_torque(motor, model, num_retry)

    def enable_torque(self, motors=None, num_retry: int = 0) -> None:
        if self._bus:
            self._bus.enable_torque(motors, num_retry)

    @property
    def is_calibrated(self) -> bool:  # type: ignore[override]
        return self._bus.is_calibrated if self._bus else True

    def read_calibration(self) -> Dict[str, MotorCalibration]:
        return self._bus.read_calibration() if self._bus else {}

    def write_calibration(self, calibration_dict: Dict[str, MotorCalibration]) -> None:
        if self._bus:
            self._bus.write_calibration(calibration_dict)

    def _get_half_turn_homings(self, positions: Dict[str, float]) -> Dict[str, float]:
        return self._bus._get_half_turn_homings(positions) if self._bus else {}

    def _encode_sign(self, data_name: str, ids_values: Dict[int, int]) -> Dict[int, int]:
        return self._bus._encode_sign(data_name, ids_values) if self._bus else ids_values

    def _decode_sign(self, data_name: str, ids_values: Dict[int, int]) -> Dict[int, int]:
        return self._bus._decode_sign(data_name, ids_values) if self._bus else ids_values

    def _split_into_byte_chunks(self, value: int, length: int) -> list[int]:
        return self._bus._split_into_byte_chunks(value, length) if self._bus else [0] * length

    def broadcast_ping(self, num_retry: int = 0, raise_on_error: bool = False):
        return self._bus.broadcast_ping(num_retry, raise_on_error) if self._bus else {}


class InferenceLoop:
    """Runs model inference and sends actions to the motors."""

    def __init__(self, camera: RemoteCamera, motors: SerialMotorsBus, model) -> None:
        self.camera = camera
        self.motors = motors
        self.model = model
        self.prev: Dict[str, float] = {}
        self.clients: Set[WebSocket] = set()
        self.running = True

    def register(self, ws: WebSocket) -> None:
        self.clients.add(ws)

    def unregister(self, ws: WebSocket) -> None:
        self.clients.discard(ws)

    async def run(self) -> None:
        while self.running:
            frame = await self.camera.async_read()
            actions = self.model.policy({"camera": frame})
            clamped: Dict[str, float] = {}
            for k, v in actions.items():
                prev = self.prev.get(k, 0.0)
                delta = max(-5.0, min(5.0, v - prev))
                new = prev + delta
                clamped[k] = new
                self.prev[k] = new
            self.motors.write(clamped)
            msg = json.dumps(clamped)
            for ws in list(self.clients):
                try:
                    await ws.send_text(msg)
                except Exception:
                    self.clients.discard(ws)
