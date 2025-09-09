import asyncio
import pickle
from typing import Any, Dict, Optional

import cv2
import numpy as np
from serial.tools import list_ports

from lerobot.common.cameras.camera import Camera
from lerobot.common.cameras.configs import CameraConfig
from lerobot.common.teleoperators.teleoperator import Teleoperator
from lerobot.common.teleoperators.config import TeleoperatorConfig
from lerobot.common.motors.motors_bus import (
    MotorsBus,
    Motor,
    MotorCalibration,
)
from lerobot.common.motors.feetech.feetech import FeetechMotorsBus
from lerobot.common.motors.dynamixel.dynamixel import DynamixelMotorsBus


class RemoteCamera(Camera):
    """Camera wrapper that receives frames from a remote WebRTC/WebSocket source."""

    def __init__(self) -> None:
        super().__init__(CameraConfig())
        self._frame: Optional[np.ndarray] = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @staticmethod
    def find_cameras() -> list[dict]:
        """Remote camera discovery is handled on the client side."""
        return []

    def connect(self, warmup: bool = True) -> None:  # noqa: D401 - see base class
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False
        self._frame = None

    def set_frame(self, frame: np.ndarray) -> None:
        """Push a new frame received over the network."""
        self._frame = frame

    def get_frame(self) -> np.ndarray:
        """Return the last received frame."""
        if self._frame is None:
            raise RuntimeError("No frame received yet")
        return self._frame

    def read(self, color_mode=None) -> np.ndarray:  # noqa: D401 - see base class
        return self.get_frame()

    async def async_read(self, timeout_ms: float = 0.0) -> np.ndarray:
        return self.get_frame()


class WebSocketTeleoperator(Teleoperator):
    """Teleoperator that receives actions from a websocket client."""

    name = "websocket"
    config_class = TeleoperatorConfig

    def __init__(self, websocket, config: TeleoperatorConfig | None = None) -> None:
        super().__init__(config or TeleoperatorConfig())
        self.websocket = websocket
        self._action: Dict[str, Any] = {}

    @property
    def action_features(self) -> dict:
        return {}

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        from starlette.websockets import WebSocketState

        return self.websocket.application_state == WebSocketState.CONNECTED

    def connect(self, calibrate: bool = True) -> None:  # noqa: D401 - see base class
        pass

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def set_action(self, action: Dict[str, Any]) -> None:
        self._action = action

    def get_action(self) -> Dict[str, Any]:
        return self._action

    def send_feedback(self, feedback: Dict[str, Any]) -> None:
        if self.is_connected:
            asyncio.create_task(self.websocket.send_json(feedback))

    def disconnect(self) -> None:
        if self.is_connected:
            asyncio.create_task(self.websocket.close())


class SerialMotorsBus(MotorsBus):
    """Dispatcher that instantiates the right bus from USB vendor/product IDs."""

    FEETECH_VIDS = {0x0483, 0x1A86}
    DYNAMIXEL_VIDS = {0x0403}

    def __new__(
        cls,
        port: str,
        motors: Dict[str, Motor],
        calibration: Dict[str, MotorCalibration] | None = None,
    ) -> MotorsBus:
        vid = next((p.vid for p in list_ports.comports() if p.device == port), None)
        if vid in cls.FEETECH_VIDS:
            return FeetechMotorsBus(port, motors, calibration)
        if vid in cls.DYNAMIXEL_VIDS:
            return DynamixelMotorsBus(port, motors, calibration)
        raise ValueError(f"No supported motor bus detected on port {port}")

    @classmethod
    def find_port(cls, vid: int, pid: int) -> str | None:
        for p in list_ports.comports():
            if p.vid == vid and p.pid == pid:
                return p.device
        return None


def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def clamp_action(action: Dict[str, float], limit: float = 5.0) -> Dict[str, float]:
    """Clamp each joint movement to avoid unsafe jumps."""
    return {k: max(-limit, min(limit, float(v))) for k, v in action.items()}

