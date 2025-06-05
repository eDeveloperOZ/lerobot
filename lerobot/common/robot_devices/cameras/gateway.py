"""Camera streaming over a WebSocket connection."""

import asyncio
import threading
import time
from typing import Any

import numpy as np

from lerobot.common.robot_devices.cameras.configs import GatewayCameraConfig
from lerobot.common.robot_devices.utils import (
    RobotDeviceAlreadyConnectedError,
    RobotDeviceNotConnectedError,
)
from lerobot.common.utils.utils import capture_timestamp_utc


class GatewayCamera:
    """Receive JPEG frames over a WebSocket."""

    def __init__(self, config: GatewayCameraConfig) -> None:
        self.url = config.url
        self.width = config.width or 640
        self.height = config.height or 480
        self.fps = config.fps
        self.mock = config.mock

        self._ws = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._frame: np.ndarray | None = None
        self._last_time = 0.0

        self.is_connected = False
        self.logs: dict[str, Any] = {}

    # ------------------------------------------------------------------
    def _run(self) -> None:
        import cv2
        import websockets  # type: ignore

        async def receive() -> None:
            async with websockets.connect(self.url) as ws:
                self._ws = ws
                async for message in ws:
                    arr = np.frombuffer(message, dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is not None:
                        now = time.perf_counter()
                        self.logs["delta_timestamp_s"] = now - self._last_time
                        self.logs["timestamp_utc"] = capture_timestamp_utc()
                        self._last_time = now
                        self._frame = img

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(receive())

    # public API -------------------------------------------------------
    def connect(self) -> None:
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError("GatewayCamera is already connected.")
        if self.mock:
            self._frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            self.is_connected = True
            return

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        # wait a tiny bit for connection establishment
        time.sleep(0.1)
        self.is_connected = True

    def read(self) -> np.ndarray:
        if not self.is_connected:
            raise RobotDeviceNotConnectedError("GatewayCamera is not connected.")
        while self._frame is None:
            time.sleep(0.01)
        return self._frame

    def async_read(self) -> np.ndarray:
        if not self.is_connected:
            raise RobotDeviceNotConnectedError("GatewayCamera is not connected.")
        return self._frame

    def disconnect(self) -> None:
        if not self.is_connected:
            raise RobotDeviceNotConnectedError("GatewayCamera is not connected.")
        if self.mock:
            self.is_connected = False
            self._frame = None
            return
        if self._loop and self._ws:
            asyncio.run_coroutine_threadsafe(self._ws.close(), self._loop).result()
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=1)
        self._frame = None
        self.is_connected = False

    def __del__(self) -> None:
        if getattr(self, "is_connected", False):
            self.disconnect()
