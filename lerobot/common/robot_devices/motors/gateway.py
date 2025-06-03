"""Motor bus routing serial traffic through a WebSocket connection."""

import asyncio
import os
import pty
import threading
from typing import Any, List

import serial

from lerobot.common.robot_devices.motors.configs import GatewayMotorsBusConfig
from lerobot.common.robot_devices.utils import (
    RobotDeviceAlreadyConnectedError,
    RobotDeviceNotConnectedError,
)


class GatewayMotorsBus:
    """Wrapper bus that forwards serial commands over a WebSocket.

    ``GatewayMotorsBus`` opens a pseudo serial port on the server and bridges it
    to a remote browser via WebSocket.  The regular ``FeetechMotorsBus`` or
    ``DynamixelMotorsBus`` is then instantiated on that pseudo port, so existing
    code can operate transparently while the actual USB connection lives on the
    client side.
    """

    def __init__(self, config: GatewayMotorsBusConfig) -> None:
        self.url = config.url
        self.mock = config.mock
        self.motors = getattr(config, "motors", {}) or {}

        self.port: str | None = None
        # once connected, this becomes an instance of ``FeetechMotorsBus`` or
        # ``DynamixelMotorsBus`` created with the pseudo serial ``self.port``
        self._backend = None
        self._ws = None
        self._serial: serial.Serial | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self.is_connected = False

    # ------------------------------------------------------------------
    def _guess_backend(self) -> str:
        """Return the SDK to use for the given motors."""
        # ``GatewayMotorsBus`` doesn't know in advance whether it should use the
        # Feetech or Dynamixel implementation.  We infer it from the configured
        # motor models so ``self._backend`` can delegate all serial operations to
        # the correct bus class.
        if not self.motors:
            # default to feetech when no motors list is provided
            return "feetech"
        model = next(iter(self.motors.values()))[1].lower()
        if model.startswith("xl") or model.startswith("xm") or model.startswith("xh"):
            return "dynamixel"
        return "feetech"

    # ------------------------------------------------------------------
    def connect(self) -> None:
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                "GatewayMotorsBus is already connected."
            )
        if self.mock:
            self.is_connected = True
            return

        import websockets  # type: ignore

        # create a local pseudo-terminal pair; the real motor SDK will talk to
        # ``self.port`` while ``master`` is bridged to the WebSocket
        master, slave = pty.openpty()
        self.port = os.ttyname(slave)
        # ``pyserial`` does not support non-standard baudrates like 1M on all
        # platforms (e.g. macOS). Since this pseudo-terminal only bridges data
        # between the local process and the browser, the actual baudrate is
        # irrelevant.  We therefore use a common value to maximize
        # compatibility.  ``pyserial`` on macOS in particular sometimes fails
        # when opening the pseudo-terminal using its file name (``OSError: [Errno 34]``).
        # Using the file descriptor directly avoids this issue.
        self._serial = serial.serial_for_url(
            f"fd://{master}", baudrate=115_200, timeout=0
        )

        async def bridge() -> None:
            async with websockets.connect(self.url) as ws:
                self._ws = ws

                async def serial_to_ws() -> None:
                    while True:
                        data = self._serial.read(self._serial.in_waiting or 1)
                        if data:
                            await ws.send(data)
                        await asyncio.sleep(0.001)

                async def ws_to_serial() -> None:
                    async for message in ws:
                        if isinstance(message, str):
                            message = message.encode()
                        self._serial.write(message)

                await asyncio.gather(serial_to_ws(), ws_to_serial())

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=lambda: self._loop.run_until_complete(bridge()), daemon=True)
        self._thread.start()

        backend_type = self._guess_backend()
        if backend_type == "dynamixel":
            from lerobot.common.robot_devices.motors.dynamixel import (
                DynamixelMotorsBus,
            )
            from lerobot.common.robot_devices.motors.configs import (
                DynamixelMotorsBusConfig,
            )

            cfg = DynamixelMotorsBusConfig(
                port=self.port,
                motors=self.motors,
                mock=self.mock,
            )
            self._backend = DynamixelMotorsBus(cfg)
        else:
            from lerobot.common.robot_devices.motors.feetech import (
                FeetechMotorsBus,
            )
            from lerobot.common.robot_devices.motors.configs import (
                FeetechMotorsBusConfig,
            )

            cfg = FeetechMotorsBusConfig(
                port=self.port,
                motors=self.motors,
                mock=self.mock,
            )
            self._backend = FeetechMotorsBus(cfg)

        self._backend.connect()
        self.is_connected = True

    def disconnect(self) -> None:
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "GatewayMotorsBus is not connected."
            )
        if not self.mock:
            if self._backend is not None:
                self._backend.disconnect()
            if self._ws is not None and self._loop is not None:
                asyncio.run_coroutine_threadsafe(self._ws.close(), self._loop).result()
            if self._loop is not None:
                self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread is not None:
                self._thread.join(timeout=1)
            if self._serial is not None:
                self._serial.close()
        self.is_connected = False

    # Basic API used in real buses -------------------------------------
    def read(self, data_name: str, motor_names: List[str] | None = None) -> Any:
        if not self.is_connected:
            raise RobotDeviceNotConnectedError("GatewayMotorsBus is not connected.")
        if self.mock:
            import numpy as np

            if motor_names is None:
                motor_names = list(self.motors)
            return np.zeros(len(motor_names))
        return self._backend.read(data_name, motor_names)

    def write(self, data_name: str, values: Any, motor_names: List[str] | None = None) -> None:
        if not self.is_connected:
            raise RobotDeviceNotConnectedError("GatewayMotorsBus is not connected.")
        if self.mock:
            return
        self._backend.write(data_name, values, motor_names)

    # convenience ------------------------------------------------------
    @property
    def motor_names(self) -> List[str]:
        return list(self.motors.keys())

    @property
    def motor_models(self) -> List[str]:
        return [model for _, model in self.motors.values()]

    @property
    def motor_indices(self) -> List[int]:
        return [idx for idx, _ in self.motors.values()]
