import threading
from pathlib import Path
from lerobot.gateway.service import GatewayService
from lerobot.common.robot_devices.motors.gateway import GatewayMotorsBus
from lerobot.common.robot_devices.motors.configs import GatewayMotorsBusConfig
from tests.utils import FEETECH_MOTORS


class DummyThread:
    def __init__(self, target, args=(), daemon=None):
        self.target = target
        self.args = args

    def start(self):
        pass


def test_start_inference_builds_command(monkeypatch):
    calls = {}

    def fake_spawn(self, cmd):
        calls["cmd"] = cmd

        class P:
            stdout = []
            returncode = 0

            def poll(self):
                return None

            def wait(self, timeout=None):
                return 0

        return P()

    monkeypatch.setattr(GatewayService, "_spawn", fake_spawn)
    monkeypatch.setattr(GatewayService, "_stream_output", lambda *a, **k: None)
    monkeypatch.setattr(threading, "Thread", DummyThread)

    service = GatewayService()
    cfg = {
        "policy_path": "model",
        "repo_id": "user/eval",
        "single_task": "task",
        "robot_type": "so100",
        "ws_url": "ws://client",
    }
    service.start_inference(cfg)
    cmd = calls["cmd"]
    assert "control_robot.py" in cmd[1]
    assert "--control.policy.path=model" in cmd
    assert "--control.repo_id=user/eval" in cmd
    assert "--robot.type=so100" in cmd
    assert "--robot.leader_arms.main.type=gateway" in cmd
    assert "--robot.leader_arms.main.url=ws://client" in cmd


def test_control_robot_parses_gateway_cli(tmp_path):
    import subprocess, sys

    script = Path("lerobot/scripts/control_robot.py").resolve()
    cmd = [
        sys.executable,
        str(script),
        "--robot.type=so100",
        "--robot.mock=true",
        "--robot.leader_arms.main.type=gateway",
        "--robot.leader_arms.main.url=ws://localhost:8765",
        "--robot.follower_arms.main.type=gateway",
        "--robot.follower_arms.main.url=ws://localhost:8765",
        "--control.type=teleoperate",
        "--control.teleop_time_s=0",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_gateway_bus_mock_mode():
    cfg = GatewayMotorsBusConfig(url="ws://localhost", motors=FEETECH_MOTORS, mock=True)
    bus = GatewayMotorsBus(cfg)
    assert not bus.is_connected
    bus.connect()
    assert bus.is_connected
    bus.write("Torque_Enable", 1)
    obs = bus.read("Present_Position")
    assert len(obs) == len(FEETECH_MOTORS)
    bus.disconnect()
    assert not bus.is_connected
