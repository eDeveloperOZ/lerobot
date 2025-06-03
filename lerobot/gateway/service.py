"""Backend gateway service to expose training and inference to browser clients."""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, render_template, request

_LOG = logging.getLogger(__name__)


class GatewayService:
    """Manage training and inference subprocesses."""

    def __init__(self) -> None:
        self._train_processes: Dict[str, subprocess.Popen] = {}
        self._inference_processes: Dict[str, subprocess.Popen] = {}
        self._logs: Dict[str, list[str]] = {}

    # ------------------------------------------------------------------
    def _spawn(self, cmd: list[str]) -> subprocess.Popen:
        _LOG.info("Running command: %s", " ".join(cmd))
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
        )

    def _stream_output(self, pid: str, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            self._logs.setdefault(pid, []).append(line)
            _LOG.info("[%s] %s", pid, line)
        proc.wait()
        _LOG.info("Process %s terminated with code %s", pid, proc.returncode)

    # public API -------------------------------------------------------
    def start_training(self, config: Dict[str, Any]) -> str:
        """Launch ``train.py`` with provided options."""
        dataset_repo_id = config.get("dataset_repo_id")
        if not dataset_repo_id:
            raise ValueError("dataset_repo_id is required")
        policy = config.get("policy", "act")
        env = config.get("env", "so100_real")
        extra_args = config.get("extra_args", [])
        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts" / "train.py"),
            f"policy={policy}",
            f"env={env}",
            f"dataset_repo_id={dataset_repo_id}",
        ] + extra_args
        proc = self._spawn(cmd)
        pid = str(uuid.uuid4())
        self._train_processes[pid] = proc
        self._logs[pid] = []
        threading.Thread(target=self._stream_output, args=(pid, proc), daemon=True).start()
        return pid

    def start_inference(self, config: Dict[str, Any]) -> str:
        """Run a pretrained policy on a real robot via ``control_robot.py``."""

        policy_path = config.get("policy_path") or config.get("model_path")
        if not policy_path:
            raise ValueError("policy_path is required")

        robot_type = config.get("robot_type", "so100")
        repo_id = config.get("repo_id")
        single_task = config.get("single_task")
        control_type = config.get("control_type", "record")
        ws_url = config.get("ws_url")
        extra_args = config.get("extra_args", [])

        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts" / "control_robot.py"),
            f"--robot.type={robot_type}",
            f"--control.type={control_type}",
            f"--control.policy.path={policy_path}",
        ]
        if repo_id:
            cmd.append(f"--control.repo_id={repo_id}")
        if single_task:
            cmd.append(f"--control.single_task={single_task}")
        if ws_url:
            cmd.extend(
                [
                    "--robot.leader_arms.main.type=gateway",
                    f"--robot.leader_arms.main.url={ws_url}",
                    "--robot.follower_arms.main.type=gateway",
                    f"--robot.follower_arms.main.url={ws_url}",
                ]
            )
        if extra_args:
            cmd.extend(extra_args)

        proc = self._spawn(cmd)
        pid = str(uuid.uuid4())
        self._inference_processes[pid] = proc
        self._logs[pid] = []
        threading.Thread(target=self._stream_output, args=(pid, proc), daemon=True).start()
        return pid

    def stop_session(self, pid: str) -> None:
        proc = self._train_processes.pop(pid, None) or self._inference_processes.pop(pid, None)
        if proc and proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=5)
            _LOG.info("Process %s stopped", pid)
        self._logs.pop(pid, None)

    def session_status(self, pid: str) -> str:
        proc = self._train_processes.get(pid) or self._inference_processes.get(pid)
        if not proc:
            return "unknown"
        if proc.poll() is None:
            return "running"
        return "finished"

    def session_logs(self, pid: str) -> list[str]:
        return self._logs.get(pid, [])


def create_app() -> Flask:
    """Return a Flask application wrapping :class:`GatewayService`."""

    service = GatewayService()
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/train", methods=["POST"])
    def train_endpoint() -> Any:
        config = request.get_json(force=True)
        try:
            pid = service.start_training(config)
        except Exception as exc:  # pragma: no cover - hardware dependency
            return jsonify({"error": str(exc)}), 400
        return jsonify({"session_id": pid})

    @app.route("/inference", methods=["POST"])
    def inference_endpoint() -> Any:
        data = request.get_json(force=True)
        try:
            pid = service.start_inference(data)
        except Exception as exc:  # pragma: no cover - hardware dependency
            return jsonify({"error": str(exc)}), 400
        return jsonify({"session_id": pid})

    @app.route("/session/<pid>")
    def session_status_endpoint(pid: str) -> Any:
        status = service.session_status(pid)
        return jsonify({"status": status})

    @app.route("/session/<pid>/logs")
    def session_logs_endpoint(pid: str) -> Any:
        logs = service.session_logs(pid)
        return jsonify({"logs": logs})

    @app.route("/session/<pid>", methods=["DELETE"])
    def stop_session_endpoint(pid: str) -> Any:
        service.stop_session(pid)
        return jsonify({"status": "stopped"})

    return app


def run_websocket_server(host: str = "0.0.0.0", port: int = 8765) -> None:
    """Run a lightweight WebSocket relay for streaming data."""
    try:
        import websockets  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("The 'websockets' package is required for streaming") from exc

    async def relay(websocket: websockets.WebSocketServerProtocol, _path: str) -> None:
        try:
            async for message in websocket:
                await websocket.send(message)
        except websockets.ConnectionClosed:  # pragma: no cover - network required
            pass

    asyncio.run(websockets.serve(relay, host, port))
