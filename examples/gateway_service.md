# Gateway Service

This document describes the backend service that exposes LeRobot training and inference through a simple HTTP API and optional WebSocket relay. The gateway is meant to be deployed on a host machine with GPU resources while browser clients control local robots via Web Serial and WebRTC.

## REST API

### `POST /train`
Start a training session. Example payload:
```json
{
  "policy": "act",
  "env": "so100_real",
  "dataset_repo_id": "<your_dataset_id>",
  "extra_args": ["--steps=1000"]
}
```
The response contains a `session_id` that can be queried for status.

### `POST /inference`
Launch an inference process using a pretrained model.
Payload example:
```json
{
  "model_path": "path/to/model",
  "extra_args": ["--device=cuda"]
}
```

### `GET /session/<session_id>`
Returns the status (`running`, `finished`, or `unknown`).

### `DELETE /session/<session_id>`
Terminates a running process.

## WebSocket Relay

Calling :func:`lerobot.gateway.run_websocket_server` starts a lightweight relay used for streaming sensor data and control commands between the browser and the LeRobot process. The implementation relies on the optional `websockets` package.

## Deployment

Run the gateway using:
```bash
python lerobot/scripts/run_gateway.py
```
This launches the REST API on port `8000` and the WebSocket relay on port `8765`.

Browser clients can interact with these endpoints to orchestrate training or inference runs and exchange real‑time data with the robot hardware.
