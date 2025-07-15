#!/bin/bash
# The -u flag ensures that Python output is sent straight to stdout without being buffered, which is useful for logging.

# 1. Start the WebSocket bridge server as a background process.
python -u src/lerobot/robots/so100_follower/websocket_bridge/websocket_bridge.py --ws-port 8765 --device cuda --no-signals &

# 2. Start the Runpod worker as the main foreground process.
python -u rp_handler.py 