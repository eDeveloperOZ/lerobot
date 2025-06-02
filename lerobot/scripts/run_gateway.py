#!/usr/bin/env python
"""Launch the gateway service."""

from lerobot.gateway import create_app, run_websocket_server
import threading


def main() -> None:
    ws_thread = threading.Thread(target=run_websocket_server, daemon=True)
    ws_thread.start()
    app = create_app()
    app.run(host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
