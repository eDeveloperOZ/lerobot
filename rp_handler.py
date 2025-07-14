import os
import runpod
import threading
import asyncio
from lerobot.robots.so100_follower.websocket_bridge import websocket_bridge

def start_bridge():
    import sys
    sys.argv = ["run_websocket_bridge.py", "--ws-port", "8765", "--device", "cuda"]
    asyncio.run(websocket_bridge.main())

bridge_thread = threading.Thread(target=start_bridge, daemon=True)
bridge_thread.start()

def handler(job):
    public_ip = os.environ.get('RUNPOD_PUBLIC_IP')
    tcp_port = os.environ.get('RUNPOD_TCP_PORT_8765')
    return {
        "ip": public_ip,
        "port": tcp_port
    }

runpod.serverless.start({"handler": handler})
