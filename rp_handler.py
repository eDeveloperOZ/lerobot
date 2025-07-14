import os
import runpod
import threading
import asyncio
from lerobot.robots.so100_follower.websocket_bridge import websocket_bridge

def start_bridge():
    import sys
    sys.argv = ["run_websocket_bridge.py", "--ws-port", "8765", "--device", "cuda", "--no-signals"]
    
    # Create and manage a new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    # Run the main async function until it completes (which is forever)
    loop.run_until_complete(websocket_bridge.main())

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
