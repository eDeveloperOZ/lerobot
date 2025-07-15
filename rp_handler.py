import os
import runpod
import multiprocessing
import asyncio
from lerobot.robots.so100_follower.websocket_bridge import websocket_bridge

def start_bridge_process():
    """This function is the entry point for the new process."""
    import sys
    sys.argv = ["run_websocket_bridge.py", "--ws-port", "8765", "--device", "cuda", "--no-signals"]
    # asyncio.run() is safe here as it runs in the main thread of the new process.
    asyncio.run(websocket_bridge.main())

def handler(job):
    """
    The handler for the Runpod serverless worker.
    It returns the public IP and the assigned TCP port.
    """
    public_ip = os.environ.get('RUNPOD_PUBLIC_IP')
    tcp_port = os.environ.get('RUNPOD_TCP_PORT_8765')
    print(f"Public IP: {public_ip}, TCP Port: {tcp_port}")
    return {
        "ip": public_ip,
        "port": tcp_port
    }

if __name__ == '__main__':
    # The __name__ == '__main__' guard is crucial for multiprocessing.
    # It prevents child processes from re-executing the main script's code.
    
    # Run the bridge in a separate process for complete isolation.
    # bridge_process = multiprocessing.Process(target=start_bridge_process, daemon=True)
    # bridge_process.start()

    # Start the Runpod serverless worker in the main process.
    print("Starting Runpod serverless worker")
    runpod.serverless.start({"handler": handler})
