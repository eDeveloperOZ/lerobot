import os
import runpod
import multiprocessing
import asyncio
from lerobot.robots.so100_follower.websocket_bridge import websocket_bridge

# Global variable to hold the process
bridge_process = None

def start_bridge_process():
    """This function is the entry point for the new process."""
    ws_bridge = websocket_bridge.WebSocketBridge()
    # ws_bridge.start() is a coroutine, so we need to run it in an event loop.
    asyncio.run(ws_bridge.start())
    print("Bridge process finished")

def handler(job):
    """
    The handler for the Runpod serverless worker.
    It returns the public IP and the assigned TCP port.
    """
    global bridge_process
    if bridge_process is None or not bridge_process.is_alive():
        bridge_process = multiprocessing.Process(target=start_bridge_process)
        bridge_process.start()
        print("Bridge process started")

    input_data = job.get("input", {})
    print(f"Input data: {input_data}")
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
    
    # Start the Runpod serverless worker in the main process.
    print("Starting Runpod serverless worker")
    runpod.serverless.start({"handler": handler})
