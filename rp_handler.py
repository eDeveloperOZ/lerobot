import os
import runpod
import multiprocessing
import asyncio
import subprocess
import base64
import json
import sys
import time
from lerobot.robots.so100_follower.websocket_bridge import websocket_bridge

# Global variable to hold the process
bridge_process = None

def handle_training_job(input_data):
    """
    Handle training job execution
    """
    try:
        # Decode the training script
        script = base64.b64decode(input_data["script"]).decode()
        print(f"Executing training script...")
        
        # Create a temporary script file
        script_path = "/tmp/training_script.sh"
        with open(script_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write("set -e\n")  # Exit on any error
            f.write(script)
        
        # Make script executable
        os.chmod(script_path, 0o755)
        
        # Execute the script
        process = subprocess.Popen(
            ["/bin/bash", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Capture output
        output_lines = []
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                line = output.strip()
                output_lines.append(line)
                print(line)  # Print to container logs
        
        # Wait for completion
        return_code = process.poll()
        
        # Clean up
        if os.path.exists(script_path):
            os.remove(script_path)
        
        if return_code == 0:
            return {
                "status": "COMPLETED",
                "message": "Training completed successfully",
                "output": "\n".join(output_lines[-50:])  # Last 50 lines
            }
        else:
            return {
                "status": "FAILED", 
                "message": f"Training failed with exit code {return_code}",
                "output": "\n".join(output_lines[-50:])  # Last 50 lines
            }
            
    except Exception as e:
        print(f"Training job error: {str(e)}")
        return {
            "status": "FAILED",
            "message": f"Training job failed: {str(e)}",
            "output": str(e)
        }

def handle_inference_job():
    """
    Handle inference job - start WebSocket bridge (current logic)
    """
    global bridge_process
    if bridge_process is None or not bridge_process.is_alive():
        bridge_process = multiprocessing.Process(target=start_bridge_process)
        bridge_process.start()
        print("Bridge process started")

    public_ip = os.environ.get('RUNPOD_PUBLIC_IP')
    tcp_port = os.environ.get('RUNPOD_TCP_PORT_8765')
    print(f"Public IP: {public_ip}, TCP Port: {tcp_port}")
    return {
        "ip": public_ip,
        "port": tcp_port
    }

def start_bridge_process():
    """This function is the entry point for the new process."""
    ws_bridge = websocket_bridge.WebSocketBridge()
    # ws_bridge.start() is a coroutine, so we need to run it in an event loop.
    asyncio.run(ws_bridge.start())
    print("Bridge process finished")

def handler(job):
    """
    The handler for the Runpod serverless worker.
    Handles both training and inference requests.
    """
    input_data = job.get("input", {})
    print(f"Input data: {input_data}")
    
    # Check request type
    request_type = input_data.get("request_type", "inference")
    
    if request_type == "training":
        print("Processing training request")
        return handle_training_job(input_data)
    else:
        print("Processing inference request")
        return handle_inference_job()

if __name__ == '__main__':
    # The __name__ == '__main__' guard is crucial for multiprocessing.
    # It prevents child processes from re-executing the main script's code.
    
    # Start the Runpod serverless worker in the main process.
    print("Starting Runpod serverless worker")
    runpod.serverless.start({"handler": handler})
