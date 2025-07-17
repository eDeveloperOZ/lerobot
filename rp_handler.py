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

def apply_pytorch_patch():
    """
    Apply a patch to torch.amp for GradScaler compatibility if needed.
    This is necessary for some versions of PyTorch where LeRobot's
    hardcoded import fails.
    """
    try:
        # Create a small script to perform the patch.
        # This is executed in a separate process to not affect the handler.
        patch_script = """
import torch
try:
    from torch.amp import GradScaler
    print("GradScaler patch not needed.")
except ImportError:
    try:
        from torch.cuda.amp import GradScaler
        import torch.amp
        torch.amp.GradScaler = GradScaler
        print("Successfully applied GradScaler patch to torch.amp.")
    except ImportError as e:
        print(f"Failed to import GradScaler from torch.cuda.amp: {e}")
"""
        # Execute the patch script using the same python interpreter
        subprocess.run([sys.executable, "-c", patch_script], check=True)
    except Exception as e:
        print(f"An error occurred during PyTorch patching: {e}")

def handle_training_job(input_data):
    """
    Handle training job execution by constructing and running the
    LeRobot training script command.
    """
    try:
        dataset_repo = input_data["dataset_repo"]
        output_repo = input_data["output_repo"]
        hf_token = input_data["hf_token"]

        print(f"Starting training for {dataset_repo} -> {output_repo}")

        # Apply the PyTorch compatibility patch before training
        apply_pytorch_patch()

        # Set environment variables for the training process
        env = os.environ.copy()
        env["HUGGINGFACE_HUB_TOKEN"] = hf_token
        env["HF_TOKEN"] = hf_token
        # MKL threading issue fixes
        env["MKL_SERVICE_FORCE_INTEL"] = "1"
        env["MKL_THREADING_LAYER"] = "GNU"
        
        # Construct the training command
        # Arguments are based on LeRobot's train.py script
        cmd = [
            sys.executable,
            "-m", "lerobot.scripts.train",
            f"--dataset.repo_id={dataset_repo}",
            f"--policy.repo_id={output_repo}",
            "--policy.type=act",
            "--output_dir=/tmp/training_output",
            "--steps=2000",
            "--batch_size=8",
            "--num_workers=4",
            "--policy.device=cuda",
            "--save_checkpoint=true",
            "--eval_freq=0",
            "--log_freq=100",
        ]
        
        print(f"Executing command: {' '.join(cmd)}")
        
        # Execute the script and capture output in real-time
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            env=env
        )
        
        output_list = []
        for line in iter(process.stdout.readline, ''):
            line = line.strip()
            print(line)
            output_list.append({"output": line})
        
        process.stdout.close()
        return_code = process.wait()
        
        output_list.append({"exit_code": return_code})
        
        if return_code == 0:
            print("Training job completed successfully.")
        else:
            print(f"Training job failed with exit code {return_code}.")
            
        return output_list
            
    except KeyError as e:
        error_msg = f"Missing required input parameter: {e}"
        print(error_msg)
        return [{"output": error_msg, "exit_code": 1}]
    except Exception as e:
        error_msg = f"An unexpected error occurred in training job: {str(e)}"
        print(error_msg)
        return [{"output": error_msg, "exit_code": 1}]

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
