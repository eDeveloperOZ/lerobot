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
    Handle training job execution by constructing and running a single,
    unified Python script that sets up the environment, patches PyTorch,
    and executes the LeRobot training.
    """
    try:
        dataset_repo = input_data["dataset_repo"]
        output_repo = input_data["output_repo"]
        hf_token = input_data["hf_token"]

        print(f"Starting training for {dataset_repo} -> {output_repo}")

        # This unified script is executed by the python interpreter, ensuring
        # all setup and execution happens in the same process.
        unified_script = '''
import os
import sys
from pathlib import Path

# 1. Set Environment Variables
print("Setting up environment variables...")
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['HUGGINGFACE_HUB_TOKEN'] = "{hf_token}"
os.environ['HF_TOKEN'] = "{hf_token}"

# 2. Apply PyTorch GradScaler Patch
print("Applying PyTorch compatibility patch...")
try:
    import torch
    print(f"PyTorch version: {{torch.__version__}}")
    from torch.amp import GradScaler
    print("GradScaler patch not needed.")
except ImportError:
    try:
        from torch.cuda.amp import GradScaler
        import torch.amp
        torch.amp.GradScaler = GradScaler
        print("Successfully applied GradScaler patch to torch.amp.")
    except ImportError as e:
        print(f"FATAL: Failed to import GradScaler from torch.cuda.amp: {{e}}")
        sys.exit(1)

# 3. Apply LeRobot Dataset Patch
print("Applying LeRobot dataset patch for timestamp handling...")
try:
    import lerobot.datasets.lerobot_dataset
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import torch

    # Store the original __init__
    original_init = LeRobotDataset.__init__

    # Define the new __init__ with the patch
    def patched_init(self, *args, **kwargs):
        # Call the original constructor first
        original_init(self, *args, **kwargs)
        # Now, patch the timestamp loading logic
        if isinstance(self.hf_dataset["timestamp"], dict):
             print("Timestamp column already processed, skipping patch.")
             return
        print("Patching timestamp loading: converting Column to list of Tensors.")
        timestamps_list = [torch.tensor(t) for t in self.hf_dataset["timestamp"]]
        self.timestamps = torch.stack(timestamps_list).numpy()

    # Monkey-patch the class
    LeRobotDataset.__init__ = patched_init
    print("Successfully patched LeRobotDataset for timestamp handling.")

except Exception as e:
    print(f"WARNING: Failed to apply LeRobot dataset patch: {{e}}")
    # This might not be fatal, so we'll continue

# 4. Find and Execute LeRobot Training Script
print("Executing LeRobot training script...")
try:
    import lerobot
    train_script_path = Path(lerobot.__file__).parent / "scripts" / "train.py"
    
    if not train_script_path.exists():
        print(f"FATAL: LeRobot train.py not found at {{train_script_path}}")
        sys.exit(1)

    # 4. Set arguments for the training script
    sys.argv = [
        str(train_script_path),
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

    print(f"Executing: python {{' '.join(sys.argv)}}")
    
    # Execute the script's code in the current process
    exec(open(train_script_path).read())
    
except SystemExit as e:
    print(f"Training script exited with code: {{e.code}}")
    sys.exit(e.code)
except Exception as e:
    import traceback
    print("FATAL: An unexpected error occurred during training execution.")
    traceback.print_exc()
    sys.exit(1)
'''.format(
    hf_token=hf_token,
    dataset_repo=dataset_repo,
    output_repo=output_repo,
)
        
        # Execute the unified script
        cmd = [sys.executable, "-c", unified_script]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
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
            print("Unified training script completed successfully.")
        else:
            print(f"Unified training script failed with exit code {return_code}.")
            
        return output_list
            
    except KeyError as e:
        error_msg = f"Missing required input parameter: {e}"
        print(error_msg)
        return [{"output": error_msg, "exit_code": 1}]
    except Exception as e:
        error_msg = f"An unexpected error occurred in training job handler: {str(e)}"
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
