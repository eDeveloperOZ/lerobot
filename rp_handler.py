import runpod
import tempfile
import os
import torch
from huggingface_hub import HfApi
from lerobot.scripts.train import train
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.default import DatasetConfig
from lerobot.policies.factory import make_policy_config
from lerobot.datasets.utils import hf_transform_to_torch


def patched_hf_transform_to_torch(items_dict):
    """Patched version of hf_transform_to_torch that handles Column objects"""
    for key in items_dict:
        # Handle case where items_dict[key] is a Column object
        if hasattr(items_dict[key], 'to_list'):
            # Convert Column to list first
            items_dict[key] = items_dict[key].to_list()
        
        if not items_dict[key]:  # Empty list
            continue
            
        first_item = items_dict[key][0]
        if hasattr(first_item, 'convert'):  # PIL Image
            to_tensor = torch.nn.functional.to_tensor
            items_dict[key] = [to_tensor(img) for img in items_dict[key]]
        elif first_item is None:
            pass
        else:
            items_dict[key] = [x if isinstance(x, str) else torch.tensor(x) for x in items_dict[key]]
    return items_dict


def handler(job):
    """
    The handler for the Runpod serverless worker.
    It returns the public IP and the assigned TCP port.
    """
    input_data = job.get("input", {})
    hf_token = input_data.get("hf_token", "")
    dataset_repo_id = input_data.get("dataset_repo_id", "")
    policy_type = input_data.get("policy_type", "")
    job_name = input_data.get("job_name", "")
    batch_size = input_data.get("batch_size", 4)
    steps = input_data.get("steps", 100_000)
    eval_freq = input_data.get("eval_freq", 20_000)
    model_repo_id = input_data.get("model_repo_id", "")  # Where to upload the trained model
    print(f"Input data: {input_data}")
    print(f"Job: {job}")

    # Set HuggingFace token for authentication
    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
    os.environ["HF_TOKEN"] = hf_token

    # Patch the hf_transform_to_torch function to handle Column objects
    import lerobot.datasets.utils
    lerobot.datasets.utils.hf_transform_to_torch = patched_hf_transform_to_torch

    # Create a temporary directory for training outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        
        # Create the configuration object
        policy_config = make_policy_config(policy_type)
        policy_config.device = "cuda"
        policy_config.push_to_hub = False  # We'll handle upload manually
        policy_config.repo_id = model_repo_id
        policy_config.tags = ["cubix"]
        
        cfg = TrainPipelineConfig(
            dataset=DatasetConfig(repo_id=dataset_repo_id),
            policy=policy_config,
            output_dir=temp_dir,
            job_name=job_name,
            resume=False,
            num_workers=4,
            batch_size=batch_size,
            steps=steps,
            eval_freq=eval_freq,
            save_checkpoint=True,
        )
        
        # Train the policy
        print("Starting training...")
        train(cfg)
        
        # Upload the trained model to Hugging Face
        if model_repo_id:
            print(f"Uploading model to {model_repo_id}...")
            api = HfApi(token=hf_token)
            
            # Upload the entire output directory
            api.upload_folder(
                folder_path=temp_dir,
                repo_id=model_repo_id,
                repo_type="model",
                commit_message=f"Trained {policy_type} model with {steps} steps"
            )
            print(f"Model uploaded successfully to {model_repo_id}")
        else:
            print("No model_repo_id provided, skipping upload")
    
    return {
        "status": "success",
        "message": f"Training completed successfully. Model uploaded to {model_repo_id if model_repo_id else 'N/A'}",
        "policy_type": policy_type,
        "steps": steps
    }

if __name__ == '__main__':
    # The __name__ == '__main__' guard is crucial for multiprocessing.
    # It prevents child processes from re-executing the main script's code.
    
    # Start the Runpod serverless worker in the main process.
    print("Starting Runpod serverless worker")
    runpod.serverless.start({"handler": handler})
