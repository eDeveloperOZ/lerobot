import runpod
import tempfile
import os
from pathlib import Path
from huggingface_hub import HfApi
from lerobot.scripts.train import train
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.default import DatasetConfig
from lerobot.policies.factory import make_policy_config


def handler(job):
    """
    Handler for RunPod serverless function
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

    # Create a unique temporary directory for training outputs
    import uuid
    temp_dir = f"/tmp/lerobot_training_{uuid.uuid4().hex}"
    os.makedirs(temp_dir, exist_ok=True)
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
        output_dir=Path(temp_dir),
        job_name=job_name,
        resume=False,
        num_workers=4,
        batch_size=batch_size,
        steps=steps,
        eval_freq=eval_freq,
        save_checkpoint=True,
    )

    # Run training
    train(cfg)

    # Upload the trained model to HuggingFace Hub
    try:
        api = HfApi(token=hf_token)
        
        # Find the latest checkpoint directory
        checkpoints_dir = Path(temp_dir) / "checkpoints"
        if checkpoints_dir.exists():
            # Get the last checkpoint
            checkpoint_dirs = [d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name != "last"]
            if checkpoint_dirs:
                # Sort by step number (assuming directory names are step numbers)
                latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.name))
                model_dir = latest_checkpoint / "pretrained_model"
                
                if model_dir.exists():
                    print(f"Uploading model from {model_dir} to {model_repo_id}")
                    api.upload_folder(
                        folder_path=str(model_dir),
                        repo_id=model_repo_id,
                        repo_type="model",
                        commit_message=f"Training completed - {steps} steps"
                    )
                    print("Model uploaded successfully!")
                else:
                    print("No model directory found in checkpoint")
            else:
                print("No checkpoint directories found")
        else:
            print("No checkpoints directory found")
    except Exception as e:
        print(f"Failed to upload model: {e}")
    
    # Clean up temporary directory
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    return {"status": "completed", "message": "Training finished successfully"}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
