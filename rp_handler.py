import runpod
import tempfile
import os
from pathlib import Path
from huggingface_hub import HfApi, whoami
from lerobot.scripts.train import train
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.default import DatasetConfig
from lerobot.policies.factory import make_policy_config
import traceback
import json


def validate_hf_token(token):
    """Validate HuggingFace token and check permissions"""
    try:
        api = HfApi(token=token)
        user_info = whoami(token=token)
        print(f"✅ HuggingFace token validated for user: {user_info['name']}")
        return api, user_info
    except Exception as e:
        print(f"❌ HuggingFace token validation failed: {e}")
        raise ValueError(f"Invalid HuggingFace token: {e}")


def check_repo_access(api, repo_id, user_info):
    """Check if user has write access to the repository"""
    try:
        # Try to get repo info
        repo_info = api.repo_info(repo_id, repo_type="model")
        print(f"✅ Repository {repo_id} exists and is accessible")
        
        # Check if user owns the repo or has write access
        repo_author = repo_id.split('/')[0] if '/' in repo_id else repo_id
        if repo_author.lower() != user_info['name'].lower():
            print(f"⚠️ Warning: User {user_info['name']} may not have write access to {repo_id}")
        
        return True
    except Exception as e:
        print(f"❌ Repository access check failed: {e}")
        # Try to create the repo if it doesn't exist
        try:
            print(f"Attempting to create repository: {repo_id}")
            api.create_repo(repo_id, repo_type="model", private=False)
            print(f"✅ Repository {repo_id} created successfully")
            return True
        except Exception as create_error:
            print(f"❌ Failed to create repository: {create_error}")
            raise ValueError(f"Cannot access or create repository {repo_id}: {create_error}")


def handler(job):
    """
    Enhanced handler for RunPod serverless function with proper error handling
    """
    input_data = job.get("input", {})
    hf_token = input_data.get("hf_token", "")
    dataset_repo_id = input_data.get("dataset_repo_id", "")
    policy_type = input_data.get("policy_type", "")
    job_name = input_data.get("job_name", "")
    batch_size = input_data.get("batch_size", 4)
    steps = input_data.get("steps", 100_000)
    eval_freq = input_data.get("eval_freq", 20_000)
    model_repo_id = input_data.get("model_repo_id", "")
    
    print(f"🚀 Starting training job with parameters:")
    print(f"  - Dataset: {dataset_repo_id}")
    print(f"  - Model output: {model_repo_id}")
    print(f"  - Policy: {policy_type}")
    print(f"  - Steps: {steps}")
    print(f"  - Batch size: {batch_size}")

    # Validate required parameters
    if not hf_token:
        error_msg = "HuggingFace token is required"
        print(f"❌ {error_msg}")
        return {"status": "failed", "error": error_msg}
    
    if not dataset_repo_id:
        error_msg = "Dataset repository ID is required"
        print(f"❌ {error_msg}")
        return {"status": "failed", "error": error_msg}
    
    if not model_repo_id:
        error_msg = "Model repository ID is required"
        print(f"❌ {error_msg}")
        return {"status": "failed", "error": error_msg}

    try:
        # Validate HuggingFace token and get user info
        api, user_info = validate_hf_token(hf_token)
        
        # Check repository access
        check_repo_access(api, model_repo_id, user_info)
        
        # Set environment variables
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
        os.environ["HF_TOKEN"] = hf_token

        # Create unique temporary directory
        import uuid
        import time
        timestamp = int(time.time() * 1000)
        temp_dir = f"/tmp/lerobot_training_{timestamp}_{uuid.uuid4().hex[:8]}"
        print(f"📁 Using temporary directory: {temp_dir}")
        
        # Create training configuration
        policy_config = make_policy_config(policy_type)
        policy_config.device = "cuda"
        policy_config.push_to_hub = False  # We'll handle upload manually
        policy_config.repo_id = model_repo_id
        policy_config.tags = ["cubix", "lerobot"]
        
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

        print(f"🏋️ Starting training...")
        # Run training
        train(cfg)
        print(f"✅ Training completed successfully")

        # Upload the trained model to HuggingFace Hub
        print(f"📤 Uploading model to {model_repo_id}...")
        try:
            api.upload_folder(
                folder_path=str(temp_dir),
                repo_id=model_repo_id,
                repo_type="model",
                commit_message=f"Training completed - {steps} steps",
                token=hf_token  # Explicitly pass token
            )
            print(f"✅ Model uploaded successfully to {model_repo_id}")
            
            # Clean up temporary directory
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"🧹 Cleaned up temporary directory")
            
            return {
                "status": "completed", 
                "message": f"Training finished successfully. Model uploaded to {model_repo_id}",
                "model_repo_id": model_repo_id,
                "steps": steps
            }
            
        except Exception as upload_error:
            error_msg = f"Training completed but failed to upload model: {str(upload_error)}"
            print(f"❌ {error_msg}")
            print(f"Full upload error: {traceback.format_exc()}")
            
            # Return failed status since upload is part of the job
            return {
                "status": "failed",
                "error": error_msg,
                "logs": traceback.format_exc()
            }
    
    except Exception as e:
        error_msg = f"Training job failed: {str(e)}"
        print(f"❌ {error_msg}")
        print(f"Full error traceback: {traceback.format_exc()}")
        
        return {
            "status": "failed",
            "error": error_msg,
            "logs": traceback.format_exc()
        }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})