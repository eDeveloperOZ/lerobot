#!/usr/bin/env python3
"""
Training script template for cloud training with LeRobot
This script is used to generate training commands for users
"""

def generate_training_script(
    dataset_repo_id: str,
    output_repo_id: str,
    hf_token: str,
    max_steps: int = 2000,
    batch_size: int = 8,
    num_workers: int = 4,
    policy_type: str = "act"
) -> str:
    """
    Generate a bash script for training a model with LeRobot
    
    Args:
        dataset_repo_id: HuggingFace dataset repository (e.g., "user/dataset-name")
        output_repo_id: HuggingFace model repository for output (e.g., "user/model-name") 
        hf_token: HuggingFace token for authentication
        max_steps: Maximum training steps
        batch_size: Training batch size
        num_workers: Number of data loader workers
        policy_type: Policy type (act, diffusion, etc.)
    
    Returns:
        Bash script as string
    """
    
    script = f"""
# Cloud Training Script for LeRobot
echo "Starting LeRobot cloud training..."
echo "Dataset: {dataset_repo_id}"
echo "Output: {output_repo_id}"
echo "Policy: {policy_type}"

# Set HuggingFace token
export HUGGINGFACE_HUB_TOKEN="{hf_token}"
export HF_TOKEN="{hf_token}"

# Login to HuggingFace
echo "Authenticating with HuggingFace..."
echo "{hf_token}" | huggingface-cli login --token-stdin

# Set training parameters
DATASET_REPO="{dataset_repo_id}"
OUTPUT_REPO="{output_repo_id}"
MAX_STEPS={max_steps}
BATCH_SIZE={batch_size}
NUM_WORKERS={num_workers}
POLICY_TYPE="{policy_type}"

# Create output directory
OUTPUT_DIR="/tmp/training_output"
mkdir -p $OUTPUT_DIR

echo "Starting training with parameters:"
echo "  Dataset: $DATASET_REPO"
echo "  Max Steps: $MAX_STEPS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Workers: $NUM_WORKERS"
echo "  Policy: $POLICY_TYPE"

# Run training
python /app/lerobot/scripts/train.py \\
    --dataset.repo_id=$DATASET_REPO \\
    --policy.type=$POLICY_TYPE \\
    --output_dir=$OUTPUT_DIR \\
    --steps=$MAX_STEPS \\
    --batch_size=$BATCH_SIZE \\
    --dataloader.num_workers=$NUM_WORKERS \\
    --device=cuda \\
    --save_checkpoint=true \\
    --eval_freq=0 \\
    --log_freq=100

# Check if training was successful
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
    
    # Upload model to HuggingFace
    echo "Uploading model to HuggingFace repository: $OUTPUT_REPO"
    
    # Find the checkpoint directory
    CHECKPOINT_DIR=$(find $OUTPUT_DIR -name "checkpoints" -type d | head -1)
    if [ -d "$CHECKPOINT_DIR" ]; then
        LAST_CHECKPOINT=$(find $CHECKPOINT_DIR -name "last" -type d | head -1)
        if [ -d "$LAST_CHECKPOINT" ]; then
            MODEL_DIR="$LAST_CHECKPOINT/pretrained_model"
            if [ -d "$MODEL_DIR" ]; then
                echo "Found model at: $MODEL_DIR"
                
                # Upload the model
                huggingface-cli upload $OUTPUT_REPO $MODEL_DIR --repo-type model
                
                if [ $? -eq 0 ]; then
                    echo "Model successfully uploaded to: https://huggingface.co/$OUTPUT_REPO"
                else
                    echo "Error: Failed to upload model to HuggingFace"
                    exit 1
                fi
            else
                echo "Error: Could not find pretrained model directory"
                exit 1
            fi
        else
            echo "Error: Could not find last checkpoint"
            exit 1
        fi
    else
        echo "Error: Could not find checkpoints directory"
        exit 1
    fi
    
    echo "Training and upload completed successfully!"
else
    echo "Error: Training failed"
    exit 1
fi

# Clean up
echo "Cleaning up temporary files..."
rm -rf $OUTPUT_DIR
echo "Training job complete!"
"""
    
    return script.strip()

def generate_basic_training_script(dataset_repo_id: str, output_repo_id: str, hf_token: str) -> str:
    """
    Generate a basic training script with default parameters
    """
    return generate_training_script(
        dataset_repo_id=dataset_repo_id,
        output_repo_id=output_repo_id, 
        hf_token=hf_token,
        max_steps=2000,
        batch_size=8,
        num_workers=4,
        policy_type="act"
    ) 