# Use an official PyTorch image with CUDA support
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# Set the working directory
WORKDIR /app

# Install system dependencies required for pyzmq and opencv
RUN apt-get update && apt-get install -y \
    libzmq3-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire project to the working directory
COPY . .

# Install Python dependencies
# 'lerobot' is installed from source. We use --no-deps to avoid installing
# the 'teleop' extra, which includes 'evdev' and requires system headers.
# We then install the required packages manually based on pyproject.toml.
RUN pip install --no-deps .
RUN pip install \
    cmake>=3.29.0.1 \
    datasets>=2.19.0 \
    deepdiff>=7.0.1 \
    diffusers>=0.27.2 \
    draccus==0.10.0 \
    einops>=0.8.0 \
    flask>=3.0.3 \
    gdown>=5.1.0 \
    gymnasium==0.29.1 \
    h5py>=3.10.0 \
    "huggingface-hub[hf-transfer,cli]>=0.27.1" \
    "imageio[ffmpeg]>=2.34.0" \
    jsonlines>=4.0.0 \
    numba>=0.59.0 \
    omegaconf>=2.3.0 \
    opencv-python-headless>=4.9.0 \
    packaging>=24.2 \
    av>=14.2.0 \
    pymunk>=6.6.0 \
    pynput>=1.7.7 \
    pyserial>=3.5 \
    pyzmq>=26.2.1 \
    rerun-sdk>=0.21.0 \
    termcolor>=2.4.0 \
    torch>=2.2.1 \
    torchvision>=0.21.0 \
    wandb>=0.16.3 \
    zarr>=2.17.0 \
    websockets

# Expose the WebSocket port that the server listens on
EXPOSE 8765

# Command to run the WebSocket bridge.
# Assumes `run_websocket_bridge.py` is in the root directory.
# The device is set to "cuda" for use with Runpod GPU instances.
CMD ["python", "run_websocket_bridge.py", "--ws-port", "8765", "--device", "cuda"] 