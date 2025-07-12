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
# extras that may not be needed or may fail to build.
# We then install the required packages based on pyproject.toml, letting pip
# resolve the correct versions for the container's environment.
# torch and torchvision are excluded as they are in the base image.
# pynput is excluded as it requires evdev which needs gcc to compile.
# NumPy version is pinned to 1.x for compatibility with PyTorch base image.
RUN pip install --no-deps .
RUN pip install \
    "cmake>=3.29" \
    "numpy<2.0" \
    "datasets>=2.19" \
    "deepdiff>=7.0" \
    "diffusers>=0.27" \
    "draccus==0.10.0" \
    "einops>=0.8" \
    "flask>=3.0" \
    "gdown>=5.1" \
    "gymnasium==0.29.1" \
    "h5py>=3.10" \
    "huggingface-hub[hf-transfer,cli]>=0.27" \
    "imageio[ffmpeg]>=2.34" \
    "jsonlines>=4.0" \
    "numba>=0.59" \
    "omegaconf>=2.3" \
    "opencv-python-headless<4.10" \
    "packaging>=24.2" \
    "av>=14.2" \
    "pymunk<7.0" \
    "pyserial>=3.5" \
    "pyzmq>=26.2" \
    "rerun-sdk<0.20" \
    "termcolor>=2.4" \
    "wandb>=0.16" \
    "zarr>=2.17" \
    "websockets"

# Expose the WebSocket port that the server listens on
EXPOSE 8765

# Command to run the WebSocket bridge.
# Assumes `run_websocket_bridge.py` is in the root directory.
# The device is set to "cuda" for use with Runpod GPU instances.
CMD ["python", "run_websocket_bridge.py", "--ws-port", "8765", "--device", "cuda"] 