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
# 'lerobot' is installed from source, which should pull in most dependencies
# from setup.py or pyproject.toml. We install others that are specific
# to the websocket bridge just in case.
RUN pip install .
RUN pip install websockets pyzmq opencv-python-headless

# Expose the WebSocket port that the server listens on
EXPOSE 8765

# Command to run the WebSocket bridge.
# Assumes `run_websocket_bridge.py` is in the root directory.
# The device is set to "cuda" for use with Runpod GPU instances.
CMD ["python", "run_websocket_bridge.py", "--ws-port", "8765", "--device", "cuda"] 