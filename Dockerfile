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

# Copy the curated requirements file
COPY requirements.docker.txt .

# Install Python dependencies
# 'lerobot' is installed from source. We use --no-deps to avoid installing
# extras that may not be needed or may fail to build.
# We then install the required packages from the curated requirements file.
RUN pip install --no-deps .
RUN pip install -r requirements.docker.txt

# Expose the WebSocket port that the server listens on
EXPOSE 8765

# Command to run the WebSocket bridge.
# Assumes `run_websocket_bridge.py` is in the root directory.
# The device is set to "cuda" for use with Runpod GPU instances.
CMD ["python", "run_websocket_bridge.py", "--ws-port", "8765", "--device", "cuda"] 