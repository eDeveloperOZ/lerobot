FROM nvidia/cuda:12.4.1-base-ubuntu22.04

# Configure environment variables
ARG PYTHON_VERSION=3.10
ENV DEBIAN_FRONTEND=noninteractive
ENV MUJOCO_GL="egl"
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies and set up Python in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git \
    libglib2.0-0 libgl1-mesa-glx libegl1-mesa ffmpeg \
    speech-dispatcher libgeos-dev libzmq3-dev \
    python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
    && ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python \
    && python -m venv /opt/venv \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && echo "source /opt/venv/bin/activate" >> /root/.bashrc

# Copy the entire project to the working directory
COPY . /app
WORKDIR /app

# Install LeRobot with dependencies
RUN /opt/venv/bin/pip install --upgrade --no-cache-dir pip \
    && /opt/venv/bin/pip install --no-cache-dir ".[test, aloha, xarm, pusht, dynamixel, smolvla]"

# Install additional dependencies for RunPod
RUN /opt/venv/bin/pip install --no-cache-dir \
    "runpod"

# Expose the WebSocket port that the server listens on
EXPOSE 8765

# Command to run the WebSocket bridge.
# Assumes `rp_handler.py` is in the root directory.
# The device is set to "cuda" for use with Runpod GPU instances.
CMD ["/opt/venv/bin/python", "rp_handler.py"] 