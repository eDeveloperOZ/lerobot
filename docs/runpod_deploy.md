# Deploying LeRobot on RunPod

This document gives a minimal example to run the cloud API on a GPU pod.

## Build and push the Docker image
```bash
docker build -t <registry>/lerobot-runpod:latest .
docker push <registry>/lerobot-runpod:latest
```

## RunPod template

```yaml
image: <registry>/lerobot-runpod:latest
cmd: uvicorn server.main:app --host 0.0.0.0 --port 8000
ports:
  - containerPort: 8000
    protocol: tcp
env:
  - name: LEROBOT_MODE
    value: "cloud"
```

Deploy the template through the RunPod dashboard. The API will be available on the exposed port.
