# LeRobot Streaming PoC

This proof-of-concept streams webcam frames from a browser to a FastAPI backend
running a LeRobot policy. The predicted joint positions are sent to a serial
bus controlling Dynamixel or Feetech motors.

## Usage
```bash
python -m venv venv && source venv/bin/activate && pip install .[poc]
export MODEL_PATH=/abs/path/to/model.pth
uvicorn server.main:app --reload
```
Then open `http://localhost:8000`, click **Connect Camera** followed by
**Connect Arm**.
