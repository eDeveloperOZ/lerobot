# LeRobot Streaming PoC

## Setup
```bash
python -m venv venv && source venv/bin/activate && pip install .[poc]
export MODEL_PATH=/abs/path/to/model.pth
uvicorn server.main:app --reload
```
Then open [http://localhost:8000](http://localhost:8000), click **Connect Camera**, then **Connect Arm**.

