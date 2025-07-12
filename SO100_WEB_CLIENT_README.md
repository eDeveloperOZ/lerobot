# SO100 Web Client for LeRobot

This module allows LeRobot to control SO100 robots through a web interface (Cubix), enabling remote robot control and policy inference through a browser-based connection.

## Architecture Overview

```
LeRobot (Python)  <-->  ZMQ  <-->  WebSocket Bridge  <-->  Web Browser (Cubix)  <-->  SO100 Robot
```

The system consists of:
1. **SO100WebClient** (this module) - Implements LeRobot's Robot interface
2. **WebSocket Bridge** - Translates between ZMQ (Python) and WebSocket (browser)
3. **Cubix Web Interface** - Browser-based robot controller with camera and motor control

## Installation

1. Ensure you have LeRobot installed and set up
2. Install additional dependencies:
   ```bash
   pip install pyzmq websockets
   ```

## Usage

### Step 1: Start the WebSocket Bridge

In the LeRobot repository:
```bash
cd /Users/ofirozeri/development/lerobot
python run_websocket_bridge.py
```

This starts:
- WebSocket server on port 8765
- ZMQ PUB socket on port 5555 (for sending observations to LeRobot)
- ZMQ SUB socket on port 5556 (for receiving actions from LeRobot)

### Step 2: Connect Robot via Web Interface

1. Open the Cubix web interface in your browser
2. Connect to your SO100 robot
3. Enable LeRobot integration (you should see status change to "Connected")

### Step 3: Run LeRobot Code

#### Test Connection
```bash
cd /Users/ofirozeri/development/lerobot
python test_so100_web_client.py
```

#### Run Mock Policy
```bash
python example_so100_web_inference.py --mock --duration 30
```

#### Run Real Policy
```python
from src.lerobot.robots.so100_follower import SO100WebClient, SO100WebClientConfig

# Configure robot
config = SO100WebClientConfig(
    zmq_host="localhost",
    zmq_sub_port=5555,
    zmq_pub_port=5556
)

# Create and connect
robot = SO100WebClient(config)
robot.connect()

# Get observations
obs = robot.get_observation()
# obs contains:
# - 'motor_0.pos' to 'motor_5.pos': individual motor positions
# - 'observation.state': numpy array of all positions
# - 'front': camera frame as numpy array (H, W, 3)

# Send actions
action = {
    'motor_0.pos': 2048,  # Position for motor 0 (0-4095)
    'motor_1.pos': 2048,
    # ... etc
}
robot.send_action(action)

# Disconnect when done
robot.disconnect()
```

## Configuration

The `SO100WebClientConfig` class supports:

```python
config = SO100WebClientConfig(
    # Connection settings
    zmq_host="localhost",        # WebSocket bridge host
    zmq_sub_port=5555,          # Port for receiving observations from bridge
    zmq_pub_port=5556,          # Port for sending actions to bridge
    
    # Timing settings
    polling_timeout_ms=100,      # Timeout for observation polling
    connect_timeout_s=5.0,       # Connection timeout
    
    # Robot settings
    num_motors=6,               # Number of motors
    
    # Camera settings (can be customized)
    cameras={
        "front": {
            "width": 640,
            "height": 480,
            "fps": 30
        }
    }
)
```

## Motor Position Mapping

The SO100 uses SCS servos with position range 0-4095:
- 0 = Minimum position
- 2048 = Center position  
- 4095 = Maximum position

## Troubleshooting

### Connection Issues

1. **"Timeout waiting for WebSocket bridge to connect"**
   - Ensure the WebSocket bridge is running
   - Check that the Cubix web interface is open and connected
   - Verify firewall settings allow connections on ports 5555, 5556, 8765

2. **No observations received**
   - Check browser console for errors
   - Ensure camera permissions are granted
   - Verify robot is powered on and connected

3. **Actions not executed**
   - Check robot connection status in web interface
   - Verify motor IDs match your robot configuration
   - Check browser console for communication errors

### Performance Optimization

- The system is designed for ~30Hz observations and ~10Hz control
- Adjust `polling_timeout_ms` for different latency requirements
- Use `CONFLATE` socket option to always get latest data

## Examples

See the example scripts:
- `test_so100_web_client.py` - Basic connection and control test
- `example_so100_web_inference.py` - Template for policy inference

## Development

To extend or modify the SO100WebClient:

1. The client implements the standard LeRobot `Robot` interface
2. Observations and actions are JSON-encoded for web compatibility
3. Camera frames use JPEG compression + Base64 encoding
4. All communication is asynchronous for real-time performance

## Related Files

- **LeRobot repo:**
  - `src/lerobot/robots/so100_follower/so100_web_client.py` - Main client implementation
  - `src/lerobot/robots/so100_follower/config_so100_web_client.py` - Configuration class
  - `src/lerobot/robots/so100_follower/websocket_bridge/` - WebSocket to ZMQ bridge
  - `run_websocket_bridge.py` - Script to start the bridge server
  - `test_so100_web_client.py` - Test script
  - `example_so100_web_inference.py` - Example inference script

- **Bambot repo:**
  - `cubix/js/lerobotIntegration.js` - Browser-side integration
  - `cubix/js/robotClient.js` - WebSocket client
  - `cubix/js/cameraCapture.js` - Camera frame capture
  - `cubix/index.html` - Main web interface 