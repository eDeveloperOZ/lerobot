import os
import json
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Set

import torch
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.robots.so100_follower import SO100Follower, SO100FollowerConfig
from lerobot.common.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.common.cameras.configs import ColorMode

logger = logging.getLogger(__name__)

# Global state
app_state = {
    "model": None,
    "robot": None,
    "camera": None,
    "websocket_clients": set(),
    "device": None,
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ACT policy model
    model_path = os.environ.get("MODEL_PATH")
    if not model_path:
        raise RuntimeError("MODEL_PATH env var required")
    
    logger.info(f"Loading ACT policy from: {model_path}")
    model = ACTPolicy.from_pretrained(model_path)
    model.eval()
    app_state["model"] = model
    
    # Detect the device the model is on
    model_device = next(model.parameters()).device
    app_state["device"] = model_device
    logger.info(f"Model loaded on device: {model_device}")
    
    logger.info("Server startup complete")
    yield
    
    # Cleanup
    if app_state["robot"] and app_state["robot"].is_connected:
        app_state["robot"].disconnect()
    if app_state["camera"] and app_state["camera"].is_connected:
        app_state["camera"].disconnect()
    logger.info("Server shutdown complete")

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def index():
    return FileResponse("index.html")

@app.get("/token")
async def get_token(device: str):
    """Simple token endpoint for WebSocket authentication."""
    return {"token": f"tok_{device}"}

@app.post("/connect_arm")
async def connect_arm(info: dict) -> dict:
    """Connect to SO100 Feetech motors using the new LeRobot API."""
    vid = info.get("usbVendorId")
    pid = info.get("usbProductId")
    logger.info(f"Received connect_arm request: vid={vid}, pid={pid}")
    
    try:
        import serial.tools.list_ports as lp
        ports = list(lp.comports())
        port = None
        
        for p in ports:
            logger.info(f"Detected port: device={p.device}, vid={getattr(p, 'vid', None)}, pid={getattr(p, 'pid', None)}")
            if getattr(p, 'vid', None) == vid and getattr(p, 'pid', None) == pid:
                port = p.device
                break
        
        if not port and ports:
            # Fallback to first available port
            port = ports[0].device
            logger.info(f"Using fallback port: {port}")
        
        if port:
            # Create SO100Follower robot configuration with proper ID and calibration directory
            from pathlib import Path
            calibration_dir = Path(".cache/calibration/so100_follower")  # Include robot name in path
            robot_config = SO100FollowerConfig(
                port=port, 
                id="main_follower",
                calibration_dir=calibration_dir
            )
            robot = SO100Follower(robot_config)
            
            # Connect without interactive calibration
            robot.connect(calibrate=False)
            
            # Ensure calibration is properly loaded and written
            logger.info(f"Robot calibration loaded: {bool(robot.calibration)}")
            logger.info(f"Robot is_calibrated before write: {robot.is_calibrated}")
            
            if robot.calibration and not robot.is_calibrated:
                logger.info("Writing cached calibration to motors")
                
                # Check for out-of-range values and clamp them
                logger.info("Checking calibration values for range issues...")
                fixed_calibration = {}
                for motor, cal in robot.calibration.items():
                    # Clamp homing offset to valid range for 11-bit signed magnitude
                    max_offset = 2047
                    clamped_offset = max(-max_offset, min(max_offset, cal.homing_offset))
                    
                    # Clamp range values to valid positive range (0-4095)
                    clamped_range_min = max(0, min(4095, cal.range_min))
                    clamped_range_max = max(0, min(4095, cal.range_max))
                    
                    if clamped_offset != cal.homing_offset:
                        logger.warning(f"{motor}: Clamping homing_offset from {cal.homing_offset} to {clamped_offset}")
                    
                    if clamped_range_min != cal.range_min:
                        logger.warning(f"{motor}: Clamping range_min from {cal.range_min} to {clamped_range_min}")
                        
                    if clamped_range_max != cal.range_max:
                        logger.warning(f"{motor}: Clamping range_max from {cal.range_max} to {clamped_range_max}")
                    
                    from lerobot.common.motors import MotorCalibration
                    fixed_calibration[motor] = MotorCalibration(
                        id=cal.id,
                        drive_mode=cal.drive_mode,
                        homing_offset=clamped_offset,
                        range_min=clamped_range_min,
                        range_max=clamped_range_max
                    )
                
                robot.bus.write_calibration(fixed_calibration)
                
                # Verify the calibration was written successfully
                logger.info(f"Robot is_calibrated after write: {robot.is_calibrated}")
                
                if not robot.is_calibrated:
                    logger.error("Failed to write calibration to motors")
                    return {"status": "error", "message": "Failed to apply calibration to motors"}
            
            # Final verification
            if not robot.is_calibrated:
                logger.error("Robot is not calibrated after connection")
                return {"status": "error", "message": "Robot calibration failed - please check calibration files"}
            
            app_state["robot"] = robot
            logger.info(f"SO100Follower connected to {port} with calibration verified")
            return {"status": "connected", "port": port}
        else:
            return {"status": "error", "message": "No suitable port found"}
            
    except Exception as e:
        logger.error(f"Failed to connect arm: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/connect_camera")
async def connect_camera() -> dict:
    """Connect to the default webcam using OpenCV."""
    # Try multiple camera indices
    camera_indices = [0, 1]
    
    for camera_index in camera_indices:
        try:
            logger.info(f"Trying camera index {camera_index}")
            
            # Create OpenCV camera configuration - let camera use its native resolution
            camera_config = OpenCVCameraConfig(
                index_or_path=camera_index,
                fps=30,
                color_mode=ColorMode.RGB
                # Don't specify width/height to use camera's native resolution
            )
            
            camera = OpenCVCamera(camera_config)
            camera.connect()
            
            # Test if we can actually read from the camera
            test_frame = camera.async_read()
            if test_frame is None or test_frame.size == 0:
                logger.warning(f"Camera {camera_index} connected but returned empty frame")
                camera.disconnect()
                continue
                
            app_state["camera"] = camera
            logger.info(f"OpenCV camera {camera_index} connected successfully - resolution: {camera.width}x{camera.height}")
            return {"status": "connected", "camera": f"opencv_{camera_index}", "resolution": f"{camera.width}x{camera.height}"}
            
        except Exception as e:
            logger.warning(f"Camera {camera_index} failed: {e}")
            continue
    
    # If we get here, no camera worked
    return {"status": "error", "message": "No working camera found. Tried indices: " + str(camera_indices)}

@app.websocket("/video")
async def websocket_video(websocket: WebSocket, t: str):
    """WebSocket endpoint for real-time inference and action streaming."""
    await websocket.accept()
    app_state["websocket_clients"].add(websocket)
    logger.info(f"WebSocket client connected with token: {t}")
    
    try:
        while True:
            # Get observation from robot and camera
            if not app_state["robot"] or not app_state["camera"]:
                await websocket.send_text(json.dumps({"error": "Robot or camera not connected"}))
                continue
                
            if not app_state["robot"].is_connected or not app_state["camera"].is_connected:
                await websocket.send_text(json.dumps({"error": "Robot or camera disconnected"}))
                continue
            
            try:
                # Check calibration status before proceeding
                if not app_state["robot"].is_calibrated:
                    error_msg = "Robot calibration lost during operation"
                    logger.error(error_msg)
                    await websocket.send_text(json.dumps({"error": error_msg}))
                    continue
                
                # Read current robot state and camera image
                robot_obs = app_state["robot"].get_observation()
                camera_frame = app_state["camera"].async_read()
                
                # Extract robot state (joint positions)
                joint_positions = []
                for motor in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]:
                    joint_positions.append(robot_obs[f"{motor}.pos"])
                
                # Prepare observation for the ACT policy
                # Convert numpy frame to torch tensor with proper dimensions and move to model device
                device = app_state["device"]
                image_tensor = torch.from_numpy(camera_frame).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                image_tensor = image_tensor.to(device)
                
                state_tensor = torch.tensor(joint_positions, dtype=torch.float32).unsqueeze(0)
                state_tensor = state_tensor.to(device)
                
                # Create observation dict for ACT policy
                # The model expects separate image keys that get converted internally to a list
                # Since we only have one camera, we'll use the same image for both laptop and phone
                observation = {
                    "observation.state": state_tensor,
                    "observation.images.laptop": image_tensor,
                    "observation.images.phone": image_tensor
                }
                
                # Run inference
                with torch.no_grad():
                    action_tensor = app_state["model"].select_action(observation)
                
                # Convert action tensor to dict format expected by robot
                action_values = action_tensor.squeeze().tolist()
                action_dict = {}
                motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
                for i, motor in enumerate(motor_names):
                    action_dict[f"{motor}.pos"] = action_values[i]
                
                # Send action to robot
                app_state["robot"].send_action(action_dict)
                logger.info(f"Action sent to robot: {action_dict}")
                
                # Send action to all connected WebSocket clients
                action_msg = json.dumps(action_dict)
                for client in list(app_state["websocket_clients"]):
                    try:
                        await client.send_text(action_msg)
                    except Exception as e:
                        logger.warning(f"Failed to send to WebSocket client: {e}")
                        app_state["websocket_clients"].discard(client)
                
            except Exception as e:
                logger.error(f"Error in inference loop: {e}")
                # Include more detailed error information
                error_details = {
                    "error": str(e),
                    "robot_connected": app_state["robot"].is_connected if app_state["robot"] else False,
                    "robot_calibrated": app_state["robot"].is_calibrated if app_state["robot"] else False
                }
                await websocket.send_text(json.dumps(error_details))
            
            # Control loop frequency (10 Hz)
            await asyncio.sleep(0.1)
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        app_state["websocket_clients"].discard(websocket)
        logger.info("WebSocket client disconnected")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
