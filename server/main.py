import os
import json
import logging
import asyncio
import time
from contextlib import asynccontextmanager
from typing import Dict, Set
from pathlib import Path

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
from lerobot.common.utils.utils import get_safe_torch_device, auto_select_torch_device

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
    # Try to load the ACT policy model
    MODEL_PATH = os.environ.get("MODEL_PATH", "lerobot/act_so100_test")
    logger.info(f"Loading model from: {MODEL_PATH}")
    
    model = None
    device = None
    
    try:
        # Try to load the actual model from HuggingFace
        from lerobot.common.policies.act.modeling_act import ACTPolicy
        from lerobot.common.utils.utils import get_safe_torch_device
        
        model = ACTPolicy.from_pretrained(MODEL_PATH)
        
        # Check for forced CPU usage (useful for debugging device issues)
        force_cpu = os.environ.get("FORCE_CPU", "false").lower() == "true"
        
        if force_cpu:
            device = torch.device("cpu")
            logger.info("Forcing CPU usage due to FORCE_CPU environment variable")
        else:
            # Get the device that was configured for the model
            if hasattr(model, 'config') and hasattr(model.config, 'device'):
                device = get_safe_torch_device(model.config.device, log=True)
            else:
                # Fallback to auto-detection
                from lerobot.common.utils.utils import auto_select_torch_device
                device = auto_select_torch_device()
        
        # Ensure model is on the correct device
        model = model.to(device)
        logger.info(f"Model loaded successfully on device: {device}")
    except Exception as e:
        logger.warning(f"Failed to load model from {MODEL_PATH}: {e}")
        logger.warning("Model loading disabled - inference will not work until you provide a valid model")
        # Set device to CPU as fallback
        device = torch.device("cpu")
    
    app_state["model"] = model
    app_state["device"] = device
    
    yield
    
    # Cleanup - disconnect robot and camera with enhanced error handling
    try:
        if app_state["robot"] and app_state["robot"].is_connected:
            logger.info("Disconnecting robot during shutdown...")
            app_state["robot"].disconnect()
            logger.info("Robot disconnected successfully")
    except Exception as e:
        # Handle overload errors gracefully during shutdown
        if "overload" in str(e).lower():
            logger.warning("Overload error during robot disconnect - this is normal during shutdown")
        else:
            logger.error(f"Error disconnecting robot: {e}")
    
    try:
        if app_state["camera"] and app_state["camera"].is_connected:
            logger.info("Disconnecting camera during shutdown...")
            app_state["camera"].disconnect()
            logger.info("Camera disconnected successfully")
    except Exception as e:
        logger.warning(f"Error disconnecting camera: {e}")

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def index():
    return FileResponse("index.html")

@app.get("/token")
async def get_token(device: str):
    """Simple token endpoint for WebSocket authentication."""
    return {"token": f"tok_{device}"}

def create_custom_so100_follower(config: SO100FollowerConfig, motor_ids: dict):
    """Create a custom SO100Follower with dynamic motor IDs."""
    from lerobot.common.robots.so100_follower.so100_follower import SO100Follower
    from lerobot.common.motors.feetech import FeetechMotorsBus
    from lerobot.common.motors import Motor, MotorNormMode
    
    class CustomSO100Follower(SO100Follower):
        def __init__(self, config: SO100FollowerConfig, motor_ids: dict):
            # Don't call super().__init__ to avoid creating default motor bus
            from lerobot.common.robots.robot import Robot
            Robot.__init__(self, config)
            self.config = config
            
            # Create motor bus with custom IDs
            norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
            self.bus = FeetechMotorsBus(
                port=self.config.port,
                motors={
                    "shoulder_pan": Motor(motor_ids["shoulder_pan"], "sts3215", norm_mode_body),
                    "shoulder_lift": Motor(motor_ids["shoulder_lift"], "sts3215", norm_mode_body),
                    "elbow_flex": Motor(motor_ids["elbow_flex"], "sts3215", norm_mode_body),
                    "wrist_flex": Motor(motor_ids["wrist_flex"], "sts3215", norm_mode_body),
                    "wrist_roll": Motor(motor_ids["wrist_roll"], "sts3215", norm_mode_body),
                    "gripper": Motor(motor_ids["gripper"], "sts3215", MotorNormMode.RANGE_0_100),
                },
                calibration=self.calibration,
            )
            
            # Set up cameras (if any)
            from lerobot.common.cameras.utils import make_cameras_from_configs
            self.cameras = make_cameras_from_configs(config.cameras)
    
    return CustomSO100Follower(config, motor_ids)

async def run_motor_diagnostics(port: str, motor_ids: dict = None) -> dict:
    """Run comprehensive motor diagnostics."""
    from lerobot.common.motors.feetech import FeetechMotorsBus
    from lerobot.common.motors import Motor, MotorNormMode
    
    logger.info(f"Running motor diagnostics on port: {port}")
    
    # Use provided motor IDs or default SO100 configuration
    if motor_ids is None:
        motor_ids = {
            "shoulder_pan": 1,
            "shoulder_lift": 2,
            "elbow_flex": 3,
            "wrist_flex": 4,
            "wrist_roll": 5,
            "gripper": 6
        }
    
    logger.info(f"Using motor IDs: {motor_ids}")
    
    # Expected SO100 configuration with dynamic IDs
    expected_motors = {
        "shoulder_pan": Motor(motor_ids["shoulder_pan"], "sts3215", MotorNormMode.RANGE_M100_100),
        "shoulder_lift": Motor(motor_ids["shoulder_lift"], "sts3215", MotorNormMode.RANGE_M100_100),
        "elbow_flex": Motor(motor_ids["elbow_flex"], "sts3215", MotorNormMode.RANGE_M100_100),
        "wrist_flex": Motor(motor_ids["wrist_flex"], "sts3215", MotorNormMode.RANGE_M100_100),
        "wrist_roll": Motor(motor_ids["wrist_roll"], "sts3215", MotorNormMode.RANGE_M100_100),
        "gripper": Motor(motor_ids["gripper"], "sts3215", MotorNormMode.RANGE_0_100),
    }
    
    try:
        bus = FeetechMotorsBus(port=port, motors=expected_motors)
        bus._connect(handshake=False)
        
        found_motors = []
        missing_motors = []
        
        # Try multiple baudrates to ensure we find the motors
        baudrates_to_try = [1000000, 500000, 115200]
        
        for baudrate in baudrates_to_try:
            logger.info(f"Testing baudrate: {baudrate}")
            try:
                bus.set_baudrate(baudrate)
                
                # Test each motor individually with more retries
                for motor_name, motor in expected_motors.items():
                    if any(fm["id"] == motor.id for fm in found_motors):
                        continue  # Already found this motor
                        
                    try:
                        # More aggressive retry strategy
                        model_number = None
                        for retry in range(5):
                            try:
                                model_number = bus.ping(motor.id, num_retry=0)
                                if model_number is not None:
                                    break
                                import time
                                time.sleep(0.05)  # Small delay between retries
                            except:
                                continue
                        
                        if model_number is not None:
                            found_motors.append({
                                "id": motor.id, 
                                "name": motor_name, 
                                "model": model_number,
                                "baudrate": baudrate
                            })
                            logger.info(f"Found motor {motor.id} ({motor_name}) at baudrate {baudrate}")
                        else:
                            # Only add to missing if we haven't found it at any baudrate
                            if not any(m["id"] == motor.id for m in missing_motors):
                                missing_motors.append({"id": motor.id, "name": motor_name})
                    except Exception as e:
                        logger.debug(f"Error pinging motor {motor.id}: {e}")
                        continue
                
                # If we found all motors at this baudrate, break
                if len(found_motors) == 6:
                    logger.info(f"All motors found at baudrate {baudrate}")
                    break
                    
            except Exception as e:
                logger.debug(f"Error at baudrate {baudrate}: {e}")
                continue
        
        # Remove motors from missing list if they were found
        found_ids = [m["id"] for m in found_motors]
        missing_motors = [m for m in missing_motors if m["id"] not in found_ids]
        
        bus.disconnect(disable_torque=False)
        
        return {
            "found_motors": found_motors,
            "missing_motors": missing_motors,
            "total_found": len(found_motors),
            "total_expected": 6
        }
        
    except Exception as e:
        logger.error(f"Diagnostic failed: {e}")
        return {"error": str(e)}

def clear_robot_state():
    """Properly disconnect and clear any existing robot state."""
    if app_state["robot"]:
        try:
            if app_state["robot"].is_connected:
                logger.info("Disconnecting existing robot...")
                app_state["robot"].disconnect()
        except Exception as e:
            logger.warning(f"Error disconnecting existing robot: {e}")
        finally:
            app_state["robot"] = None
            logger.info("Robot state cleared")

@app.post("/force_connect_arm")
async def force_connect_arm(request: dict) -> dict:
    """Force connect to the SO100 follower arm with specified configuration."""
    try:
        clear_robot_state()
        
        usb_vendor_id = request.get("usbVendorId", 6790)
        usb_product_id = request.get("usbProductId", 21971)
        motor_config = request.get("motorConfig", {
            "shoulder_pan": 1, "shoulder_lift": 2, "elbow_flex": 3,
            "wrist_flex": 4, "wrist_roll": 5, "gripper": 6
        })
        
        logger.info(f"Attempting to connect robot with USB VID:PID {usb_vendor_id}:{usb_product_id}")
        logger.info(f"Motor configuration: {motor_config}")
        
        # Search for matching device
        import serial.tools.list_ports
        matching_ports = []
        for port in serial.tools.list_ports.comports():
            if port.vid == usb_vendor_id and port.pid == usb_product_id:
                matching_ports.append(port.device)
        
        if not matching_ports:
            return {
                "status": "error", 
                "message": f"No device found with USB VID:PID {usb_vendor_id}:{usb_product_id}"
            }
        
        robot_port = matching_ports[0]
        logger.info(f"Found robot at port: {robot_port}")
        
        # Create robot configuration - standard SO100 uses motor IDs 1-6 by default
        robot_config = SO100FollowerConfig(
            port=robot_port,
            id="main_follower"  # Use a standard ID for calibration file lookup
        )
        
        # Create robot instance
        robot = SO100Follower(robot_config)
        
        # Connect to robot
        robot.connect()
        logger.info("Robot connected successfully")
        
        # Check calibration status
        calibration_status = "unknown"
        calibration_message = ""
        
        try:
            is_calibrated = robot.is_calibrated
            if is_calibrated:
                calibration_status = "calibrated"
                calibration_message = "Robot is properly calibrated and ready for use."
            else:
                calibration_status = "not_calibrated"
                if robot.calibration:
                    calibration_message = "Calibration file exists but robot needs recalibration."
                else:
                    calibration_message = "No calibration found. Please calibrate the robot before use."
        except Exception as cal_error:
            logger.warning(f"Could not check calibration status: {cal_error}")
            calibration_status = "error"
            calibration_message = f"Calibration check failed: {cal_error}. Robot may need calibration."
        
        app_state["robot"] = robot
        
        return {
            "status": "success",
            "message": "Robot connected successfully",
            "port": robot_port,
            "motor_ids": list(motor_config.values()),
            "calibration_status": calibration_status,
            "calibration_message": calibration_message,
            "calibration_available": bool(robot.calibration) if robot.calibration else False
        }
        
    except Exception as e:
        logger.error(f"Error connecting robot: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/connect_arm")
async def connect_arm(info: dict) -> dict:
    """Connect to SO100 Feetech motors with enhanced diagnostics and error handling."""
    vid = info.get("usbVendorId")
    pid = info.get("usbProductId")
    force_connect = info.get("force", False)
    motor_config = info.get("motorConfig")
    logger.info(f"Received connect_arm request: vid={vid}, pid={pid}, force={force_connect}, motor_config={motor_config}")
    
    # If force connect requested, use the bypass method
    if force_connect:
        return await force_connect_arm(info)
    
    # Clear any existing robot state first
    clear_robot_state()
    
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
        
        if not port:
            return {"status": "error", "message": "No suitable port found"}
        
        # Convert motor config format if provided, default to 1-6
        motor_ids = {
            "shoulder_pan": 1,
            "shoulder_lift": 2,
            "elbow_flex": 3,
            "wrist_flex": 4,
            "wrist_roll": 5,
            "gripper": 6
        }
        if motor_config:
            motor_ids.update({
                "shoulder_pan": motor_config.get("shoulder_pan", 1),
                "shoulder_lift": motor_config.get("shoulder_lift", 2),
                "elbow_flex": motor_config.get("elbow_flex", 3),
                "wrist_flex": motor_config.get("wrist_flex", 4),
                "wrist_roll": motor_config.get("wrist_roll", 5),
                "gripper": motor_config.get("gripper", 6)
            })
        
        logger.info(f"Using motor IDs: {motor_ids}")
        
        # Run diagnostics first
        diagnostics = await run_motor_diagnostics(port, motor_ids)
        
        if "error" in diagnostics:
            return {
                "status": "error", 
                "message": f"Motor diagnostics failed: {diagnostics['error']}"
            }
        
        found_count = diagnostics["total_found"]
        expected_count = diagnostics["total_expected"]
        
        if found_count == 0:
            return {
                "status": "error",
                "message": "No motors detected. Check power supply and connections.",
                "diagnostics": diagnostics
            }
        elif found_count < expected_count:
            # If we have most motors (4+ out of 6), try connecting anyway
            if found_count >= 4:
                logger.warning(f"Partial motor detection ({found_count}/6) but attempting connection anyway")
                missing_names = [m["name"] for m in diagnostics["missing_motors"]]
                
                # Try to connect with available motors
                try:
                    from pathlib import Path
                    calibration_dir = Path(".cache/calibration/so100_follower")
                    robot_config = SO100FollowerConfig(
                        port=port, 
                        id="main_follower",
                        calibration_dir=calibration_dir
                    )
                    
                    # Create custom robot with dynamic motor IDs
                    logger.info(f"Creating custom SO100Follower with motor IDs: {motor_ids}")
                    robot = create_custom_so100_follower(robot_config, motor_ids)
                    
                    # Connect with handshake disabled to avoid strict motor checking
                    robot.bus._connect(handshake=False)
                    
                    app_state["robot"] = robot
                    logger.warning(f"SO100Follower connected with {found_count}/6 motors. Missing: {missing_names}")
                    
                    return {
                        "status": "connected", 
                        "port": port,
                        "motor_ids": motor_ids,
                        "motors_found": found_count,
                        "diagnostics": diagnostics,
                        "warning": f"Connected with {found_count}/6 motors. Missing: {', '.join(missing_names)}"
                    }
                    
                except Exception as e:
                    logger.error(f"Failed to connect with partial motors: {e}")
                    # Fall through to error case below
            
            # Provide detailed diagnostic information for partial detection
            missing_names = [m["name"] for m in diagnostics["missing_motors"]]
            return {
                "status": "error",
                "message": f"Only {found_count}/{expected_count} motors detected. Missing: {', '.join(missing_names)}. "
                         "This indicates a daisy-chain break or power issue. Check connections after the detected motors.",
                "diagnostics": diagnostics,
                "troubleshooting": {
                    "likely_cause": "Daisy-chain communication break",
                    "suggestions": [
                        "Check physical connections between motors",
                        "Verify power supply can handle all 6 motors", 
                        "Inspect cables for damage or loose connections",
                        "Test each motor individually if possible",
                        "Try connecting anyway if 4+ motors detected"
                    ]
                }
            }
        
        # If all motors detected, proceed with connection
        logger.info(f"All {found_count} motors detected. Proceeding with connection...")
        
        # Create SO100Follower robot configuration
        from pathlib import Path
        calibration_dir = Path(".cache/calibration/so100_follower")
        robot_config = SO100FollowerConfig(
            port=port, 
            id="main_follower",
            calibration_dir=calibration_dir
        )
        
        # Create custom robot with dynamic motor IDs
        logger.info(f"Creating custom SO100Follower with motor IDs: {motor_ids}")
        robot = create_custom_so100_follower(robot_config, motor_ids)
        
        # Connect without interactive calibration, using reduced communication frequency
        robot.connect(calibrate=False)
        
        # Handle calibration with better error handling
        logger.info(f"Robot calibration loaded: {bool(robot.calibration)}")
        logger.info(f"Robot is_calibrated before write: {robot.is_calibrated}")
        
        if robot.calibration and not robot.is_calibrated:
            logger.info("Writing cached calibration to motors")
            
            # Check for out-of-range values and clamp them
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
            
            # Write calibration with enhanced error handling and retry logic
            try:
                robot.bus.write_calibration(fixed_calibration)
                logger.info("Calibration written successfully")
            except Exception as e:
                logger.error(f"Failed to write calibration: {e}")
                # Continue anyway - the robot might still be functional
        
        app_state["robot"] = robot
        logger.info(f"SO100Follower connected to {port} with motor IDs: {motor_ids}")
        
        return {
            "status": "connected", 
            "port": port,
            "motor_ids": motor_ids,
            "motors_found": found_count,
            "diagnostics": diagnostics
        }
        
    except Exception as e:
        logger.error(f"Failed to connect arm: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/diagnose_motors")
async def diagnose_motors(info: dict) -> dict:
    """Run motor diagnostics without attempting connection."""
    vid = info.get("usbVendorId")
    pid = info.get("usbProductId")
    motor_config = info.get("motorConfig")
    logger.info(f"Received diagnose_motors request: vid={vid}, pid={pid}, motor_config={motor_config}")
    
    try:
        import serial.tools.list_ports as lp
        ports = list(lp.comports())
        port = None
        
        for p in ports:
            if getattr(p, 'vid', None) == vid and getattr(p, 'pid', None) == pid:
                port = p.device
                break
        
        if not port and ports:
            port = ports[0].device
        
        if not port:
            return {"status": "error", "message": "No suitable port found"}
        
        # Convert motor config format if provided
        motor_ids = None
        if motor_config:
            motor_ids = {
                "shoulder_pan": motor_config.get("shoulder_pan", 1),
                "shoulder_lift": motor_config.get("shoulder_lift", 2),
                "elbow_flex": motor_config.get("elbow_flex", 3),
                "wrist_flex": motor_config.get("wrist_flex", 4),
                "wrist_roll": motor_config.get("wrist_roll", 5),
                "gripper": motor_config.get("gripper", 6)
            }
        
        diagnostics = await run_motor_diagnostics(port, motor_ids)
        
        if "error" in diagnostics:
            return {"status": "error", "message": diagnostics["error"]}
        
        return {
            "status": "success",
            "port": port,
            "motor_config": motor_ids,
            "diagnostics": diagnostics
        }
        
    except Exception as e:
        logger.error(f"Motor diagnostics failed: {e}")
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
    
    consecutive_errors = 0
    max_consecutive_errors = 5
    
    try:
        while True:
            # Get observation from robot and camera
            if not app_state["robot"] or not app_state["camera"]:
                await websocket.send_text(json.dumps({"error": "Robot or camera not connected"}))
                await asyncio.sleep(1.0)  # Wait longer if not connected
                continue
                
            if not app_state["robot"].is_connected or not app_state["camera"].is_connected:
                await websocket.send_text(json.dumps({"error": "Robot or camera disconnected"}))
                await asyncio.sleep(1.0)  # Wait longer if disconnected
                continue
            
            try:
                # Check calibration status before proceeding
                if not app_state["robot"].is_calibrated:
                    error_msg = "Robot calibration lost during operation"
                    logger.error(error_msg)
                    await websocket.send_text(json.dumps({"error": error_msg}))
                    await asyncio.sleep(1.0)
                    continue
                
                # Check if model is available
                if not app_state["model"]:
                    error_msg = "No AI model loaded - inference disabled. Set MODEL_PATH environment variable to a valid HuggingFace model."
                    await websocket.send_text(json.dumps({"error": error_msg}))
                    await asyncio.sleep(2.0)
                    continue
                
                # Read current robot state and camera image with enhanced error handling
                robot_obs = None
                camera_frame = None
                
                # Try to get robot observation with retries and exponential backoff
                for retry in range(3):
                    try:
                        robot_obs = app_state["robot"].get_observation()
                        break
                    except Exception as e:
                        error_msg = str(e)
                        if "overload" in error_msg.lower():
                            # For overload errors, wait longer between retries
                            wait_time = 0.1 * (2 ** retry)  # Exponential backoff: 0.1, 0.2, 0.4 seconds
                            logger.warning(f"Overload error during observation (attempt {retry + 1}): {e}")
                            if retry < 2:
                                await asyncio.sleep(wait_time)
                            else:
                                # After 3 attempts, skip this cycle
                                logger.error(f"Failed to get robot observation after 3 attempts due to overload: {e}")
                                consecutive_errors += 1
                                raise e
                        else:
                            if retry < 2:
                                logger.warning(f"Failed to get robot observation (attempt {retry + 1}): {e}")
                                await asyncio.sleep(0.05)  # Small delay before retry
                            else:
                                raise e
                
                if robot_obs is None:
                    await websocket.send_text(json.dumps({"error": "Failed to get robot state after 3 attempts"}))
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        logger.error(f"Too many consecutive errors ({consecutive_errors}), slowing down inference")
                        await asyncio.sleep(2.0)  # Longer break to let motors recover
                        consecutive_errors = 0
                    else:
                        await asyncio.sleep(0.5)  # Wait before retry
                    continue
                
                # Get camera frame
                try:
                    camera_frame = app_state["camera"].async_read()
                    if camera_frame is None or camera_frame.size == 0:
                        await websocket.send_text(json.dumps({"error": "Camera returned empty frame"}))
                        await asyncio.sleep(0.2)
                        continue
                except Exception as e:
                    logger.warning(f"Failed to get camera frame: {e}")
                    await websocket.send_text(json.dumps({"error": f"Camera error: {str(e)}"}))
                    await asyncio.sleep(0.2)
                    continue
                
                # Extract robot state (joint positions) - use actual motor names from connected robot
                joint_positions = []
                motor_names = list(app_state["robot"].bus.motors.keys())  # Get actual motor names from robot
                for motor in motor_names:
                    joint_positions.append(robot_obs[f"{motor}.pos"])
                
                # Prepare observation for the ACT policy
                # Convert numpy frame to torch tensor with proper dimensions and move to model device
                device = app_state["device"]
                
                # Ensure device is consistent and properly set
                if device is None:
                    device = torch.device("cpu")
                    logger.warning("Device was None, falling back to CPU")
                
                # Convert camera frame to tensor on the correct device
                image_tensor = torch.from_numpy(camera_frame).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                image_tensor = image_tensor.to(device)
                
                # Convert joint positions to tensor on the correct device
                state_tensor = torch.tensor(joint_positions, dtype=torch.float32, device=device).unsqueeze(0)
                
                # Create observation dict for ACT policy
                # The model expects separate image keys that get converted internally to a list
                # Since we only have one camera, we'll use the same image for both laptop and phone
                observation = {
                    "observation.state": state_tensor,
                    "observation.images.laptop": image_tensor,
                    "observation.images.phone": image_tensor
                }
                
                # Verify all tensors are on the same device before inference
                for key, tensor in observation.items():
                    if isinstance(tensor, torch.Tensor) and tensor.device != device:
                        logger.warning(f"Tensor {key} is on {tensor.device}, moving to {device}")
                        observation[key] = tensor.to(device)
                
                # Run inference with proper device context
                with torch.no_grad():
                    action_tensor = app_state["model"].select_action(observation)
                
                # Convert action tensor to dict format expected by robot
                action_values = action_tensor.squeeze().tolist()
                action_dict = {}
                for i, motor in enumerate(motor_names):
                    action_dict[f"{motor}.pos"] = action_values[i]
                
                # Send action to robot with enhanced error handling
                action_status = "sent"
                try:
                    app_state["robot"].send_action(action_dict)
                except Exception as e:
                    error_msg = str(e)
                    if "overload" in error_msg.lower():
                        logger.warning(f"Overload error during action send: {e}")
                        action_status = "overload_error"
                        consecutive_errors += 1
                    else:
                        logger.warning(f"Failed to send action to robot: {e}")
                        action_status = f"failed: {str(e)}"
                        consecutive_errors += 1
                
                # Send status to WebSocket clients (less verbose logging)
                status_msg = json.dumps({
                    "action": action_dict,
                    "joint_positions": joint_positions,
                    "action_status": action_status,
                    "motor_ids": [app_state["robot"].bus.motors[motor].id for motor in motor_names],
                    "timestamp": time.time()
                })
                
                for client in list(app_state["websocket_clients"]):
                    try:
                        await client.send_text(status_msg)
                    except Exception as e:
                        logger.warning(f"Failed to send to WebSocket client: {e}")
                        app_state["websocket_clients"].discard(client)
                
                # Reset consecutive error counter on success
                if action_status == "sent":
                    consecutive_errors = 0
                
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Error in inference loop: {e}")
                
                # Include more detailed error information
                error_details = {
                    "error": str(e),
                    "robot_connected": app_state["robot"].is_connected if app_state["robot"] else False,
                    "robot_calibrated": app_state["robot"].is_calibrated if app_state["robot"] else False,
                    "consecutive_errors": consecutive_errors,
                    "timestamp": time.time()
                }
                await websocket.send_text(json.dumps(error_details))
                
                # If too many consecutive errors, take a longer break
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(f"Too many consecutive errors ({consecutive_errors}), taking extended break")
                    await asyncio.sleep(3.0)  # Extended break to let system recover
                    consecutive_errors = 0
            
            # Control loop frequency - adaptive based on error state
            if consecutive_errors > 0:
                # Slow down if we're having errors
                await asyncio.sleep(0.5)  # 2 Hz when errors occur
            else:
                # Normal frequency when everything is working
                await asyncio.sleep(0.3)  # ~3 Hz normal operation
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        app_state["websocket_clients"].discard(websocket)
        logger.info("WebSocket client disconnected")

@app.post("/check_calibration")
async def check_calibration() -> dict:
    """Check the calibration status of the connected robot."""
    if not app_state["robot"]:
        return {"status": "error", "message": "No robot connected"}
    
    try:
        robot = app_state["robot"]
        has_calibration_file = bool(robot.calibration)
        is_calibrated = robot.is_calibrated if robot.is_connected else False
        
        calibration_info = {}
        if robot.calibration:
            calibration_info = {
                motor: {
                    "id": cal.id,
                    "homing_offset": cal.homing_offset,
                    "range_min": cal.range_min,
                    "range_max": cal.range_max
                }
                for motor, cal in robot.calibration.items()
            }
        
        return {
            "status": "success",
            "has_calibration_file": has_calibration_file,
            "is_calibrated": is_calibrated,
            "motor_ids": [robot.bus.motors[motor].id for motor in robot.bus.motors.keys()],
            "calibration_info": calibration_info
        }
        
    except Exception as e:
        logger.error(f"Error checking calibration: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/start_guided_calibration")
async def start_guided_calibration() -> dict:
    """Start a guided calibration process for the robot."""
    if not app_state["robot"]:
        return {"status": "error", "message": "No robot connected"}
    
    if not app_state["robot"].is_connected:
        return {"status": "error", "message": "Robot not connected"}
    
    try:
        robot = app_state["robot"]
        
        # Set up for calibration - disable torque and set position mode
        from lerobot.common.motors.feetech import OperatingMode
        
        robot.bus.disable_torque()
        for motor in robot.bus.motors:
            robot.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
        
        return {
            "status": "success",
            "message": "Robot prepared for calibration. Motors disabled for manual positioning.",
            "instructions": [
                "1. Manually move the robot to the middle of its range of motion",
                "2. Ensure all joints are in their center positions",
                "3. Click 'Set Home Position' when ready",
                "4. Then move each joint through its full range of motion",
                "5. Click 'Record Range' to capture the motion limits"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error starting calibration: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/set_home_position")
async def set_home_position() -> dict:
    """Set the current position as the home/center position for calibration."""
    if not app_state["robot"]:
        return {"status": "error", "message": "No robot connected"}
    
    try:
        robot = app_state["robot"]
        
        # Read current positions
        current_positions = robot.bus.sync_read("Present_Position", normalize=False)
        
        # Calculate homing offsets (middle position technique)
        homing_offsets = robot.bus._get_half_turn_homings(current_positions)
        
        # Store temporarily in app state for the calibration process
        app_state["calibration_temp"] = {
            "homing_offsets": homing_offsets,
            "current_positions": current_positions
        }
        
        return {
            "status": "success",
            "message": "Home position set successfully",
            "positions": current_positions,
            "homing_offsets": homing_offsets,
            "next_instruction": "Now move each joint through its full range of motion, then click 'Record Range'"
        }
        
    except Exception as e:
        logger.error(f"Error setting home position: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/record_range")
async def record_range() -> dict:
    """Record the range of motion for all joints and complete calibration."""
    if not app_state["robot"]:
        return {"status": "error", "message": "No robot connected"}
    
    if "calibration_temp" not in app_state:
        return {"status": "error", "message": "Home position not set. Please set home position first."}
    
    try:
        robot = app_state["robot"]
        
        # For SO100, we use a simplified approach
        # Most joints use recorded ranges, wrist_roll gets full range
        unknown_range_motors = [motor for motor in robot.bus.motors if motor != "wrist_roll"]
        
        logger.info("Recording range of motion for joints...")
        range_mins, range_maxes = robot.bus.record_ranges_of_motion(unknown_range_motors)
        
        # Set full range for wrist_roll (continuous rotation joint)
        range_mins["wrist_roll"] = 0
        range_maxes["wrist_roll"] = 4095
        
        # Get homing offsets from temporary storage
        homing_offsets = app_state["calibration_temp"]["homing_offsets"]
        
        # Create calibration objects
        from lerobot.common.motors import MotorCalibration
        calibration = {}
        for motor, m in robot.bus.motors.items():
            calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )
        
        # Write calibration to motors and save to file
        robot.bus.write_calibration(calibration)
        robot.calibration = calibration
        robot._save_calibration()
        
        # Clean up temporary data
        del app_state["calibration_temp"]
        
        # Re-enable torque
        robot.bus.enable_torque()
        
        logger.info(f"Calibration completed and saved to {robot.calibration_fpath}")
        
        return {
            "status": "success",
            "message": "Calibration completed and saved successfully!",
            "calibration_file": str(robot.calibration_fpath),
            "calibration": {
                motor: {
                    "id": cal.id,
                    "homing_offset": cal.homing_offset,
                    "range_min": cal.range_min,
                    "range_max": cal.range_max
                }
                for motor, cal in calibration.items()
            }
        }
        
    except Exception as e:
        logger.error(f"Error recording range: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/skip_calibration")
async def skip_calibration() -> dict:
    """Skip calibration and create a basic default calibration for testing."""
    if not app_state["robot"]:
        return {"status": "error", "message": "No robot connected"}
    
    try:
        robot = app_state["robot"]
        
        # Create a basic default calibration with safe ranges
        from lerobot.common.motors import MotorCalibration
        calibration = {}
        
        for motor, m in robot.bus.motors.items():
            # Use conservative default ranges
            if motor == "wrist_roll":
                # Full range for continuous rotation
                range_min, range_max = 0, 4095
            elif motor == "gripper":
                # Typical gripper range
                range_min, range_max = 1500, 3000
            else:
                # Conservative joint ranges
                range_min, range_max = 1024, 3072
            
            calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=0,  # No offset for default calibration
                range_min=range_min,
                range_max=range_max,
            )
        
        # Apply the default calibration
        robot.calibration = calibration
        robot.bus.calibration = calibration
        robot._save_calibration()
        
        logger.info("Default calibration created and applied")
        
        return {
            "status": "success",
            "message": "Default calibration created. Robot is ready for basic operation.",
            "warning": "This is a basic calibration. For precise operation, perform proper calibration.",
            "calibration": {
                motor: {
                    "id": cal.id,
                    "homing_offset": cal.homing_offset,
                    "range_min": cal.range_min,
                    "range_max": cal.range_max
                }
                for motor, cal in calibration.items()
            }
        }
        
    except Exception as e:
        logger.error(f"Error creating default calibration: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
