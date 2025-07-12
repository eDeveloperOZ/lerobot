# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SO100 Web Client - Connects to SO100 robot via web interface

This client communicates with a web-based controller (Cubix) through a 
WebSocket-to-ZMQ bridge, allowing LeRobot to control SO100 robots that
are connected via a web browser interface.
"""

import base64
import json
import logging
import time
from functools import cached_property
from typing import Any, Dict, Optional

import cv2
import numpy as np
import zmq

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_so100_web_client import SO100WebClientConfig


class SO100WebClient(Robot):
    """
    SO100 robot client that communicates via ZMQ with a web interface.
    
    The communication flow is:
    LeRobot (this client) <-> ZMQ <-> WebSocket Bridge <-> Web Browser <-> SO100 Hardware
    """
    
    config_class = SO100WebClientConfig
    name = "so100_web_client"
    
    def __init__(self, config: SO100WebClientConfig):
        super().__init__(config)
        self.config = config
        self.id = config.id
        self.robot_type = config.type
        
        # ZMQ configuration
        self.zmq_sub_port = getattr(config, 'zmq_sub_port', 5555)  # Subscribe to observations from bridge
        self.zmq_pub_port = getattr(config, 'zmq_pub_port', 5556)  # Publish actions to bridge
        self.zmq_host = getattr(config, 'zmq_host', 'localhost')
        
        # Connection settings
        self.polling_timeout_ms = getattr(config, 'polling_timeout_ms', 100)
        self.connect_timeout_s = getattr(config, 'connect_timeout_s', 5.0)
        
        # ZMQ sockets
        self.zmq_context = None
        self.zmq_sub_socket = None  # Subscribe to observations from web
        self.zmq_pub_socket = None  # Publish actions to web
        
        # Cache for latest data
        self.last_frames = {}
        self.last_observation = {}
        self.last_observation_time = 0
        
        self._is_connected = False
        self.logs = {}
        
        # Motor configuration
        self.num_motors = 6  # SO100 has 6 motors
        
    @cached_property
    def _state_ft(self) -> dict[str, type]:
        """Define state features for SO100, following convention."""
        return {
            "shoulder_pan.pos": float,
            "shoulder_lift.pos": float,
            "elbow_flex.pos": float,
            "wrist_flex.pos": float,
            "wrist_roll.pos": float,
            "gripper.pos": float,
        }
    
    @cached_property
    def _state_order(self) -> tuple[str, ...]:
        """Order of state features"""
        return tuple(self._state_ft.keys())
    
    @cached_property
    def _cameras_ft(self) -> dict[str, tuple[int, int, int]]:
        """Camera features"""
        # Default camera configuration
        return {
            "front": (480, 640, 3),  # height, width, channels
            "aux": (480, 640, 3),    # Auxiliary camera
        }
    
    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """All observation features"""
        return {**self._state_ft, **self._cameras_ft}
    
    @cached_property
    def action_features(self) -> dict[str, type]:
        """Action features (same as state for position control)"""
        return self._state_ft
    
    @property
    def is_connected(self) -> bool:
        return self._is_connected
    
    @property
    def is_calibrated(self) -> bool:
        # SO100 doesn't require calibration in this setup
        return True
    
    def connect(self) -> None:
        """Connect to the WebSocket bridge via ZMQ"""
        if self._is_connected:
            raise DeviceAlreadyConnectedError(
                "SO100 Web Client is already connected. Do not run `robot.connect()` twice."
            )
        
        logging.info(f"Connecting to WebSocket bridge at {self.zmq_host}")
        
        # Create ZMQ context
        self.zmq_context = zmq.Context()
        
        # Create SUB socket for receiving observations
        self.zmq_sub_socket = self.zmq_context.socket(zmq.SUB)
        self.zmq_sub_socket.connect(f"tcp://{self.zmq_host}:{self.zmq_sub_port}")
        self.zmq_sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.zmq_sub_socket.setsockopt(zmq.CONFLATE, 1)  # Only keep latest message
        
        # Create PUB socket for sending actions
        self.zmq_pub_socket = self.zmq_context.socket(zmq.PUB)
        self.zmq_pub_socket.connect(f"tcp://{self.zmq_host}:{self.zmq_pub_port}")
        
        # Wait for connection to establish
        time.sleep(0.5)
        
        # Check if we can receive observations
        poller = zmq.Poller()
        poller.register(self.zmq_sub_socket, zmq.POLLIN)
        socks = dict(poller.poll(self.connect_timeout_s * 1000))
        
        if self.zmq_sub_socket not in socks or socks[self.zmq_sub_socket] != zmq.POLLIN:
            raise DeviceNotConnectedError(
                f"Timeout waiting for WebSocket bridge to connect. "
                f"Make sure the bridge is running and the web interface is connected."
            )
        
        self._is_connected = True
        logging.info("Successfully connected to SO100 via web interface")
    
    def calibrate(self) -> None:
        """SO100 doesn't require calibration in this setup"""
        pass
    
    def configure(self) -> None:
        """SO100 doesn't require additional configuration in this setup"""
        pass
    
    def _poll_and_get_latest_message(self) -> Optional[str]:
        """Poll ZMQ socket for latest observation"""
        poller = zmq.Poller()
        poller.register(self.zmq_sub_socket, zmq.POLLIN)
        
        try:
            socks = dict(poller.poll(self.polling_timeout_ms))
        except zmq.ZMQError as e:
            logging.error(f"ZMQ polling error: {e}")
            return None
        
        if self.zmq_sub_socket not in socks:
            return None
        
        # Get the latest message (discard older ones)
        last_msg = None
        while True:
            try:
                msg = self.zmq_sub_socket.recv_string(zmq.NOBLOCK)
                last_msg = msg
            except zmq.Again:
                break
        
        return last_msg
    
    def _parse_observation(self, obs_string: str) -> Optional[Dict[str, Any]]:
        """Parse JSON observation from web client"""
        try:
            data = json.loads(obs_string)
            
            # Check message type
            if data.get('type') != 'observation':
                return None
                
            return data
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON observation: {e}")
            return None
    
    def _decode_frame(self, image_b64: str) -> Optional[np.ndarray]:
        """Decode base64 encoded JPEG image"""
        if not image_b64:
            return None
            
        try:
            jpg_data = base64.b64decode(image_b64)
            np_arr = np.frombuffer(jpg_data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is None:
                logging.warning("cv2.imdecode returned None for an image")
                return None
                
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame
            
        except Exception as e:
            logging.error(f"Error decoding base64 image: {e}")
            return None
    
    def get_observation(self) -> dict[str, Any]:
        """Get latest observation from the web interface"""
        if not self._is_connected:
            raise DeviceNotConnectedError(
                "SO100 Web Client is not connected. You need to run `robot.connect()`."
            )
        
        # Poll for latest observation
        latest_msg = self._poll_and_get_latest_message()
        
        if latest_msg:
            observation = self._parse_observation(latest_msg)
            if observation:
                # Update cache
                self.last_observation_time = observation.get('timestamp', time.time())
                
                # Process motor states
                motor_states = observation.get('motor_states', {})
                positions = motor_states.get('positions', [0] * self.num_motors)
                
                # Create observation dict
                obs_dict = {}
                
                # Map received positions array to named joint states
                for i, key in enumerate(self._state_order):
                    obs_dict[key] = float(positions[i]) if i < len(positions) else 0.0
                
                # Create a flat observation.state vector for policies that might need it.
                # Note: `build_dataset_frame` will build this from individual keys if not present.
                state_vec = np.array([obs_dict[key] for key in self._state_order], dtype=np.float32)
                obs_dict["observation.state"] = state_vec
                
                # Process camera frames
                frames = observation.get('frames', {})
                for cam_name, frame_b64 in frames.items():
                    frame = self._decode_frame(frame_b64)
                    if frame is not None:
                        obs_dict[cam_name] = frame
                        self.last_frames[cam_name] = frame
                
                self.last_observation = obs_dict
                return obs_dict
        
        # Return cached observation if no new data
        if self.last_observation:
            return self.last_observation
        
        # Return default observation if nothing cached
        default_obs = {
            "observation.state": np.zeros(self.num_motors, dtype=np.float32)
        }
        for i in range(self.num_motors):
            default_obs[f"motor_{i}.pos"] = 0.0
        
        # Add blank frames for cameras
        for cam_name in self._cameras_ft:
            h, w, c = self._cameras_ft[cam_name]
            default_obs[cam_name] = np.zeros((h, w, c), dtype=np.uint8)
            
        return default_obs
    
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Send action to the web interface"""
        if not self._is_connected:
            raise DeviceNotConnectedError(
                "SO100 Web Client is not connected. You need to run `robot.connect()`."
            )
        
        # Extract motor positions from action dict, respecting the order
        positions = [0.0] * self.num_motors
        for i, key in enumerate(self._state_order):
            if key in action:
                positions[i] = float(action[key])
            elif "action" in action and isinstance(action["action"], (list, np.ndarray)) and i < len(action["action"]):
                # Fallback to using a flat action array if available
                positions[i] = float(action["action"][i])

        # Create action message
        action_msg = {
            "type": "action",
            "timestamp": time.time(),
            "goal_positions": positions
        }
        
        # Send via ZMQ
        try:
            self.zmq_pub_socket.send_string(json.dumps(action_msg))
        except Exception as e:
            logging.error(f"Failed to send action: {e}")
        
        # Return the action that was sent
        action_sent = {key: positions[i] for i, key in enumerate(self._state_order)}
        action_sent["action"] = np.array(positions, dtype=np.float32)
        
        return action_sent
    
    def disconnect(self):
        """Disconnect from the WebSocket bridge"""
        if not self._is_connected:
            raise DeviceNotConnectedError(
                "SO100 Web Client is not connected."
            )
        
        logging.info("Disconnecting from WebSocket bridge")
        
        if self.zmq_sub_socket:
            self.zmq_sub_socket.close()
        if self.zmq_pub_socket:
            self.zmq_pub_socket.close()
        if self.zmq_context:
            self.zmq_context.term()
            
        self._is_connected = False
        logging.info("Disconnected from SO100 web interface") 