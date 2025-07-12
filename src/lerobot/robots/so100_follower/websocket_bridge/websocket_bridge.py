"""
WebSocket to ZMQ Bridge Server

This server bridges WebSocket connections from the browser
to ZMQ connections for the SO100_client.
"""

import asyncio
import websockets
import zmq
import zmq.asyncio
import json
import logging
import argparse
from typing import Set, Dict, Any, Optional
import signal
import sys
import time
import numpy as np
import base64
import cv2
from pathlib import Path
from dataclasses import dataclass, field

from .message_protocol import parse_message, StatusMessage, ConnectionState

# LeRobot imports
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.utils import build_dataset_frame
from lerobot.policies.factory import make_policy, make_policy_config
from lerobot.robots.so100_follower.config_so100_web_client import SO100WebClientConfig
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device, init_logging
from lerobot.envs.configs import EnvConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from huggingface_hub import hf_hub_download


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WebSocketBridge:
    """
    Bridge between WebSocket clients and inference
    """
    
    def __init__(
        self,
        ws_port: int = 8765,
        inference_fps: int = 10,
        device: str = "mps",
        no_amp: bool = False,
        chunk_size: int = 1,
    ):
        """
        Initialize the bridge server.
        
        Args:
            ws_port: WebSocket server port
            inference_fps: Inference frequency in Hz
            device: Device to run inference on
            no_amp: Whether to disable automatic mixed precision
            chunk_size: Number of actions to predict in each inference (1 for single, >1 for chunking)
        """
        self.ws_port = ws_port
        self.inference_fps = inference_fps
        self.device = device
        self.no_amp = no_amp
        self.chunk_size = chunk_size
        
        # WebSocket clients
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        
        # Running state
        self.running = False

        # Last inference time for rate limiting
        self.last_inference_time: Dict[websockets.WebSocketServerProtocol, float] = {}
        self.camera_mappings: Dict[websockets.WebSocketServerProtocol, Dict[str, str]] = {}
        
        # Inference components
        self.policy = None
        self.policy_path = None
        
        # Robot configuration
        self.num_motors = 6
        self.dataset_features: dict = {}
        
        # Latest observation cache
        self.latest_observation = None
        self.observation_lock = asyncio.Lock()
        
        # Root directory for mock datasets
        self.root = Path.home() / ".cache" / "lerobot"
        
        logger.info(f"WebSocket bridge initialized - WS:{ws_port}")
    
    async def start(self):
        """Start the bridge server"""
        # Start WebSocket server
        self.running = True
        async with websockets.serve(
            self._handle_client,
            "localhost",
            self.ws_port
        ):
            logger.info(f"WebSocket server started on port {self.ws_port}")
            
            try:
                await asyncio.Future()  # Run forever
            except asyncio.CancelledError:
                pass
            finally:
                self.running = False
                await self._cleanup()
    
    async def _cleanup(self):
        """Clean up resources"""
        # Close all WebSocket connections
        if self.clients:
            await asyncio.gather(
                *[client.close() for client in self.clients],
                return_exceptions=True
            )
        
        logger.info("Bridge server cleaned up")
    
    async def _handle_client(self, websocket):
        """Handle a WebSocket client connection"""
        self.clients.add(websocket)
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        self.last_inference_time[websocket] = 0.0
        self.camera_mappings[websocket] = {}

        logger.info(f"Client connected: {client_id}")
        
        try:
            async for message in websocket:
                await self._handle_ws_message(message, websocket)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {client_id}")
        except Exception as e:
            logger.error(f"Client error {client_id}: {e}")
        finally:
            self.clients.remove(websocket)
            if websocket in self.last_inference_time:
                del self.last_inference_time[websocket]
            if websocket in self.camera_mappings:
                del self.camera_mappings[websocket]
    
    async def _handle_ws_message(self, message: str, websocket):
        """Handle incoming WebSocket message"""
        try:
            data = json.loads(message)
            msg_type = data.get('type')
            
            if msg_type == 'init':
                # Handle initialization with policy path
                await self._handle_init_message(data, websocket)
                
            elif msg_type == 'observation':
                # Apply camera mapping to the frames
                camera_mapping = self.camera_mappings.get(websocket, {})
                if 'frames' in data and camera_mapping:
                    original_frames = data['frames']
                    data['frames'] = {
                        camera_mapping.get(key, key): value
                        for key, value in original_frames.items()
                    }

                # Rate limit inference to inference_fps
                now = time.time()
                if (now - self.last_inference_time.get(websocket, 0)) < (1 / self.inference_fps):
                    return  # Skip frame
                
                self.last_inference_time[websocket] = now

                # Store latest observation for inference
                async with self.observation_lock:
                    self.latest_observation = data
                
                # Run inference if policy is loaded
                if self.policy:
                    observation_frame = await self._get_observation_dict(websocket)
                    if observation_frame:
                        await self._run_inference(observation_frame, websocket)

            elif msg_type == 'heartbeat':
                # Echo heartbeat back
                await websocket.send(json.dumps({'type': 'heartbeat'}))
                
            else:
                logger.warning(f"Unknown message type: {msg_type}")
                
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON: {e}")
            await websocket.send(json.dumps({
                'type': 'error',
                'message': 'Invalid JSON format'
            }))
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def _handle_init_message(self, data: dict, websocket):
        """Handle initialization message with policy configuration"""
        config = data.get('config', {})
        policy_path = config.get('policyPath')
        camera_mapping = config.get('camera_mapping')
        
        if camera_mapping:
            self.camera_mappings[websocket] = camera_mapping
            logger.info(f"Received camera mapping: {camera_mapping}")
        else:
            self.camera_mappings[websocket] = {} # reset
        
        if not policy_path:
            await websocket.send(json.dumps({
                'type': 'error',
                'message': 'No policy path provided in init message'
            }))
            return
        
        logger.info(f"Received init message with policy path: {policy_path}")
        
        # Load the policy but don't start any loops
        try:
            await self._load_policy(policy_path)
            
            # Send success status
            await websocket.send(json.dumps({
                'type': 'status',
                'state': 'connected',
                'message': f'Policy loaded and ready: {policy_path}'
            }))
        except Exception as e:
            logger.error(f"Failed to load policy: {e}")
            await websocket.send(json.dumps({
                'type': 'error',
                'message': f'Failed to load policy: {str(e)}'
            }))
    
    async def _load_policy(self, policy_path: str, device_str="mps"):
        """Load policy model with proper configuration"""
        logger.info(f"Loading policy from: {policy_path}")
        
        self.policy_path = policy_path
        device = get_safe_torch_device(device_str)

        try:
            policy_config = PreTrainedConfig.from_pretrained(self.policy_path)
            policy_config.device = str(device)
        except Exception as e:
            logger.warning(f"Custom config loading failed, trying default method: {e}")
            # Try loading as a simple dict and add type field if missing
            config_path = hf_hub_download(repo_id=self.policy_path, filename="config.json")
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            # Infer policy type from config structure
            if "use_vae" in config_dict and "vision_backbone" in config_dict:
                config_dict["type"] = "act"
            elif "diffusion_step_embed_dim" in config_dict:
                config_dict["type"] = "diffusion"
            else:
                raise ValueError("Cannot infer policy type from config")

            policy_config = make_policy_config(**config_dict)
            policy_config.device = str(device)

        # Ensure pretrained_path is set
        policy_config.pretrained_path = self.policy_path
        
        # --- DYNAMIC ENV CONFIG BUILDING ---
        # Try to extract features from policy config or associated dataset config
        features = {}
        features_map = {}
        num_actions = None
        primary_camera = None
        fps = getattr(policy_config, 'fps', 10)
        task = getattr(policy_config, 'task', 'so100_websocket')

        # Try to get input/output features from policy config
        input_features = getattr(policy_config, 'input_features', None)
        output_features = getattr(policy_config, 'output_features', None)
        dataset_repo_id = getattr(policy_config, 'dataset_repo_id', None)
        dataset_config = None

        # If dataset_repo_id is present, try to load dataset config
        if dataset_repo_id:
            try:
                dataset_config_path = hf_hub_download(repo_id=dataset_repo_id, filename="config.json")
                with open(dataset_config_path, 'r') as f:
                    dataset_config = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load dataset config: {e}. Proceeding with policy config only.")
        
        # Prefer dataset config features if available, else use policy config
        if dataset_config and 'features' in dataset_config:
            features_dict = dataset_config['features']
        elif input_features and output_features:
            features_dict = {**input_features, **output_features}
        else:
            raise ValueError("No features found in either dataset or policy config.")

        # Collect all camera keys found in features
        all_camera_keys = []
        
        # Build features and features_map
        for key, ft in features_dict.items():
            # Handle both dict and PolicyFeature
            if isinstance(ft, dict):
                dtype = ft.get('type', None)
                shape = tuple(ft.get('shape', []))
            elif isinstance(ft, PolicyFeature):
                dtype = ft.type
                shape = ft.shape
            else:
                logger.warning(f"Feature '{key}' is neither dict nor PolicyFeature: {ft}")
                continue

            # Heuristic for FeatureType
            if dtype == 'ACTION' or dtype == FeatureType.ACTION or key == 'action' or key.startswith('action'):
                ftype = FeatureType.ACTION
                num_actions = shape[0] if shape else None
                features[key] = PolicyFeature(type=ftype, shape=shape)
                features_map[key] = 'action'
            elif dtype == 'STATE' or dtype == FeatureType.STATE or key == 'observation.state' or key.endswith('.state'):
                ftype = FeatureType.STATE
                features[key] = PolicyFeature(type=ftype, shape=shape)
                features_map[key] = 'observation.state'
            elif dtype == 'VISUAL' or dtype == FeatureType.VISUAL or key.startswith('observation.images.'):
                ftype = FeatureType.VISUAL
                features[key] = PolicyFeature(type=ftype, shape=shape)
                features_map[key] = key  # Use as-is for images
                
                # Extract camera name and add to list
                parts = key.split('.')
                if len(parts) >= 3:
                    camera_name = parts[-1]
                    all_camera_keys.append(camera_name)
                    if not primary_camera:
                        primary_camera = camera_name
            elif dtype == 'ENV' or dtype == FeatureType.ENV:
                ftype = FeatureType.ENV
                features[key] = PolicyFeature(type=ftype, shape=shape)
                features_map[key] = key
            else:
                # Unknown type, skip or raise error
                logger.warning(f"Unknown or unsupported feature type for key '{key}': {ft}")
                continue

        if not num_actions:
            raise ValueError("Could not determine number of actions from config features.")
        if not primary_camera:
            # Fallback to 'webcam' if no camera found
            primary_camera = 'webcam'
            all_camera_keys = [primary_camera]

        logger.info(f"Found camera keys in policy: {all_camera_keys}")

        # Ensure task and fps are set before DynamicEnvConfig
        task_val = getattr(policy_config, 'task', None)
        if not task_val:
            # Try to get from dataset_config
            if dataset_config and 'task' in dataset_config:
                task_val = dataset_config['task']
            else:
                task_val = 'so100_websocket'
        fps_val = getattr(policy_config, 'fps', None)
        if not fps_val:
            if dataset_config and 'fps' in dataset_config:
                fps_val = dataset_config['fps']
            else:
                fps_val = 10

        # Dynamically create EnvConfig
        @dataclass
        class DynamicEnvConfig(EnvConfig):
            task: str = task_val
            fps: int = fps_val
            features: dict[str, PolicyFeature] = field(default_factory=lambda: features)
            features_map: dict[str, str] = field(default_factory=lambda: features_map)
            @property
            def gym_kwargs(self) -> dict:
                return {}
        
        env_cfg = DynamicEnvConfig()
        self.num_motors = num_actions
        self.primary_camera = primary_camera
        
        # Dynamically build dataset_features for build_dataset_frame - INCLUDE ALL CAMERAS
        self.dataset_features = {}
        
        # Add all camera features
        for camera_key in all_camera_keys:
            image_feature_key = f"observation.images.{camera_key}"
            if image_feature_key in env_cfg.features:
                # The shape in the policy config is channel-first (C, H, W). We need raw shape (H, W, C).
                policy_image_shape = env_cfg.features[image_feature_key].shape
                raw_image_shape = (policy_image_shape[1], policy_image_shape[2], policy_image_shape[0])
                
                self.dataset_features[image_feature_key] = {
                    "dtype": "image",
                    "shape": raw_image_shape,
                    "names": ["height", "width", "channels"],
                }
        
        # Add state features
        self.dataset_features["observation.state"] = {
            "dtype": "float32",
            "shape": (self.num_motors,),
            "names": [f"motor_{i}.pos" for i in range(self.num_motors)],
        }
        
        # Set chunk_size if supported
        if self.chunk_size > 1 and hasattr(policy_config, 'chunk_size'):
            policy_config.chunk_size = self.chunk_size
            logger.info(f"Set policy chunk_size to {self.chunk_size}")

        # Make policy
        self.policy = make_policy(policy_config, env_cfg=env_cfg)
        self.policy.reset()

        logger.info(f"Dataset features: {list(self.dataset_features.keys())}")
        logger.info(f"Policy loaded. Primary camera key: {self.primary_camera}")
        
        return self.policy
    
    def _decode_image(self, image_b64: str) -> Optional[np.ndarray]:
        """Decode base64 image"""
        try:
            # Pad the base64 string if necessary
            missing_padding = len(image_b64) % 4
            if missing_padding:
                image_b64 += '=' * (4 - missing_padding)
            
            img_bytes = base64.b64decode(image_b64)
            img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            
            if img is None:
                logger.warning("Failed to decode image, it might be corrupted.")
                return None
                
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.error(f"Error decoding image: {e}")
            return None
    
    async def _get_observation_dict(self, websocket: websockets.WebSocketServerProtocol) -> dict:
        """Get observation dictionary from latest websocket data"""
        async with self.observation_lock:
            observation = self.latest_observation
            
        if not observation:
            return {}

        # This function should return the raw data, not the built frame.
        # The building happens in _run_inference
        return observation
    
    async def _run_inference(self, observation_data: dict, websocket):
        """Run a single inference pass and send back the action."""
        # Apply camera mapping to the frames
        camera_mapping = self.camera_mappings.get(websocket, {})
        if 'frames' in observation_data and camera_mapping:
            original_frames = observation_data['frames']
            logger.info(f"Original frames received: {list(original_frames.keys())}")
            observation_data['frames'] = {
                camera_mapping.get(key, key): value
                for key, value in original_frames.items()
            }
            logger.info(f"Remapped frames: {list(observation_data['frames'].keys())}")

        # Now, build the flat obs_dict that build_dataset_frame expects
        obs_dict = {}

        # Decode all image data and add to obs_dict
        if 'frames' in observation_data:
            for key, image_b64 in observation_data['frames'].items():
                decoded_frame = self._decode_image(image_b64)
                if decoded_frame is not None:
                    obs_dict[key] = decoded_frame
                    logger.info(f"Successfully decoded frame for key: {key}")
                else:
                    logger.warning(f"Failed to decode frame for key: {key}")

        logger.info(f"Final obs_dict keys: {list(obs_dict.keys())}")

        # Get motor states and add to obs_dict
        motor_positions = observation_data.get('motor_states', {}).get('positions', [0] * self.num_motors)
        
        if "observation.state" not in self.dataset_features:
            logger.error("`observation.state` not in dataset_features. Cannot process motor states.")
            return 

        joint_names = self.dataset_features["observation.state"]["names"]

        for i in range(self.num_motors):
            raw_pos = float(motor_positions[i]) if i < len(motor_positions) else 2048.0
            normalized_pos = (raw_pos - 2048.0) / 2047.0
            normalized_pos = max(-1.0, min(1.0, normalized_pos))
            key = joint_names[i] if i < len(joint_names) else f"motor_{i}.pos"
            obs_dict[key] = normalized_pos

        device = get_safe_torch_device(self.device)
        use_amp = not self.no_amp
        
        try:
            # Build dataset frame for policy
            observation_frame = build_dataset_frame(
                self.dataset_features,
                obs_dict,
                prefix="observation"
            )
            
            logger.info(f"Built observation frame keys: {list(observation_frame.keys())}")
            
            # Run inference
            action_tensor = predict_action(
                observation_frame,
                self.policy,
                device,
                use_amp,
                task=None,  # Task is embedded in policy if needed
                robot_type="so100"  # Or make dynamic if needed
            )
            
            # Handle chunking and convert to servo positions
            if self.chunk_size > 1:
                # Assume action_tensor is (chunk_size, action_dim)
                chunk_positions = []
                for t in range(action_tensor.shape[0]):
                    positions = []
                    for i in range(self.num_motors):
                        value = action_tensor[t, i].item()
                        servo_pos = 2048 + value * 2047
                        servo_pos = max(0, min(4095, servo_pos))
                        positions.append(servo_pos)
                    chunk_positions.append(positions)
                
                action_msg = {
                    "type": "action",
                    "timestamp": time.time(),
                    "goal_positions": chunk_positions  # list of lists
                }
            else:
                # Single action - convert normalized values back to servo positions
                positions = []
                for i in range(self.num_motors):
                    value = action_tensor[i].item()
                    servo_pos = 2048 + value * 2047
                    servo_pos = max(0, min(4095, servo_pos))
                    positions.append(servo_pos)
                
                action_msg = {
                    "type": "action",
                    "timestamp": time.time(),
                    "goal_positions": positions  # single list
                }
            
            # Send to the requesting client
            await websocket.send(json.dumps(action_msg))
            
        except Exception as e:
            logger.error(f"Error in inference task: {e}")
            import traceback
            traceback.print_exc()
            await websocket.send(json.dumps({
                'type': 'error',
                'message': f'Inference error: {str(e)}'
            }))
    
    def stop(self):
        """Stop the bridge server"""
        logger.info("Stopping bridge server...")
        self.running = False


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="WebSocket Bridge Server with Integrated Inference")
    
    parser.add_argument(
        "--ws-port",
        type=int,
        default=8765,
        help="WebSocket server port"
    )
    
    parser.add_argument(
        "--inference-fps",
        type=int,
        default=10,
        help="Inference frequency in Hz"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        choices=["cuda", "cpu", "mps"],
        help="Device to run inference on"
    )
    
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1,
        help="Number of actions to predict in each inference (1 for single, >1 for chunking)"
    )
    
    args = parser.parse_args()
    
    # Initialize logging
    init_logging()
    
    # Create bridge
    bridge = WebSocketBridge(
        ws_port=args.ws_port,
        inference_fps=args.inference_fps,
        device=args.device,
        no_amp=args.no_amp,
        chunk_size=args.chunk_size,
    )
    
    # Setup signal handlers
    loop = asyncio.get_event_loop()
    
    def signal_handler():
        logger.info("Received interrupt signal")
        bridge.stop()
        loop.stop()
    
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)
    
    # Run bridge
    try:
        await bridge.start()
    except KeyboardInterrupt:
        logger.info("Bridge interrupted by user")
    except Exception as e:
        logger.error(f"Bridge error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 