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

from dataclasses import dataclass, field
from typing import Dict

from lerobot.robots.config import RobotConfig


@dataclass
class SO100WebClientConfig(RobotConfig):
    """Configuration for SO100 robot controlled via web interface"""
    
    # Robot identification
    id: str = "so100_web"
    type: str = "so100"
    
    # ZMQ communication settings
    zmq_host: str = "localhost"
    zmq_sub_port: int = 5555  # Subscribe to observations from bridge PUB socket
    zmq_pub_port: int = 5556  # Publish actions to bridge SUB socket
    
    # Connection settings
    polling_timeout_ms: int = 100
    connect_timeout_s: float = 5.0
    
    # Motor configuration
    num_motors: int = 6
    
    # Control settings
    max_motor_position: float = 4095.0  # SCS servo range
    min_motor_position: float = 0.0 