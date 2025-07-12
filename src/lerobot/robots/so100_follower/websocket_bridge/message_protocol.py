"""
ZMQ Message Protocol for SO100 Robot Communication

This module defines the message format for communication between
the web client (Cubix) and the SO100_client (LeRobot middleware).
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import json
import time
from enum import Enum


class MessageType(Enum):
    """Types of messages exchanged between client and server"""
    # Client -> Server
    OBSERVATION = "observation"
    CONNECTION_STATUS = "connection_status"
    HEARTBEAT = "heartbeat"
    
    # Server -> Client  
    ACTION = "action"
    STATUS = "status"
    ERROR = "error"


class ConnectionState(Enum):
    """Robot connection states"""
    IDLE = "idle"
    READY = "ready"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    LOADING = "loading"
    ERROR = "error"


@dataclass
class ObservationMessage:
    """Message sent from client to server containing robot observations"""
    type: str = MessageType.OBSERVATION.value
    timestamp: float = None
    frames: Dict[str, str] = None  # camera_name -> base64_encoded_jpeg
    motor_states: Dict[str, List[float]] = None  # positions, velocities, etc.
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.frames is None:
            self.frames = {}
        if self.motor_states is None:
            self.motor_states = {"positions": [], "velocities": []}
    
    def to_json(self) -> str:
        return json.dumps(asdict(self))
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ObservationMessage':
        data = json.loads(json_str)
        return cls(**data)


@dataclass
class ActionMessage:
    """Message sent from server to client containing robot actions"""
    type: str = MessageType.ACTION.value
    timestamp: float = None
    goal_positions: List[float] = None
    action_id: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.goal_positions is None:
            self.goal_positions = []
    
    def to_json(self) -> str:
        return json.dumps(asdict(self))
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ActionMessage':
        data = json.loads(json_str)
        return cls(**data)


@dataclass
class StatusMessage:
    """Status update messages"""
    type: str = MessageType.STATUS.value
    timestamp: float = None
    state: str = ConnectionState.IDLE.value
    message: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def to_json(self) -> str:
        return json.dumps(asdict(self))
    
    @classmethod
    def from_json(cls, json_str: str) -> 'StatusMessage':
        data = json.loads(json_str)
        return cls(**data)


def parse_message(json_str: str) -> Any:
    """Parse a JSON message and return the appropriate message object"""
    data = json.loads(json_str)
    msg_type = data.get('type')
    
    if msg_type == MessageType.OBSERVATION.value:
        return ObservationMessage.from_json(json_str)
    elif msg_type == MessageType.ACTION.value:
        return ActionMessage.from_json(json_str)
    elif msg_type == MessageType.STATUS.value:
        return StatusMessage.from_json(json_str)
    else:
        raise ValueError(f"Unknown message type: {msg_type}") 