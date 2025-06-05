import secrets
from typing import Dict

tokens: Dict[str, str] = {}

def create_token(device: str) -> str:
    """Create and store a random token for *device*."""
    token = secrets.token_urlsafe(16)
    tokens[token] = device
    return token

def verify_token(token: str) -> bool:
    """Return ``True`` if *token* is valid."""
    return token in tokens
