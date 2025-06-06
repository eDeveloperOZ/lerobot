import secrets
from typing import Dict

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# Simple in-memory token store
_tokens: Dict[str, str] = {}
_scheme = HTTPBearer(auto_error=False)


def create_token(device: str) -> str:
    token = secrets.token_urlsafe(16)
    _tokens[token] = device
    return token


def require_token(credentials: HTTPAuthorizationCredentials = Depends(_scheme)) -> str:
    if not credentials or credentials.credentials not in _tokens:
        raise HTTPException(status_code=401, detail="invalid token")
    return credentials.credentials
