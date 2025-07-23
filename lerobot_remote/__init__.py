"""Dynamic import helper based on ``LEROBOT_MODE``."""
from __future__ import annotations

import os

mode = os.environ.get("LEROBOT_MODE", "cloud")

if mode == "cloud":
    from lerobot_core_cloud import *  # noqa: F401,F403
elif mode == "edge":
    from lerobot_edge import *  # noqa: F401,F403
else:
    raise ValueError(f"Invalid LEROBOT_MODE: {mode}")

__all__ = ["mode"]
