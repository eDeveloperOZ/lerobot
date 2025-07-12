import asyncio
import sys
import logging

from lerobot.robots.so100_follower.websocket_bridge.websocket_bridge import main

logger = logging.getLogger(__name__)



if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bridge interrupted by user")
    except Exception as e:
        logger.error(f"Bridge error: {e}")
        sys.exit(1)