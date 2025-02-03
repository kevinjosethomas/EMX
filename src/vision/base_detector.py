from abc import ABC, abstractmethod
from pyee.asyncio import AsyncIOEventEmitter


class BaseDetector(ABC, AsyncIOEventEmitter):
    """Base class for all detection modules.

    Provides common functionality and interface that all detectors must implement.
    """

    def __init__(self):
        super().__init__()
        self.is_running = False

    @abstractmethod
    async def process_frame(self, frame):
        """Process a single frame and emit relevant events.

        Args:
            frame: CV2 image frame to process
        """
        pass

    @abstractmethod
    async def setup(self):
        """Initialize any required models or resources."""
        pass

    @abstractmethod
    async def cleanup(self):
        """Clean up resources when detector is stopped."""
        pass
