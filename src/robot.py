import asyncio
from src.vision.vision import Vision
from src.emotion.emotion import Emotion


class Robot:
    """Main robot control system integrating vision and emotion systems.

    Manages the coordination between vision and emotion subsystems, handling
    events and their corresponding callbacks. Uses asyncio for concurrent
    operation of vision and emotion systems.

    Attributes:
        emotion (Emotion): Facial expression and animation system
        vision (Vision): Computer vision system for face detection
        event_handlers (dict): Mapping of event names to handler functions

    Events can be registered using the @event decorator, which will automatically
    wire up handlers to both vision and emotion subsystems.
    """

    def __init__(self):
        """Initialize robot with vision and emotion engines."""

        self.emotion = Emotion()
        self.vision = Vision()
        self.event_handlers = {}

    def event(self, event_name):
        def decorator(func):
            self.event_handlers[event_name] = func

            async def wrapper(*args, **kwargs):
                try:
                    await func(*args, **kwargs)
                except Exception as e:
                    print(f"ERROR in {event_name} handler: {e}")

            self.vision.on(event_name, wrapper)
            self.emotion.on(event_name, wrapper)
            return func

        return decorator

    async def run(self):
        """Start the robot's main processing loop.

        Concurrently runs the vision and emotion subsystems using asyncio.
        Emit 'ready' event when initialization is complete.
        """

        await asyncio.gather(self.emotion.run(), self.vision.run())
        self.emit("ready")

    def emit(self, event_name, *args, **kwargs):
        """Emit an event to registered handlers.

        Args:
            event_name (str): Name of the event to emit
            *args: Variable length argument list for the event handler
            **kwargs: Arbitrary keyword arguments for the event handler

        Creates an asyncio task to run the handler if one exists for the event.
        Does nothing if no handler is registered for the event name.
        """

        if event_name in self.event_handlers:
            handler = self.event_handlers[event_name]
            asyncio.create_task(handler(*args, **kwargs))
