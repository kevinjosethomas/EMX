import asyncio
from src.vision.vision import Vision
from src.emotion.emotion import Emotion


class Robot:
    def __init__(self):
        self.emotion = Emotion()
        self.vision = Vision()
        self.event_handlers = {}

    def event(self, event_name):
        def decorator(func):
            self.event_handlers[event_name] = func
            self.vision.on(event_name, func)
            self.emotion.on(event_name, func)
            return func

        return decorator

    async def run(self):
        await asyncio.gather(self.emotion.run(), self.vision.run())
        self.emit("ready")

    def emit(self, event_name, *args, **kwargs):
        print(f"Emitting event: {event_name}")
        if event_name in self.event_handlers:
            handler = self.event_handlers[event_name]
            print(f"Handling event: {event_name}")
            asyncio.create_task(handler(*args, **kwargs))
        else:
            print(f"No handler for event: {event_name}")
