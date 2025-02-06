import time
import random
import asyncio
from src.voice import Voice
from src.vision import Vision
from src.emotion import Emotion
from src.emotion.expressions import Angry, Love, Scared, Happy, Sad, Neutral


class Robot:
    """Main robot control system integrating vision, emotion, and voice systems.

    Manages the coordination between vision, emotion, and voice subsystems, handling
    events and their corresponding callbacks. Uses asyncio for concurrent
    operation of vision, emotion, and voice systems.

    Attributes:
        emotion (Emotion): Facial expression and animation system
        vision (Vision): Computer vision system for face detection
        voice (Voice): Voice engine with OpenAI TTS and emotion analysis
        event_handlers (dict): Mapping of event names to handler functions

    Events can be registered using the @event decorator, which will automatically
    wire up handlers to vision, emotion, and voice subsystems.
    """

    def __init__(self, openai_api_key):
        """Initialize robot with vision and emotion engines."""

        self.emotion = Emotion()
        self.vision = Vision()
        self.voice = Voice(
            openai_api_key=openai_api_key,
        )
        self.event_handlers = {}

        self.is_idle = False
        self.last_activity_time = time.time()
        self.idle_timeout = 5
        self.idle_check_task = None

        self.voice.on("_assistant_message", self._handle_voice_emotion)
        self.voice.on(
            "_assistant_message_end",
            lambda: asyncio.create_task(
                self.emotion.queue_animation(Neutral(sticky=True))
            ),
        )

    async def _check_idle_state(self):
        """Background task to check and update idle state."""

        while True:
            current_time = time.time()
            if (
                not self.is_idle
                and (current_time - self.last_activity_time)
                >= self.idle_timeout
            ):
                await self.start_idle()
            await asyncio.sleep(1)

    async def _handle_activity(self):
        """Handle any activity that should reset the idle timer."""

        self.last_activity_time = time.time()
        if self.is_idle:
            await self.stop_idle()

    async def start_idle(self):
        """Start idle behavior"""

        self.is_idle = True
        self.emotion.idle_manager.running = True
        self.emit("idle_started")

    async def stop_idle(self):
        """Stop idle behavio."""

        self.is_idle = False
        self.emotion.idle_manager.running = False
        self.emit("idle_stopped")

    async def _handle_voice_emotion(self, data):
        """Handle emotion data from voice system and queue corresponding animation.

        Takes emotion analysis results from the emotion2vec+ model and maps them
        to appropriate facial expressions. Expressions are scaled and positioned
        randomly for natural variation.

        Args:
            data (dict): Emotion analysis data containing:
                - emotion (str): Detected emotion category
                - duration (float): Audio segment duration in seconds
        """

        await self._handle_activity()

        print(emotion)

        emotion = data["emotion"]
        duration = data["duration"]

        random_scale = random.uniform(0.97, 1.03)
        random_position = (
            random.uniform(-0.04, 0.04),
            random.uniform(-0.04, 0.04),
        )
        # random_scale = 1
        # random_position = (0, 0)
        transition_duration = 0.3

        if emotion == "happiness":
            await self.emotion.queue_animation(
                Happy(
                    scale=random_scale,
                    position=random_position,
                    duration=duration,
                    transition_duration=transition_duration,
                ),
            )
        elif emotion == "surprised":
            await self.emotion.queue_animation(
                Love(
                    scale=random_scale,
                    position=random_position,
                    duration=duration,
                    transition_duration=transition_duration,
                ),
            )
        elif emotion == "fear":
            await self.emotion.queue_animation(
                Scared(
                    scale=random_scale,
                    position=random_position,
                    duration=duration,
                    transition_duration=transition_duration,
                ),
            )
        elif emotion == "sadness":
            await self.emotion.queue_animation(
                Sad(
                    scale=random_scale,
                    position=random_position,
                    duration=duration,
                    transition_duration=transition_duration,
                ),
            )
        elif emotion == "anger":
            await self.emotion.queue_animation(
                Angry(
                    scale=random_scale,
                    position=random_position,
                    duration=duration,
                    transition_duration=transition_duration,
                ),
            )
        else:
            await self.emotion.queue_animation(
                Neutral(
                    scale=random_scale,
                    position=random_position,
                    duration=duration,
                    transition_duration=transition_duration,
                ),
            )

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

        self.idle_check_task = asyncio.create_task(self._check_idle_state())

        self.emit("ready")
        await asyncio.gather(
            self.emotion.run(),
            self.vision.run(),
            self.voice.run(),
            self.idle_check_task,
        )

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
