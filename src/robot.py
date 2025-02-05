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
        voice (Voice): Voice engine for Hume AI text-to-speech
        event_handlers (dict): Mapping of event names to handler functions

    Events can be registered using the @event decorator, which will automatically
    wire up handlers to vision, emotion, and voice subsystems.
    """

    def __init__(self, voice_api_key, voice_secret_key, voice_config_id):
        """Initialize robot with vision and emotion engines."""

        self.emotion = Emotion()
        self.vision = Vision()
        self.voice = Voice(
            api_key=voice_api_key,
            secret_key=voice_secret_key,
            config_id=voice_config_id,
        )
        self.event_handlers = {}

        self.voice.on("_assistant_message", self._handle_voice_emotion)
        self.voice.on(
            "_assistant_message_end",
            lambda: asyncio.create_task(
                self.emotion.queue_animation(Neutral(sticky=True))
            ),
        )

    async def _handle_voice_emotion(self, emotion_scores: dict):
        """Handle emotion data from voice system and queue corresponding animation.

        Args:
            emotion (dict): Emotion data from voice system with scores for different emotions
        """

        emotion = max(emotion_scores, key=emotion_scores.get)
        print(emotion)

        if emotion == "happiness":
            await self.emotion.queue_animation(Happy(sticky=True))
        elif emotion == "love" or emotion == "desire":
            await self.emotion.queue_animation(Love(sticky=True))
        elif emotion == "fear":
            await self.emotion.queue_animation(Scared(sticky=True))
        elif emotion == "sadness":
            await self.emotion.queue_animation(Sad(sticky=True))
        elif emotion == "anger":
            await self.emotion.queue_animation(Angry(sticky=True))
        else:
            print("beep boop")
            await self.emotion.queue_animation(Neutral(sticky=True))

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

        self.emit("ready")
        await asyncio.gather(
            self.emotion.run(), self.vision.run(), self.voice.run()
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
