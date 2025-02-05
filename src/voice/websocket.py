import base64
from hume import Stream
from hume.core.api_error import ApiError
from pyee.asyncio import AsyncIOEventEmitter
from hume.empathic_voice.chat.types import SubscribeEvent
from hume.empathic_voice.chat.socket_client import ChatWebsocketConnection

EMOTION_MAPPING = {
    # Happiness category
    "amusement": "happiness",
    "aesthetic_appreciation": "happiness",
    "contentment": "happiness",
    "ecstasy": "happiness",
    "excitement": "happiness",
    "interest": "happiness",
    "joy": "happiness",
    "pride": "happiness",
    "relief": "happiness",
    "satisfaction": "happiness",
    "triumph": "happiness",
    "realization": "happiness",
    "surprise_positive": "happiness",
    # Love category
    "admiration": "love",
    "adoration": "love",
    "awe": "love",
    "love": "love",
    "romance": "love",
    "sympathy": "love",
    # Fear category
    "fear": "fear",
    "horror": "fear",
    "surprise_negative": "fear",
    # Sadness category
    "disappointment": "sadness",
    "distress": "sadness",
    "guilt": "sadness",
    "nostalgia": "sadness",
    "sadness": "sadness",
    "shame": "sadness",
    "tiredness": "sadness",
    "empathic_pain": "sadness",
    "pain": "sadness",
    # Anger category
    "anger": "anger",
    "contempt": "anger",
    "disgust": "anger",
    "envy": "anger",
    # Discomfort category
    "doubt": "discomfort",
    "anxiety": "discomfort",
    "awkwardness": "discomfort",
    "embarrassment": "discomfort",
    "confusion": "discomfort",
    "boredom": "discomfort",
    # Concentration category
    "calmness": "concentration",
    "concentration": "concentration",
    "contemplation": "concentration",
    # Desire category
    "craving": "desire",
    "desire": "desire",
}


class WebSocketHandler(AsyncIOEventEmitter):
    """Handles WebSocket communication for Hume AI's Empathic Voice Interface.

    Manages bi-directional audio streaming and emotion inference processing through
    WebSocket connection. Processes incoming messages and emits relevant events.

    Events:
        _assistant_message: Emitted when assistant speaks with emotion scores
            - happiness (float): Score for happy emotions
            - love (float): Score for loving emotions
            - fear (float): Score for fearful emotions
            - sadness (float): Score for sad emotions
            - anger (float): Score for angry emotions
            - discomfort (float): Score for uncomfortable emotions
            - concentration (float): Score for focused emotions
            - desire (float): Score for desire emotions

        _assistant_message_end: Emitted when assistant finishes speaking

    Attributes:
        socket (ChatWebsocketConnection): Active WebSocket connection to Hume AI
        byte_strs (Stream): Stream of audio bytes for playback
    """

    def __init__(self):
        """Initialize WebSocket handler with empty socket and new byte stream."""

        super().__init__()
        self.socket = None
        self.byte_strs = Stream.new()

    def set_socket(self, socket: ChatWebsocketConnection):
        """Set the active WebSocket connection.

        Args:
            socket (ChatWebsocketConnection): WebSocket connection to Hume AI API
        """

        self.socket = socket

    async def on_open(self):
        """Handle WebSocket connection opened event."""

        print("Hume AI WebSocket connection opened")

    async def on_message(self, message: SubscribeEvent):
        """Process incoming WebSocket messages.

        Handles different message types including:
        - Chat metadata with session IDs
        - Assistant messages with emotion inference
        - Audio output data for playback
        - Error messages from API

        Emits events for emotion scores and end of assistant messages.
        Processes audio data into playable bytes.

        Args:
            message (SubscribeEvent): Incoming WebSocket message to process

        Raises:
            ApiError: If an error message is received from the API
        """

        if message.type == "chat_metadata":
            chat_id = message.chat_id
            chat_group_id = message.chat_group_id

        elif message.type == "assistant_message":
            scores = message.models.prosody.scores

            # Find the emotion with the highest score
            max_score = 0
            dominant_emotion = None

            for emotion, score in scores.__dict__.items():
                if score > max_score:
                    max_score = score
                    dominant_emotion = EMOTION_MAPPING.get(emotion)

            self.emit("_assistant_message", dominant_emotion)

        elif message.type == "assistant_end":
            self.emit("_assistant_message_end")

        elif message.type == "audio_output":
            message_str: str = message.data
            message_bytes = base64.b64decode(message_str.encode("utf-8"))
            await self.byte_strs.put(message_bytes)
            return

        elif message.type == "error":
            error_message = message.message
            error_code = message.code
            raise ApiError(f"Error ({error_code}): {error_message}")

    async def on_close(self):
        """Logic invoked when the WebSocket connection is closed."""

        print("WebSocket connection closed.")

    async def on_error(self, error):
        """Logic invoked when an error occurs in the WebSocket connection."""

        print(f"Error: {error}")
