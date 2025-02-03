import base64
from hume import Stream
from hume.core.api_error import ApiError
from pyee.asyncio import AsyncIOEventEmitter
from hume.empathic_voice.chat.types import SubscribeEvent
from hume.empathic_voice.chat.socket_client import ChatWebsocketConnection


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

            emotions = {
                "happiness": sum(
                    [
                        scores.amusement,
                        scores.aesthetic_appreciation,
                        scores.contentment,
                        scores.ecstasy,
                        scores.excitement,
                        scores.interest,
                        scores.joy,
                        scores.pride,
                        scores.relief,
                        scores.satisfaction,
                        scores.triumph,
                        scores.realization,
                        scores.surprise_positive,
                    ]
                ),
                "love": sum(
                    [
                        scores.admiration,
                        scores.adoration,
                        scores.awe,
                        scores.love,
                        scores.romance,
                        scores.sympathy,
                    ]
                ),
                "fear": sum(
                    [scores.fear, scores.horror, scores.surprise_negative]
                ),
                "sadness": sum(
                    [
                        scores.disappointment,
                        scores.distress,
                        scores.guilt,
                        scores.nostalgia,
                        scores.sadness,
                        scores.shame,
                        scores.tiredness,
                        scores.empathic_pain,
                        scores.pain,
                    ]
                ),
                "anger": sum(
                    [
                        scores.anger,
                        scores.contempt,
                        scores.disgust,
                        scores.envy,
                    ]
                ),
                "discomfort": sum(
                    [
                        scores.doubt,
                        scores.anxiety,
                        scores.awkwardness,
                        scores.embarrassment,
                        scores.confusion,
                        scores.boredom,
                    ]
                ),
                "concentration": sum(
                    [
                        scores.calmness,
                        scores.concentration,
                        scores.contemplation,
                    ]
                ),
                "desire": sum([scores.craving, scores.desire]),
            }

            self.emit("_assistant_message", emotions)

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
