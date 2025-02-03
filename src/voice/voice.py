import asyncio

from hume import AsyncHumeClient, MicrophoneInterface
from .websocket import WebSocketHandler
from pyee.asyncio import AsyncIOEventEmitter
from hume.empathic_voice.chat.socket_client import ChatConnectOptions


class Voice(AsyncIOEventEmitter):
    """Manages voice interaction using Hume AI's Empathic Voice Interface.

    Handles real-time conversation and speech synthesis using the Hume AI API.
    Connects to a WebSocket for bi-directional audio streaming and emotion analysis.

    Events:
        _assistant_message: Emitted when assistant speaks with emotion data:
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
        client (AsyncHumeClient): Async client for Hume AI API
        options (ChatConnectOptions): WebSocket connection configuration
        microphone_id (int): ID of input microphone device
        websocket_handler (WebSocketHandler): Handles WebSocket events and audio streaming
    """

    def __init__(self, api_key, secret_key, config_id, microphone_id=1):
        """Initialize the voice engine and set up websockets for Hume AI

        Args:
            api_key (str): Hume AI API key
            voice_id (str): ID of the voice to use for synthesis
        """

        super().__init__()
        self.client = AsyncHumeClient(api_key=api_key)
        self.options = ChatConnectOptions(
            config_id=config_id, secret_key=secret_key
        )
        self.microphone_id = microphone_id

        self.websocket_handler = WebSocketHandler()
        self.websocket_handler.on(
            "_assistant_message",
            lambda emotion: self.emit("_assistant_message", emotion),
        )
        self.websocket_handler.on(
            "_assistant_message_end",
            lambda _: self.emit("_assistant_message_end"),
        )

    async def run(self):
        """Starts the voice interaction system with Hume AI.

        Creates a WebSocket connection to Hume AI's empathic voice service and
        initializes a microphone stream for real-time audio input. The connection
        handles bi-directional communication:
        - Sending audio from the microphone to Hume AI
        - Receiving speech synthesis and emotion analysis responses

        The method runs until interrupted by the user or an error occurs.

        Raises:
            ConnectionError: If WebSocket connection fails
            RuntimeError: If microphone initialization fails
        """

        async with self.client.empathic_voice.chat.connect_with_callbacks(
            options=self.options,
            on_open=self.websocket_handler.on_open,
            on_message=self.websocket_handler.on_message,
            on_close=self.websocket_handler.on_close,
            on_error=self.websocket_handler.on_error,
        ) as socket:
            self.websocket_handler.set_socket(socket)

            microphone_task = asyncio.create_task(
                MicrophoneInterface.start(
                    socket,
                    device=self.microphone_id,
                    allow_user_interrupt=True,
                    byte_stream=self.websocket_handler.byte_strs,
                )
            )

            await microphone_task
