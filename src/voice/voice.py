import base64
import asyncio
import numpy as np
import sounddevice as sd
from typing import cast, Any
from openai import AsyncOpenAI
from pyee.asyncio import AsyncIOEventEmitter
from hume import AsyncHumeClient, MicrophoneInterface
from .audio import CHANNELS, SAMPLE_RATE, AudioPlayerAsync
from openai.resources.beta.realtime.realtime import AsyncRealtimeConnection


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

    def __init__(
        self,
        api_key,
        secret_key,
        openai_api_key,
        config_id,
        microphone_id=None,
    ):
        """Initialize the voice engine and set up websockets for Hume AI

        Args:
            api_key (str): Hume AI API key
            voice_id (str): ID of the voice to use for synthesis
        """

        super().__init__()
        self.client = AsyncOpenAI(
            api_key=openai_api_key,
        )
        self.microphone_id = microphone_id
        self.connection = None
        self.session = None
        self.audio_player = AudioPlayerAsync()
        self.should_send_audio = asyncio.Event()
        self.connected = asyncio.Event()
        self.last_audio_item_id = None

    async def run(self):
        asyncio.create_task(self.send_mic_audio())

        async with self.client.beta.realtime.connect(
            model="gpt-4o-mini-realtime-preview"
        ) as conn:
            self.connection = conn
            self.connected.set()
            self.should_send_audio.clear()  # Start with audio disabled

            async for event in conn:
                print(f"Received event type: {event.type}")

                if event.type == "session.created":
                    print("session.created")
                    self.session = event.session
                    self.should_send_audio.set()

                    await conn.session.update(
                        session={
                            "voice": "ash",
                            "turn_detection": {
                                "type": "server_vad",
                                "threshold": 0.7,
                            },
                        }
                    )
                    continue

                if event.type == "session.updated":
                    print("session.updated")
                    self.session = event.session
                    continue

                if event.type == "response.audio.delta":
                    print("response.audio.delta")
                    self.should_send_audio.clear()

                    if event.item_id != self.last_audio_item_id:
                        self.audio_player.reset_frame_count()
                        self.last_audio_item_id = event.item_id

                    bytes_data = base64.b64decode(event.delta)
                    self.audio_player.add_data(bytes_data)
                    continue

                if event.type == "response.done":
                    print("response.done")

                    while (
                        self.audio_player.queue and self.audio_player.playing
                    ):
                        await asyncio.sleep(0.1)
                    self.should_send_audio.set()
                    continue

                if event.type == "error":
                    print(event.error)
                    continue

    async def _get_connection(self) -> AsyncRealtimeConnection:
        await self.connected.wait()
        assert self.connection is not None
        return self.connection

    async def send_mic_audio(self) -> None:
        sent_audio = False
        read_size = int(SAMPLE_RATE * 0.1)

        stream = sd.InputStream(
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            dtype="int16",
            blocksize=read_size,
            latency="low",
        )

        stream.start()

        try:
            while True:
                await asyncio.sleep(0.05)

                if not self.should_send_audio.is_set():
                    continue

                frames_available = stream.read_available
                if frames_available >= read_size:
                    data, _ = stream.read(read_size)

                    connection = await self._get_connection()

                    if not sent_audio:
                        await connection.send({"type": "response.cancel"})
                        sent_audio = True

                    await connection.input_audio_buffer.append(
                        audio=base64.b64encode(cast(Any, data)).decode("utf-8")
                    )

        except Exception as e:
            print(f"Error in audio capture: {e}")
        finally:
            stream.stop()
            stream.close()
