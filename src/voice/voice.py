import time
import base64
import asyncio
import numpy as np
import sounddevice as sd
from typing import cast, Any
from openai import AsyncOpenAI
from pydub import AudioSegment
from pyee.asyncio import AsyncIOEventEmitter
from .audio import CHANNELS, SAMPLE_RATE, AudioPlayerAsync
from openai.resources.beta.realtime.realtime import AsyncRealtimeConnection


import io
import base64


import websockets
import json
import base64
from typing import Optional


import asyncio


class Voice(AsyncIOEventEmitter):
    def __init__(
        self,
        api_key,
        secret_key,
        openai_api_key,
        config_id,
        microphone_id=None,
    ):
        super().__init__()
        # OpenAI client
        self.client = AsyncOpenAI(api_key=openai_api_key)

        # Hume credentials and config
        self.hume_api_key = api_key
        self.hume_websocket: Optional[websockets.WebSocketClientProtocol] = (
            None
        )

        self.microphone_id = microphone_id
        self.connection = None
        self.session = None
        self.audio_player = AudioPlayerAsync()
        self.should_send_audio = asyncio.Event()
        self.connected = asyncio.Event()
        self.last_audio_item_id = None

        # Buffer for collecting OpenAI audio chunks
        self.audio_buffer = io.BytesIO()
        self.last_hume_send = 0

        # Lock for WebSocket recv
        self.recv_lock = asyncio.Lock()

    async def process_hume_emotions(self, audio_data):
        """Process audio through Hume WebSocket API and emit emotion events"""
        try:
            # Connect to WebSocket if not already connected
            if not self.hume_websocket:
                self.hume_websocket = await websockets.connect(
                    "wss://api.hume.ai/v0/stream/models",
                    additional_headers={"X-Hume-Api-Key": self.hume_api_key},
                )

            audio = AudioSegment(
                data=audio_data,
                sample_width=2,  # 16-bit audio
                frame_rate=24000,  # 24kHz sample rate
                channels=1,  # Mono audio
            )

            # Export to WAV format in memory
            wav_buffer = io.BytesIO()
            audio.export(wav_buffer, format="wav")
            wav_data = wav_buffer.getvalue()

            # Inside process_hume_emotions method, after creating wav_data:
            # wav_buffer.seek(0)
            # timestamp = int(time.time())
            # with open(f"debug_audio_{timestamp}.wav", "wb") as f:
            #     f.write(wav_data)
            # print(f"Saved debug audio to debug_audio_{timestamp}.wav")

            # Encode WAV data to base64
            encoded_data = base64.b64encode(wav_data).decode("utf-8")

            # Prepare the message
            message = {
                "models": {"prosody": {}},
                "raw_text": False,
                "data": encoded_data,
            }

            # Send the message
            await self.hume_websocket.send(json.dumps(message))

            # Receive the response
            async with self.recv_lock:
                response = await self.hume_websocket.recv()
            result = json.loads(response)

            print("received result")
            print(result)

            if result and "prosody" in result:
                emotions = result["prosody"][0]["emotions"]
                # Get the highest scoring emotion
                top_emotion = max(emotions, key=lambda x: x["score"])
                # self.emit("_assistant_message", top_emotion["name"])

        except Exception as e:
            print(f"Error processing emotions: {e}")
            # Try to close the connection on error
            if self.hume_websocket:
                await self.hume_websocket.close()
                self.hume_websocket = None

    async def run(self):
        asyncio.create_task(self.send_mic_audio())

        async with self.client.beta.realtime.connect(
            model="gpt-4o-mini-realtime-preview"
        ) as conn:
            self.connection = conn
            self.connected.set()
            self.should_send_audio.clear()

            async for event in conn:
                # print(f"Received event type: {event.type}")

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

                if event.type == "response.audio.delta":
                    print("response.audio.delta")
                    self.should_send_audio.clear()

                    if event.item_id != self.last_audio_item_id:
                        self.audio_player.reset_frame_count()
                        self.last_audio_item_id = event.item_id
                        self.audio_buffer = io.BytesIO()

                    # Decode audio data
                    bytes_data = base64.b64decode(event.delta)

                    # Add to buffer
                    self.audio_buffer.write(bytes_data)
                    current_time = time.time()

                    # Every 5 seconds, process emotions
                    if current_time - self.last_hume_send >= 4:
                        buffer_data = self.audio_buffer.getvalue()
                        if buffer_data:
                            asyncio.create_task(
                                self.process_hume_emotions(buffer_data)
                            )
                            self.audio_buffer = io.BytesIO()
                            self.last_hume_send = current_time

                    # Play audio immediately
                    self.audio_player.add_data(bytes_data)
                    continue

                if event.type == "response.done":
                    print("response.done")
                    # Process any remaining buffered audio
                    buffer_data = self.audio_buffer.getvalue()
                    if buffer_data:
                        asyncio.create_task(
                            self.process_hume_emotions(buffer_data)
                        )
                    self.audio_buffer = io.BytesIO()

                    while (
                        self.audio_player.queue and self.audio_player.playing
                    ):
                        await asyncio.sleep(0.1)
                    self.should_send_audio.set()
                    self.emit("_assistant_message_end")
                    continue

                if event.type == "error":
                    print(event.error)

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
